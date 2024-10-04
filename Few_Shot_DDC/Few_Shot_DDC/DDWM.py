import os
import time
import json
import pickle
import itertools
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn
from torch.optim import LBFGS
from torch.distributions.multivariate_normal import MultivariateNormal

from FSLTask import FSLTaskMaker
from utils.io_utils import DataWriter, logger
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 对输入的张量进行L2范数归一化
def normalize_l2(x, dim=-1): 
    '''x.shape = (batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim)'''
    x_norm = torch.linalg.norm(x, dim=dim, keepdims=True)
    x = torch.div(x, x_norm)
    return x

# 使用LBFGS优化器进行逻辑回归分类
def torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, firth_c=0.0, max_iter=1000, verbose=True):
    batch_dim, n_samps, n_dim = X_aug.shape
    assert Y_aug.shape == (batch_dim, n_samps)
    num_classes = Y_aug.unique().numel()

    device = X_aug.device
    tch_dtype = X_aug.dtype
    # default value from https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    # from scipy.minimize.lbfgsb. In pytorch, it is the equivalent "max_iter"
    # (note that "max_iter" in torch.optim.LBFGS is defined per epoch and a step function call!)
    max_corr = 10
    tolerance_grad = 1e-05
    tolerance_change = 1e-09
    line_search_fn = 'strong_wolfe'
    l2_c = 1.0
    use_bias = True

    # According to https://github.com/scipy/scipy/blob/master/scipy/optimize/_lbfgsb_py.py#L339
    # wa (i.e., the equivalenet of history_size) is 2 * m * n (where m is max_corrections and n is the dimensions).
    history_size = max_corr * 2  # since wa is O(2*m*n) in size

    num_epochs = max_iter // max_corr  # number of optimization steps
    max_eval_per_epoch = None  # int(max_corr * max_evals / max_iter) matches the 15000 default limit in scipy!

    W = torch.nn.Parameter(torch.zeros((batch_dim, n_dim, num_classes), device=device, dtype=tch_dtype))
    opt_params = [W]
    linlayer = lambda x_: x_.matmul(W)
    if use_bias:
        bias = torch.nn.Parameter(torch.zeros((batch_dim, 1, num_classes), device=device, dtype=tch_dtype))
        opt_params.append(bias)
        linlayer = lambda x_: (x_.matmul(W) + bias)

    optimizer = LBFGS(opt_params, lr=1, max_iter=max_corr, max_eval=max_eval_per_epoch,
                      tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                      history_size=history_size, line_search_fn=line_search_fn)

    Y_aug_i64 = Y_aug.to(device=device, dtype=torch.int64)
    for epoch in range(num_epochs):
        if verbose:
            running_loss = 0.0

        inputs_, labels_ = X_aug, Y_aug_i64

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            batch_dim_, n_samps_, n_dim_ = inputs_.shape
            outputs_ = linlayer(inputs_)
            # outputs_.shape -> batch_dim, n_samps, num_classes
            logp = outputs_ - torch.logsumexp(outputs_, dim=-1, keepdims=True)
            # logp.shape -> batch_dim, n_samps, num_classes
            label_logps = -logp.gather(dim=-1, index=labels_.reshape(batch_dim_, n_samps_, 1))
            # label_logps.shape -> batch_dim, n_samps, 1
            loss_cross = label_logps.mean(dim=(-1, -2)).sum(dim=0)
            loss_firth = -logp.mean(dim=(-1, -2)).sum(dim=0)
            loss_l2 = 0.5 * torch.square(W).sum() / n_samps_
            loss = loss_cross + firth_c * loss_firth + l2_c * loss_l2
            loss = loss / batch_dim_
            if loss.requires_grad:
                loss.backward()
            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        if verbose:
            loss = closure()
            running_loss += loss.item()
            logger(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")
    return linlayer

# 计算基于查询样本的分布拟合结果
def Distribution_fitting_with_DDWM(query, base_means, base_means_matrix, base_cov, k, alpha, gamma):
    assert torch.is_tensor(query)
    assert torch.is_tensor(base_means)
    assert torch.is_tensor(base_cov)
    # print("query------------------------",query)
    # print("query.shape------------------------",query.shape)
    # print("base_means_matrix.shape------------------------",base_means_matrix.shape)

    batch_dims, n_dim = query.shape[:-1], query.shape[-1]
    
    batch_dim = int(np.prod(batch_dims))
    # print("batch_dim------------------------",batch_dim)
    # print("n_dim------------------------",n_dim)
    n_classes = base_means.shape[0]
    assert base_means.shape == (n_classes, n_dim)
    assert base_cov.shape == (n_classes, n_dim, n_dim)

    base_means = base_means.unsqueeze(0).expand(batch_dim, n_classes, n_dim)
    base_cov = base_cov.unsqueeze(0).expand(batch_dim, n_classes, n_dim, n_dim)
    # query      -> shape = (batch_dim, n_dim)
    # base_means -> shape = (batch_dim, n_classes, n_dim)
    # base_cov   -> shape = (batch_dim, n_classes, n_dim, n_dim)
    # --- Calculate the feature description matrix of support samples --- #
    query_matrix = torch.matmul(query.reshape(batch_dim, 1, n_dim, 1),
                                query.reshape(batch_dim, 1, n_dim, 1).permute(0, 1, 3, 2))
    # print("query_matrix.shape------------------------",query_matrix.shape)
    
    # --- Calculate Frobenius norm values of the difference and Select k nearest base classes--- #
    matrix_L2_dist =  torch.linalg.norm(query_matrix - base_means_matrix, ord='fro', dim=(2, 3))
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices  # index.shape == ( , k)

    # --- Calculate weight factors of k nearest base classes --- #
    dist = torch.linalg.norm(query.reshape(batch_dim, 1, n_dim) - base_means, 2,
                             dim=-1)  # dist.shape == (batch_dim, n_classes)
    Weight = torch.div(1, torch.pow(1 + dist, gamma))
    query_matrix = query_matrix.cpu()
    matrix_L2_dist = matrix_L2_dist.cpu()
    torch.cuda.empty_cache()
    gather_weight = torch.gather(Weight, dim=-1, index=index).unsqueeze(-1).reshape(batch_dim, k, 1)
    assert gather_weight.shape == (batch_dim, k, 1)

    # --- Calculate the weighted mean and Covariance of base classes --- #
    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(batch_dim, k, n_dim))
    assert gathered_mean.shape == (batch_dim, k, n_dim)
    gathered_cov = torch.gather(base_cov, dim=-3, index=index.unsqueeze(-1).unsqueeze(-1).expand(batch_dim, k, n_dim, n_dim))
    assert gathered_cov.shape == (batch_dim, k, n_dim, n_dim)

    Weight_gathered_mean = torch.matmul(gathered_mean.permute(0, 2, 1), gather_weight).reshape(batch_dim, n_dim)
    assert Weight_gathered_mean.shape == ((batch_dim, n_dim))
    Weight_gathered_cov = torch.sum(gathered_cov * gather_weight.reshape(batch_dim, k, 1, 1), dim=1)
    assert Weight_gathered_cov.shape == (batch_dim, n_dim, n_dim)

    gathered_mean = gathered_mean.cpu()
    gathered_cov = gathered_cov.cpu()
    torch.cuda.empty_cache()

    # --- Calculate the mean and the covariance of the learned feature distribution --- #
    learned_mean = torch.div((Weight_gathered_mean + query.reshape(batch_dim, n_dim)),
                                torch.sum(gather_weight, dim=1) + 1)
    assert learned_mean.shape == (batch_dim, n_dim) # learned_mean.shape == (batch_dim, n_dim)

    learned_cov = torch.div(Weight_gathered_cov + alpha, torch.sum(gather_weight, dim=1).reshape(batch_dim, 1, 1) + 1)
    learned_cov = learned_cov + 1e-6 * torch.eye(n_dim).unsqueeze(0).expand(batch_dim, n_dim, n_dim).to(
        device='cuda:0', dtype=torch.float32)
    assert learned_cov.shape == (batch_dim, n_dim, n_dim)  # learned_cov.shape == (batch_dim, n_dim, n_dim)

    Weight_gathered_mean = Weight_gathered_mean.cpu()
    Weight_gathered_cov = Weight_gathered_cov.cpu()
    gather_weight = gather_weight.cpu()
    torch.cuda.empty_cache()

    return learned_mean.reshape(*batch_dims, n_dim), learned_cov.reshape(*batch_dims, n_dim, n_dim)

def main(config_dict):
    config_id = config_dict['config_id']
    device_name = config_dict['device_name']
    rng_seed = config_dict['rng_seed']
    n_tasks = config_dict['n_tasks']   # 10000 这个参数要在5以上，更改这个参数并不会让显存占用变小
    source_dataset = config_dict['source_dataset']
    target_dataset = config_dict['target_dataset']
    n_shots_list = config_dict['n_shots_list']
    n_ways_list = config_dict['n_ways_list']
    split_name_list = config_dict['split_list']
    firth_coeff_list = config_dict['firth_coeff_list']
    n_query = config_dict['n_query']
    dc_tukey_lambda = config_dict['dc_tukey_lambda']
    gm = config_dict['gamma']
    n_aug_list = config_dict['n_aug_list']
    dc_k = config_dict['dc_k']
    dc_alpha = config_dict['dc_alpha']
    backbone_arch = config_dict['backbone_arch']
    backbone_method = config_dict['backbone_method']
    lbfgs_iters = config_dict['lbfgs_iters']
    store_results = config_dict['store_results']
    results_dir = config_dict['results_dir']
    features_dir = config_dict['features_dir']
    cache_dir = config_dict['cache_dir']
    dump_period = config_dict['dump_period']
    torch_threads = config_dict['torch_threads']
    task_bs = 1 # The number of tasks to stack to each other for parallel optimization  任务数是k-way n-shot里面的way

    dsname2abbrv = {'miniImagenet': 'mini', 'CUB': 'cub', 'CIFAR-FS': 'cifar'}

    data_writer = None
    if store_results:
        assert results_dir is not None, 'Please provide results_dir in the config_dict.'
        Path(results_dir).mkdir(parents=True, exist_ok=True)  # 这行代码用于创建存储结果的目录
        data_writer = DataWriter(dump_period=dump_period) # 参数 dump_period，用于设置结果写入的周期，即多少次迭代后将结果写入文件。

    tch_dtype = torch.float32
    untouched_torch_thread = torch.get_num_threads()
    if torch_threads:
        torch.set_num_threads(torch_threads)
    # 这是一个嵌套循环，它使用 itertools.product 生成了一系列配置的组合
    for setting in itertools.product(firth_coeff_list, n_ways_list, n_shots_list, n_aug_list, split_name_list):
        firth_coeff, n_ways, n_shots, n_aug, split = setting
        os.makedirs(results_dir, exist_ok=True)  # 在每次迭代时，这行代码创建存储实验结果的目录
        np.random.seed(999)
        torch.manual_seed(rng_seed)
        # 创建一个有序字典，用于存储当前配置的各个参数。
        config_cols_dict = OrderedDict(n_shots=n_shots, n_ways=n_ways, source_dataset=source_dataset,
                                       target_dataset=target_dataset, backbone_arch=backbone_arch,
                                       backbone_method=backbone_method, n_aug=n_aug, split=split,
                                       firth_coeff=firth_coeff, n_query=n_query,
                                       dc_tukey_lambda=dc_tukey_lambda, gamma=gm, dc_k=dc_k,
                                       dc_alpha=dc_alpha, lbfgs_iters=lbfgs_iters,
                                       rng_seed=rng_seed)
        print('-' * 80)
        logger('Current configuration:')
        for (cfg_key_, cfg_val_) in config_cols_dict.items(): # 打印所有参数
            logger(f"  --> {cfg_key_}: {cfg_val_}", flush=True)

        task_maker = FSLTaskMaker()
        task_maker.reset_global_vars()

        features_bb_dir = f"{features_dir}/{backbone_arch}_{backbone_method}/{source_dataset}"
        Path(features_bb_dir).mkdir(parents=True, exist_ok=True)
        task_maker.loadDataSet(f'{dsname2abbrv[source_dataset]}_{split}_features',features_dir=features_bb_dir)
        logger("* Target Dataset loaded", flush=True)

        n_lsamples = n_ways * n_shots
        n_usamples = n_ways * n_query
        n_samples = n_lsamples + n_usamples

        cfg = {'n_shots': n_shots, 'n_ways': n_ways, 'n_query': n_query, 'seed': rng_seed}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        task_maker.setRandomStates(cfg, cache_dir=cache_dir)
        ndatas = task_maker.GenerateRunSet(end=n_tasks, cfg=cfg) # 这段代码的主要功能是生成每个任务所需的数据集，其中包括支持集和查询集，用于元学习模型的训练和测试。
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_tasks, n_samples, -1) # 进行维度重排，其参数指定了新的维度顺序。
        labels = torch.arange(n_ways).view(1, 1, n_ways)
        labels = labels.expand(n_tasks, n_shots + n_query, n_ways)
        labels = labels.clone().view(n_tasks, n_samples)

        # ---- Base class statistics 基类数据统计
        base_means = []
        base_cov = []
        # base_features_path = f"{features_dir}/{backbone_arch}_{backbone_method}/{src_ds_abbrv}2{src_ds_abbrv}_base.pkl"
        base_features_path = f"{features_dir}/{backbone_arch}_{backbone_method}/{source_dataset}/{dsname2abbrv[source_dataset]}_base_features.plk"
        logger(f"* Reading Base Features from {base_features_path}", flush=True)
        with open(base_features_path, 'rb') as fp:
            data = pickle.load(fp)
            for key in data.keys():
                feature = np.array(data[key])
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)
        logger("* Means and Covariance Matrices are calculated", flush=True)

        # --- Calculate feature description matrices of each base class ---# 计算特征描述矩阵
        with torch.no_grad():
            base_means = torch.cat(
                [torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype) for x in base_means])
            base_cov = torch.cat([torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype)
                for x in base_cov])
            
            base_means_matrix = torch.matmul(base_means.unsqueeze(-1).expand(task_bs * n_ways_list[0] * n_shots_list[0], base_means.shape[0], base_means.shape[1], 1),
                                         base_means.unsqueeze(-1).expand(task_bs * n_ways_list[0] * n_shots_list[0], base_means.shape[0], base_means.shape[1], 1).permute(0, 1, 3, 2)).to(device=device_name, dtype=tch_dtype)

        # ---- classification for each task 每个任务进行分类
        test_acc_list = []
        logger(f'* Starting Classification for {n_tasks} Tasks...')
        st_time = time.time()

        all_run_idxs = np.arange(n_tasks)
        all_run_idxs = all_run_idxs.reshape(-1, task_bs)

        n_dim = ndatas.shape[-1]
        for ii, run_idxs in enumerate(all_run_idxs):  # 用于执行元学习中的任务
            run_idxs = run_idxs.astype(int).tolist()
            batch_dim = len(run_idxs)

            # 获取支持集和查询集数据：
            support_data = ndatas[run_idxs][:, :n_lsamples, :].to(device=device_name, dtype=tch_dtype)
            assert support_data.shape == (batch_dim, n_lsamples, n_dim)

            support_label = labels[run_idxs][:, :n_lsamples].to(device=device_name, dtype=torch.int64)
            assert support_label.shape == (batch_dim, n_lsamples)

            query_data = ndatas[run_idxs][:, n_lsamples:, :].to(device=device_name, dtype=tch_dtype)
            assert query_data.shape == (batch_dim, n_usamples, n_dim)

            query_label = labels[run_idxs][:, n_lsamples:].to(device=device_name, dtype=torch.int64)
            assert query_label.shape == (batch_dim, n_usamples)

            # ----Transform support sets and query sets with Tukey's Ladder of Power transformation ----# 对支持集和查询集的特征数据进行 Tukey's Ladder of Power 转换，即对特征数据进行幂次操作。
            support_data = torch.pow(support_data, dc_tukey_lambda)
            query_data = torch.pow(query_data, dc_tukey_lambda)

            # ---- distribution calibration and feature sampling  分布校准和特征采样：
            num_sampled = int(n_aug / n_shots)

            with torch.no_grad():
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(support_data, base_means, base_means_matrix, base_cov,
                                                                 alpha=dc_alpha, k=dc_k, gamma=gm)
            assert mean_tch.shape == (batch_dim, n_lsamples, n_dim)
            assert cov_tch.shape == (batch_dim, n_lsamples, n_dim, n_dim)

            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time,))
                    # norm_samps_tch.shape -> (samps_at_a_time, batch_dim, n_lsamples, n_dim)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                # sampled_data.shape -> (num_sampled, batch_dim, n_lsamples, n_dim)
                assert sampled_data.shape == (num_sampled, batch_dim, n_lsamples, n_dim)
                sampled_data = sampled_data.permute(1, 2, 0, 3)
                assert sampled_data.shape == (batch_dim, n_lsamples, num_sampled, n_dim)
                # time_lst_gen.append(time.time() - start_time)

            with torch.no_grad():
                sampled_label__ = support_label.unsqueeze(-1)
                sampled_label_ = sampled_label__.expand(batch_dim, n_lsamples, num_sampled)
                sampled_label = sampled_label_.reshape(batch_dim, n_lsamples * num_sampled)
                sampled_data = sampled_data.reshape(batch_dim, n_lsamples * num_sampled, n_dim)
                X_aug = normalize_l2(torch.cat([support_data, sampled_data], dim=-2))
                # X_aug.shape -> batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim
                Y_aug = torch.cat([support_label, sampled_label], dim=-1)
                # Y_aug.shape -> batch_dim, n_lsamples + n_lsamples*num_sampled


            #### 这边计算T-SNE图 ##################################

                features = X_aug
                labels = Y_aug

                # 调整特征向量和标签的形状
                features = features.view(-1, 640)  # 将形状从 [N, 5, 640] 转为 [N*5, 640]
                features = features.cpu()
                labels = labels.cpu()
                labels = np.repeat(labels, 1)  # 将每个标签重复 1 次以匹配新的特征向量形状
                print("labels.shape", labels.shape)

                # 转换为 numpy 数组
                features_np = features.numpy()
                labels_np = labels.numpy()

                # 使用 T-SNE 进行降维
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features_np)

                # 定义标签颜色字典 (按 0-255 的 RGB 值)
                custom_colors_255 = {
                    0: (234, 234, 254),    # Red
                    1: (252, 245, 158),    # Blue
                    2: (132, 175, 152),    # Green
                    3: (152, 173, 214),  # Purple
                    4: (254, 193, 193),  # Orange
                    # 根据需要添加更多标签和颜色
                }

                # 将 0-255 的 RGB 值转换为 0-1 范围的浮点数
                custom_colors = {label: (r/255, g/255, b/255) for label, (r, g, b) in custom_colors_255.items()}

                unique_labels = np.unique(labels_np)

                # 检查是否有比颜色字典更多的标签，如果有，需要扩展颜色字典
                if len(unique_labels) > len(custom_colors):
                    raise ValueError("Number of unique labels exceeds the number of custom colors defined.")

                # 可视化
                plt.figure(figsize=(10, 8))

                # 绘制所有特征
                for unique_label in unique_labels:
                    plt.scatter(
                        features_2d[labels_np == unique_label, 0],
                        features_2d[labels_np == unique_label, 1],
                        color=custom_colors[unique_label],  # 从颜色字典中获取颜色
                        label=f'Class {unique_label}',
                        alpha=0.6,
                        marker='^'  # 使用不同的标记
                    )



                # 定义标签颜色字典 (按 0-255 的 RGB 值)  5个单独的颜色
                custom_colors_255_support = {
                    0: (179, 179, 255),    # Red
                    1: (248, 231, 16),    # Blue
                    2: (64, 131, 94),    # Green
                    3: (0, 51, 153),  # Purple
                    4: (255, 102, 102),  # Orange
                    # 根据需要添加更多标签和颜色
                }

                # 将 0-255 的 RGB 值转换为 0-1 范围的浮点数
                custom_colors_support = {label: (r/255, g/255, b/255) for label, (r, g, b) in custom_colors_255_support.items()}


                # 将前五个特征用三角形标记，颜色与其标签颜色一致
                for i in range(5):
                    plt.scatter(
                        features_2d[i, 0],  # 前五个特征的 X 坐标
                        features_2d[i, 1],  # 前五个特征的 Y 坐标
                        color=custom_colors_support[labels_np[i]],  # 使用标签对应的颜色
                        alpha=1.0,
                        marker='*',  # 使用三角形标记
                        s=100  # 调整标记大小
                    )

                # plt.legend()
                # plt.title('T-SNE Visualization of Image Features')
                # plt.xlabel('TSNE Component 1')
                # plt.ylabel('TSNE Component 2')
                plt.savefig('tsne_visualization.png', dpi=300)
                plt.show()



            #### 这边计算T-SNE图 ##################################






            # ---- train classifier  使用支持集数据训练逻辑回归分类器，得到一个分类模型
            classifier = torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, firth_coeff,
                                                        max_iter=lbfgs_iters, verbose=False)
           
            # 计算准确率并输出信息---------
            with torch.no_grad():  
                query_data = normalize_l2(query_data)
                predicts = classifier(query_data).argmax(dim=-1)
                # predicts.shape -> batch_dim, n_usamples

            with torch.no_grad():
                acc = (predicts == query_label).double().mean(dim=(-1)).detach().cpu().numpy().ravel()
            test_acc_list += acc.tolist()

            runs_so_far = len(test_acc_list)

            if (ii + 1) % 2 == 0:
                time_per_iter = (time.time() - st_time) / runs_so_far
                acc_mean = 100 * np.mean(test_acc_list)
                acc_ci = 1.96 * 100.0 * float(np.std(test_acc_list) / np.sqrt(len(test_acc_list)))
                print('.' * acc.size + f' (Accuracy So Far: {acc_mean:.2f} +/- {acc_ci:.2f},    ' +
                      f'{time_per_iter:.3f} sec/iter,    {runs_so_far:05d}/{n_tasks:05d} Tasks Done)',
                      flush=True)
            else:
                logger('.' * acc.size, end='', flush=True)

        tam = 100.0 * float(np.mean(test_acc_list))
        tac = 1.96 * 100.0 * float(np.std(test_acc_list) / np.sqrt(len(test_acc_list)))
        logger(f' --> Final Accuracy: {tam:.2f} +/- {tac:.2f}' + '%', flush=True)

        if store_results:
            csv_path = f'{results_dir}/{config_id}.csv'
            for task_id, task_acc in enumerate(test_acc_list):
                row_dict = config_cols_dict.copy()  # shallow copy
                row_dict['task_id'] = task_id
                row_dict['test_acc'] = task_acc
                row_dict['test_accuracy_mean'] = tam
                data_writer.add(row_dict, csv_path)

    if store_results:
        # We need to make a final dump before exiting to make sure all data is stored
        data_writer.dump()

    torch.set_num_threads(untouched_torch_thread)


if __name__ == '__main__':
    import argparse
    my_parser = argparse.ArgumentParser()
    # my_parser.add_argument('--configid', default='5ways/miniImagenet_1s5w', type=str)  # 5-way 1-shot
    my_parser.add_argument('--configid', default='5ways/CUB_1s5w', type=str)
    # my_parser.add_argument('--configid', default='5ways/CIFAR-FS_1s5w', type=str)

    my_parser.add_argument('--device', default='cuda:0', type=str)
    args = my_parser.parse_args()
    args_configid = args.configid
    args_device_name = args.device
 
    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        my_config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ''
    print(config_tree)
    PROJPATH = os.getcwd()  # 获取当前工作目录的路径
    cfg_dir = f'{PROJPATH}/configs'
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = f'{PROJPATH}/configs/{config_tree}/{my_config_id}.json' # miniImagenet_1s5w.json
    logger(f'Reading Configuration from {cfg_path}', flush=True)

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    # 添加一些配置
    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['device_name'] = args_device_name
    proced_config_dict['results_dir'] = f'{PROJPATH}/results/{config_tree}'
    proced_config_dict['cache_dir'] = f'{PROJPATH}/cache'
    proced_config_dict['features_dir'] = f'{PROJPATH}/features'

    main(proced_config_dict)
