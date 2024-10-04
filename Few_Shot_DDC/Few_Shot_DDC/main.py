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
from geomloss import SamplesLoss
import torch.nn.functional as F

 # 对输入的张量进行L2范数归一化
def normalize_l2(x, dim=-1): 
    '''x.shape = (batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim)'''
    x_norm = torch.linalg.norm(x, dim=dim, keepdims=True)
    x = torch.div(x, x_norm)
    return x

# 计算JS散度
def js_divergence(p, q):
    # 计算p和q的均值
    m = 0.5 * (p + q)
    # 计算KL散度
    kl_pm = F.kl_div(m.log(), p, log_target=True, reduction='none')
    kl_qm = F.kl_div(m.log(), q, log_target=True, reduction='none')
    # 计算JS散度
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div.sum(-1).sum(-1)  # 对最后两个维度求和以得到单个散度值


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


# 自己的计算基于查询样本的分布拟合结果 our 组合度量结果
def Distribution_DDC(query, base_means, base_means_matrix, base_cov, k, alpha, gamma, a, b, c, T): # base_means.shape=(64,640)
    assert torch.is_tensor(query)
    assert torch.is_tensor(base_means)
    assert torch.is_tensor(base_cov)
    # print("query------------------------",query)
    # print("query.shape------------------------",query.shape)
    # print("base_means_matrix.shape------------------------",base_means_matrix.shape)

    batch_dims, n_dim = query.shape[:-1], query.shape[-1]
    batch_dim = int(np.prod(batch_dims))
    base_means_matrix_class = base_means_matrix.shape[0] # 基类种类
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
    # query_matrix = torch.matmul(query.reshape(batch_dim, 1, n_dim, 1),
    #                             query.reshape(batch_dim, 1, n_dim, 1).permute(0, 1, 3, 2))
    
    query_matrix = torch.matmul(query.reshape(batch_dim, n_dim, 1),
                                query.reshape(batch_dim, n_dim, 1).permute(0, 2, 1))    
    # print("query_matrix.shape------------------------",query_matrix.shape)
    

    # 我需要改的-----------------------------------------------------------------------------------------------------------------

    # # --- Calculate  wasserstein_distance values --- # 
    # # 将数据reshape为合适的形状
    # base_means_flat = base_means_matrix.view(base_means_matrix_class, -1)
    # query_flat = query_matrix.view(batch_dim, -1)

    # # 初始化一个SamplesLoss对象用于计算Wasserstein距离
    # loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)

    # # 初始化一个矩阵用于存放Wasserstein距离值
    # wasserstein_matrix = torch.zeros(batch_dim, base_means_matrix_class)  # wasserstein_matrix = torch.zeros(5, 64)

    # # 计算Wasserstein距离矩阵
    # for i in range(batch_dim):
    #     for j in range(base_means_matrix_class):
    #         # 获取base_means_matrix中的一个特征图
    #         base_feat = base_means_flat[j].unsqueeze(0)
    #         # 获取query_matrix中的一个特征图
    #         query_feat = query_flat[i].unsqueeze(0)
    #         # 计算Wasserstein距离
    #         wasserstein_distance = loss(base_feat, query_feat)
    #         # 将距离值存入矩阵
    #         wasserstein_matrix[i, j] = wasserstein_distance

    # print("Wasserstein distance matrix:")  
    # # print(wasserstein_matrix)
 

    # --- Calculate  wasserstein_distance values 优化版本--- --- --- --- ---  # 
    # 将数据reshape为合适的形状
    base_means_flat = base_means_matrix.view(base_means_matrix_class, -1)
    query_flat = query_matrix.view(batch_dim, -1)

    # 初始化一个SamplesLoss对象用于计算Wasserstein距离
    loss = SamplesLoss(loss='energy', p=2, blur=0.005)

    # 初始化一个矩阵用于存放Wasserstein距离值
    wasserstein_matrix = torch.zeros(batch_dim, base_means_matrix_class)  # wasserstein_matrix = torch.zeros(5, 64)

    # 尝试使用广播机制计算所有距离
    # 注意：这需要确认SamplesLoss是否支持批量计算
    for i in range(batch_dim):
        wasserstein_matrix[i, :] = loss(query_flat[i].unsqueeze(0).repeat(base_means_matrix_class, 1), base_means_flat)
    # --- Calculate  wasserstein_distance values 优化版本--- --- --- --- ---  # 


    # --- Calculate Frobenius norm values of the difference and Select k nearest base classes--- #
    # 将 base_means_matrix 的形状调整为（1, 64, 640, 640）以便进行广播
    differences = query_matrix.unsqueeze(1) - base_means_matrix.unsqueeze(0)
    # 计算 Frobenius 范数，即每个元素的平方和的平方根
    matrix_L2_dist = torch.linalg.norm(differences, ord='fro', dim=(2, 3))
    # 输出距离矩阵的大小
    # print(matrix_L2_dist.size())  # 输出应该为 (5, 64)
    # --- Calculate Frobenius norm values of the difference and Select k nearest base classes--- #


    # --- Calculate  JS 散度 values ----- --- --- --- --- --- -- --- -- ---  # 
    # 归一化图像数据为概率分布
    query_probs = F.softmax(query_matrix.view(batch_dim, -1), dim=1).view(batch_dim, 1, 640, 640)   # .view(5, 1, 640, 640)  
    base_probs = F.softmax(base_means_matrix.view(base_means_matrix_class, -1), dim=1).view(1, base_means_matrix_class, 640, 640)   # view(1, 64, 640, 640) 
    # 广播使得每个query与每个base比较
    query_probs_expanded = query_probs.expand(batch_dim, base_means_matrix_class, 640, 640)
    base_probs_expanded = base_probs.expand(batch_dim, base_means_matrix_class, 640, 640)
    # 计算JS散度
    js_matrix = js_divergence(query_probs_expanded, base_probs_expanded)
    # print("Jensen-Shannon divergence matrix:")
    # print(js_matrix.shape)
    # print(js_matrix)
    # --- Calculate  JS 散度 values ----- --- --- --- --- --- -- --- -- ---  # 


    #  --- 组合度量  and Calculate adaptive weight---  ---  ---  ---  ---  ---  ---  --- - #
    # max-min normalization 找到最大值和最小值
    max_vals, _ = torch.max(wasserstein_matrix, dim=1, keepdim=True)
    min_vals, _ = torch.min(wasserstein_matrix, dim=1, keepdim=True)
    # 添加一个小的常数值，避免除以零的情况
    epsilon = 1e-8
    # 对每行进行归一化处理
    similarity_matrix_normalize_wasserstein = (wasserstein_matrix - min_vals) / (max_vals - min_vals + epsilon)

    # max-min normalization 找到最大值和最小值
    max_vals, _ = torch.max(matrix_L2_dist, dim=1, keepdim=True)
    min_vals, _ = torch.min(matrix_L2_dist, dim=1, keepdim=True)
    # 添加一个小的常数值，避免除以零的情况
    epsilon = 1e-8
    # 对每行进行归一化处理
    similarity_matrix_normalize_L2 = (matrix_L2_dist - min_vals) / (max_vals - min_vals + epsilon)

    # max-min normalization 找到最大值和最小值
    max_vals, _ = torch.max(js_matrix, dim=1, keepdim=True)
    min_vals, _ = torch.min(js_matrix, dim=1, keepdim=True)
    # 添加一个小的常数值，避免除以零的情况
    epsilon = 1e-8
    # 对每行进行归一化处理 
    similarity_matrix_normalize_JS = (js_matrix - min_vals) / (max_vals - min_vals + epsilon)

    similarity_matrix_normalize_wasserstein = similarity_matrix_normalize_wasserstein.cuda()
    similarity_matrix_normalize_L2 = similarity_matrix_normalize_L2.cuda()
    similarity_matrix_normalize_JS = similarity_matrix_normalize_JS.cuda()

    # 组合度量及其参数
    # Combined_matrix = 5*similarity_matrix_normalize_wasserstein + 1*similarity_matrix_normalize_L2
    # Combined_matrix = 2*similarity_matrix_normalize_wasserstein + 1*similarity_matrix_normalize_L2 + 1*similarity_matrix_normalize_JS ########################组合度量参数################################
 
    Combined_matrix = a*similarity_matrix_normalize_wasserstein + b*similarity_matrix_normalize_L2 + c*similarity_matrix_normalize_JS ########################组合度量参数################################
    # print("Combined_matrix-----------------", Combined_matrix)

    index = torch.topk(Combined_matrix, k, dim=-1, largest=False, sorted=True).indices


    # elements = Combined_matrix[torch.arange(Combined_matrix.size(0)).unsqueeze(1), index]
    # print("elements-----------------", elements)

    #  --- 组合度量  and Calculate adaptive weight---  ---  ---  ---  ---  ---  ---  --- - # 
    # # 标准归一化
    # similarity_matrix_normalize  = F.normalize(wasserstein_matrix, p=2, dim=1)  # 相似性矩阵归一化
    # print("similarity_matrix_normalize", similarity_matrix_normalize)

    query_matrix = query_matrix.cuda()
    Combined_matrix = Combined_matrix.cuda()
    index = index.cuda()
    torch.cuda.empty_cache()
    gather_weight = torch.gather(Combined_matrix, dim=-1, index=index).unsqueeze(-1).reshape(batch_dim, k, 1)
    gather_weight = 1-gather_weight
    # print("gather_weight_shift",gather_weight_shift)
    # print("gather_weight",gather_weight)

    # T = 0.4
    gather_weight[gather_weight < T ] = 0 # 自适应K值设定，T为相似性阈值，低于这个阈值的便不纳入考虑
    # print("gather_weight", gather_weight)
    assert gather_weight.shape == (batch_dim, k, 1)

    # --- Calculate the weighted mean and Covariance of base classes --- #  
    base_means = base_means.cuda()
    index = index.cuda()  
    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(batch_dim, k, n_dim))     
    assert gathered_mean.shape == (batch_dim, k, n_dim)

    gathered_cov = torch.gather(base_cov, dim=-3, index=index.unsqueeze(-1).unsqueeze(-1).expand(batch_dim, k, n_dim, n_dim))
    assert gathered_cov.shape == (batch_dim, k, n_dim, n_dim)

    gathered_mean,  gather_weight = gathered_mean.cuda(), gather_weight.cuda()
    Weight_gathered_mean = torch.matmul(gathered_mean.permute(0, 2, 1), gather_weight).reshape(batch_dim, n_dim)
    # print("Weight_gathered_mean",Weight_gathered_mean)
    assert Weight_gathered_mean.shape == ((batch_dim, n_dim))
    
    gathered_cov = gathered_cov.cuda()
    Weight_gathered_cov = torch.sum(gathered_cov * gather_weight.reshape(batch_dim, k, 1, 1), dim=1)
    assert Weight_gathered_cov.shape == (batch_dim, n_dim, n_dim)

    gathered_mean = gathered_mean.cpu()
    gathered_cov = gathered_cov.cpu()     
    torch.cuda.empty_cache()


    # --- Calculate the mean and the covariance of the learned feature distribution --- #
    # learned_mean = torch.div((Weight_gathered_mean + query.reshape(batch_dim, n_dim)),
    #                             torch.sum(gather_weight, dim=1) + 1)
    # 自适应K值后的均值计算
    learned_mean = torch.div((Weight_gathered_mean + query.reshape(batch_dim, n_dim)),
                                torch.count_nonzero(gather_weight, dim=1) + 1)
    assert learned_mean.shape == (batch_dim, n_dim) # learned_mean.shape == (batch_dim, n_dim)

    # learned_cov = torch.div(Weight_gathered_cov + alpha, torch.sum(gather_weight, dim=1).reshape(batch_dim, 1, 1) + 1)
    # 自适应K值后的方差计算
    learned_cov = torch.div(Weight_gathered_cov + alpha, torch.count_nonzero(gather_weight, dim=1).reshape(batch_dim, 1, 1) + 1)
    learned_cov = learned_cov + 1e-6 * torch.eye(n_dim).unsqueeze(0).expand(batch_dim, n_dim, n_dim).to(
        device='cuda:0', dtype=torch.float32)
    assert learned_cov.shape == (batch_dim, n_dim, n_dim)  # learned_cov.shape == (batch_dim, n_dim, n_dim)

    Weight_gathered_mean = Weight_gathered_mean.cpu()
    Weight_gathered_cov = Weight_gathered_cov.cpu()
    gather_weight = gather_weight.cpu()
    torch.cuda.empty_cache()

     # 我需要改的--------------------------------------------------------------------------------------------------------------

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
    a =  config_dict['a']
    b =  config_dict['b']
    c =  config_dict['c']
    T =  config_dict['T']
    task_bs = 1 # The number of tasks to stack to each other for parallel optimization  代表一次计算一个任务，固定这个数值先不变化

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
        np.random.seed(rng_seed)
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
        logger(f"  --> a: {a}", flush=True)  # 记录abc
        logger(f"  --> b: {b}", flush=True)  # 记录abc
        logger(f"  --> c: {c}", flush=True)  # 记录abc
        logger(f"  --> T: {T}", flush=True)  # 记录abc
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
        ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_tasks, n_samples, -1) # n_tasks代表一共需要进行几组任务。进行维度重排，其参数指定了新的维度顺序。 n_samples=5类*1张* (1张supprot+15张query)= 80
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
                base_means.append(mean)  # 基类里面一类求一个均值，减少计算量
                base_cov.append(cov)
        logger("* Means and Covariance Matrices are calculated", flush=True)

        # --- Calculate feature description matrices of each base class ---# 计算特征描述矩阵
        with torch.no_grad():
            base_means = torch.cat(
                [torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype) for x in base_means])
            base_cov = torch.cat([torch.from_numpy(x).unsqueeze(0).to(device=device_name, dtype=tch_dtype)
                for x in base_cov])
            # base_means_matrix = torch.matmul(base_means.unsqueeze(-1).expand(task_bs * n_ways_list[0] * n_shots_list[0], base_means.shape[0], base_means.shape[1], 1),
            #                              base_means.unsqueeze(-1).expand(task_bs * n_ways_list[0] * n_shots_list[0], base_means.shape[0], base_means.shape[1], 1).permute(0, 1, 3, 2)).to(device=device_name, dtype=tch_dtype)
            
            # task_bs=1下的基类特征描述矩阵，代表一个任务下的base_means_matrix
            base_means_matrix = torch.matmul(base_means.unsqueeze(-1).expand( base_means.shape[0], base_means.shape[1], 1),
                                         base_means.unsqueeze(-1).expand( base_means.shape[0], base_means.shape[1], 1).permute(0, 2, 1)).to(device=device_name, dtype=tch_dtype)
            

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
                mean_tch, cov_tch = Distribution_DDC(support_data, base_means, base_means_matrix, base_cov,
                                                                 alpha=dc_alpha, k=dc_k, gamma=gm, a=a, b=b, c=c, T=T) 
            assert mean_tch.shape == (batch_dim, n_lsamples, n_dim)
            assert cov_tch.shape == (batch_dim, n_lsamples, n_dim, n_dim)

            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch) #表示多元正态分布的类。它用于生成具有指定均值和协方差矩阵的随机样本，常用于概率建模和统计模拟
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

            if (ii + 1) % 10 == 0:
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
    my_parser.add_argument('--configid', default='5ways/miniImagenet_5s5w', type=str)  # 5-way 1-shot
    # my_parser.add_argument('--configid', default='5ways/CUB_1s5w', type=str)
#     my_parser.add_argument('--configid', default='5ways/CIFAR-FS_5s5w', type=str)

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
    # print("PROJPATH", PROJPATH)
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


