from __future__ import print_function

import collections
import os
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import configs
import wrn_model
from data.datamgr import SimpleDataManager
from io_utils import parse_args, get_resume_file

use_gpu = torch.cuda.is_available()

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, tag='last',set='base'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)          # 构建保存目录路径
    if os.path.isfile(save_dir + '/%s_features.plk'%set):   # 如果已经存在保存文件，则直接加载并返回特征数据
        data = load_pickle(save_dir + '/%s_features.plk'%set)
        return data
    else:       # 如果保存目录不存在，则创建目录
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        # 初始化一个字典，用于存储特征信息
        output_dict = collections.defaultdict(list)

        # 遍历数据加载器中的每个批次
        for i, (inputs, labels) in enumerate(val_loader):  
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs, _ = model(inputs)      # 计算模型的输出
            outputs = outputs.cpu().data.numpy()

            # 将每个样本的特征与其对应的标签进行映射存储
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
        
        # 将所有样本的特征信息存储到文件中
        all_info = output_dict 
        save_pickle(save_dir + '/%s_features.plk' % set, all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('train')
    params.model = 'WideResNet28_10'
    params.method = 'rotation'

    # 构建文件路径
    loadfile_base = configs.data_dir[params.dataset] + '/base.json'  
    # print("loadfile_base", loadfile_base)
    loadfile_novel = configs.data_dir[params.dataset] + '/novel.json'
    loadfile_val = configs.data_dir[params.dataset] + '/val.json'
    if params.dataset == 'CIFAR-FS':
        datamgr = SimpleDataManager(32, batch_size=128)
    else:
        datamgr = SimpleDataManager(80, batch_size=32)

    base_loader = datamgr.get_data_loader(loadfile_base, aug=False)
    novel_loader = datamgr.get_data_loader(loadfile_novel, aug=False)
    val_loader = datamgr.get_data_loader(loadfile_val, aug=False)
    # checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    checkpoint_dir = 'checkpoints/%s' % params.dataset
    modelfile = get_resume_file(checkpoint_dir)

    if params.model == 'WideResNet28_10':
        model = wrn_model.wrn28_10(num_classes=params.num_classes)


    model = model.cuda()
    cudnn.benchmark = True  # 启用 cuDNN 的 benchmark 模式以提高性能

    checkpoint = torch.load(modelfile) #  从文件中加载模型的检查点
    state = checkpoint['state']
    state_keys = list(state.keys())
    # print(state_keys)
    callwrap = False
    if 'module' in state_keys[0]: # 检查状态字典的第一个键是否包含字符串 'module'，如果是，则说明模型已经被包装
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict() # 获取包装后模型的状态字典。
    state.popitem()
    state.popitem()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()
    print(checkpoint_dir)
    output_dict_base = extract_feature(base_loader, model, checkpoint_dir, tag='feature', set='base')
    print("base set features saved!")
    output_dict_novel = extract_feature(novel_loader, model, checkpoint_dir, tag='feature', set='novel')
    print("novel features saved!")
    output_dict_val = extract_feature(val_loader, model, checkpoint_dir, tag='feature', set='val')
    print("val set features saved!")

