# coding=utf-8
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import json
import os
from shutil import copyfile


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def save_info(dict_data):
    path_save=dict_data["saved"]
    with open(os.path.join(path_save,"infor.json"),"w") as file:
        json.dump(dict_data,file)

def save_train_info(dict_data,path_save,config_file="config.py"):
	save_info(dict_data)
	copyfile(config_file,os.path.join(path_save,config_file))