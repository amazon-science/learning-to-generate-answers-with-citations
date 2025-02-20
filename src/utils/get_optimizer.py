"""
Based on https://github.com/r-three/t-few.
"""
import re
from collections import defaultdict

import torch
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import Adafactor


def get_optimizer(model, config):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config.optimizer

    def param_name_to_group_name(param_name):
        if False:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()
    for param_name, param in model.named_parameters():
        # if param.ndim == 1:
        #     # cast the small parameters (e.g. layernorm) to fp32 for stability
        #     param.data = param.data.to(torch.float32) # BF16 float
        # param.data = param.data.to(torch.bfloat16) # BF16 float

        if re.fullmatch(config.trainable_param_names, param_name):
            param_groups[param_name_to_group_name(param_name)]["params"].append(param)
            trainable_param_names.add(param_name)
        else:
            param.requires_grad = False

    param_groups = param_groups.values()
    if optim_name.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=config.lr)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scale_parameter=config.scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
    elif optim_name.lower() == "deepspeed_adam":
        optimizer = DeepSpeedCPUAdam(param_groups, lr=config.lr)
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return optimizer, trainable_param_names
