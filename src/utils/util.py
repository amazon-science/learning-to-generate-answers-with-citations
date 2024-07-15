import argparse
import datetime
import logging
import os
import random
import re
import string
import sys
from shutil import copytree, ignore_patterns

import numpy as np
import psutil
import torch
from pytorch_lightning import seed_everything

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ROOT_DIR = os.path.abspath(os.curdir)


def my_collate_fn(batch):
    dict_batch = {}
    dict_batch["input"] = {}
    dict_batch["output"] = {}

    for datapoint in batch:
        for k, v in datapoint["input"].items():
            if k in dict_batch["input"]:
                dict_batch["input"][k].append(v)
                # dict_batch["input"][k].append(v[0])
            else:
                # dict_batch["input"][k] = [v[0]]
                dict_batch["input"][k] = [v]

        for k, v in datapoint["output"].items():
            if k in dict_batch["output"]:
                # dict_batch["output"][k].append(v[0])
                dict_batch["output"][k].append(v)

            else:
                # dict_batch["output"][k] = [v[0]]
                dict_batch["output"][k] = [v]

    for k, list_v in dict_batch["input"].items():
        if isinstance(list_v[0], int):
            dict_batch["input"][k] = torch.tensor(list_v)
    for k, list_v in dict_batch["output"].items():
        if isinstance(list_v[0], int):
            dict_batch["output"][k] = torch.tensor(list_v)

    return dict_batch


def init_logging(logging, config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(config.exp_dir, "output.log")), logging.StreamHandler()],
    )


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)


def make_dir(dir_name):
    """
    Makes a directory if it doesn't exists yet
    Args:
        dir_name: directory name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_exp_dir(base_exp_dir):
    """
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
    Returns:
        exp_dir_name: experiment directory name
    """
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, "src")

    copytree(os.path.join(os.environ["NICL_ROOT"], "src"), src_file, ignore=ignore_patterns("*.pyc", "tmp*"))

    return exp_dir_name


def print_mem_usage(loc):
    """
    Print memory usage in GB
    :return:
    """
    print(
        "%s gpu mem allocated: %.2f GB; reserved: %.2f GB; max: %.2f GB; cpu mem %d"
        % (
            loc,
            float(torch.cuda.memory_allocated() / 1e9),
            float(torch.cuda.memory_reserved() / 1e9),
            float(torch.cuda.max_memory_allocated() / 1e9),
            psutil.virtual_memory().percent,
        )
    )
    sys.stdout.flush()


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def update_dict_val_store(dict_val_store, dict_update_val, grad_accum_factor):
    """
    Update dict_val_store with dict_update_val

    :param dict_val_store:
    :param dict_update_val:
    :return:
    """
    if dict_val_store is None:
        dict_val_store = dict_update_val
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k] / grad_accum_factor

    return dict_val_store


def get_avg_dict_val_store(dict_val_store, num_batches, grad_accumulation_factor):
    """
    Get average dictionary val

    :param dict_val_store:
    :param eval_every:
    :return:
    """
    dict_avg_val = {}

    for k in dict_val_store.keys():
        dict_avg_val[k] = float("%.3f" % (dict_val_store[k] / num_batches / grad_accumulation_factor))

    return dict_avg_val


def cast_to_precision(model, precision):
    if precision == "fp32":
        return model
    elif precision == "fp16":
        model.to(torch.float16)
    elif precision == "bf16":
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported precision {precision}, must be one of fp32, fp16, bf16")
    return model


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def get_unwrapped_model_if_wrapped(model):
    if hasattr(model, "module"):
        return model.module
    return model


def normalize_answer_faithdial(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    re_art = re.compile(r"\b(a|an|the)\b")
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = s.lower()
    s = re_punc.sub(" ", s)
    s = re_art.sub(" ", s)
    s = " ".join(s.split())
    return s


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


# def remove_citations_w_space(sent):
#     return re.sub(r"\[\d+", "", re.sub(r" \[\d+", " ", sent)).replace(" |", "").replace("]", "")


def postprocess_generation(gen, reader_model_origin):
    # TODO: Remove wrong citation formats since this causes errors for focused learning step (e.g [n] [b], etc.)
    gen = (
        gen.replace("\u043d", "")
        .replace("\u043e", "")
        .replace("\u0439", "")
        .replace("\u0438", "")
        .replace("\u00ad", "")
        .strip()
    )
    gen = gen.replace("<|im_end|>", "").strip()
    gen = gen.replace("</s>", "").strip()
    gen = gen.replace("<unk>", "").strip()
    gen = gen.replace("</s>", "").strip()

    # Use non-space after citation-fullstop as heuristics for cutting sequence.
    res = re.search(r"]\.(?!\s|$)", gen)
    if res:
        gen = gen[: res.end()]

    if "orca" in reader_model_origin.lower():
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        if "\n\n" in gen:
            gen = gen.split("\n\n")[0]
    elif "instruct" in reader_model_origin.lower():
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        if "\n\n" in gen:
            gen = gen.split("\n\n")[0]
    elif "llama" in reader_model_origin.lower():
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        if "\n\n" in gen:
            gen = gen.split("\n\n")[0]
    elif "DeciLM" in reader_model_origin.lower():
        gen = gen.replace("<s>", "").strip()
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        gen = gen.split("\n\n\n")[0]  # Sometimes generates initial query again, TODO: Why
        if "\n\n" in gen:
            gen = gen.split("\n\n")[0]
    return gen.strip()


def get_max_memory(rank=-1):
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"  # f'{free_in_GB-2}GB'
    n_gpus = torch.cuda.device_count()
    if rank >= 0:  # Distribute eval model to last GPU
        # max_memory = {n_gpus - (rank+1): max_memory}
        max_memory = {rank: max_memory}
    else:
        max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory
