import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch import nn

from src.data.data_module import FinetuneDataModule
from src.lit_module import LitModule
from src.utils.Config import Config
from src.utils.util import ParseKwargs, init_logging, set_seeds

logger = logging.getLogger(__name__)


def main(config):

    logger.info("Start Training...")

    datamodule = FinetuneDataModule(config)
    litmodule = LitModule(config, datamodule, split="train")

    val_check_interval = int((config.num_steps) * config.grad_accum_factor) + 10000000

    # Add one step per additional training sample
    config.num_steps = (
        config.num_steps if not config.dynamic_steps else config.num_steps + datamodule.get_num_training_samples()
    )

    trainer = Trainer(
        enable_checkpointing=False,
        logger=False,
        accelerator="gpu",
        devices=1,  # torch.cuda.device_count() - config.evaluation_gpus,
        # precision=16,
        # precision=config.compute_precision,
        strategy="auto",
        max_steps=config.num_steps,
        num_sanity_val_steps=0,  # Important to have this flag to merge predictions over multiple batches
        check_val_every_n_epoch=None,  # config.check_val_every_n_epoch,
        val_check_interval=val_check_interval,  # config.val_check_interval, #int((config.num_steps) * config.grad_accum_factor), # Since no validation, evaluate at last step
        accumulate_grad_batches=config.grad_accum_factor,
        gradient_clip_val=config.grad_clip_norm,
    )

    trainer.fit(litmodule, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)

    init_logging(logging, config)

    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in [
        "auto",
        "fsdp",
        "ddp",
        "ddp_spawn",
        "deepspeed_stage_3_offload",
        "deepspeed_stage_3",
    ]

    print(config.to_json())

    set_seeds(config.seed)
    main(config)
