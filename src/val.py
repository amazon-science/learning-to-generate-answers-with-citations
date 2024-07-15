import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import timedelta

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from src.data.data_module import FinetuneDataModule
from src.lit_module import LitModule
from src.utils.Config import Config
from src.utils.util import ParseKwargs, init_logging, set_seeds

logger = logging.getLogger(__name__)


@torch.no_grad()
def main(config):

    logger.info("Start Evaluation...")

    if torch.cuda.device_count() > 1:
        # config.compute_strategy = "ddp"
        config.compute_strategy = "ddp"
        compute_strategy = DDPStrategy(timeout=timedelta(hours=6))
    else:
        config.compute_strategy = "auto"
        compute_strategy = "auto"

    logger.info("Current strategy: {}".format(config.compute_strategy))

    datamodule = FinetuneDataModule(config)
    # datamodule.setup(stage=None) # Init all dataloaders

    litmodule = LitModule(config, datamodule, split="dev")

    trainer = Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        # precision=config.compute_precision,
        strategy=compute_strategy if compute_strategy != "none" else None,
    )

    trainer.validate(litmodule, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    init_logging(logging, config)

    logger.info(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["auto", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    set_seeds(config.seed)
    main(config)
