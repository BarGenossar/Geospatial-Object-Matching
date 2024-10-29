import json
import os
import config
from utils import *
from blocking import Blocker
from collections import defaultdict
from process_pairs import PairProcessor
from object_properties import ObjectPropertiesProcessor
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import FlexibleClassifier
from pipelines import PipelineManagerClassicModels, PipelineManagerGNN
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    logger = define_logger()
    print_config(logger)
    model = config.Constants.model
    for seed in range(1, config.Constants.seeds_num + 1):
        logger.info(f"Seed: {seed}")
        logger.info(3*'--------------------------')
        if model == 'GNN':
            PipelineManagerGNN(seed, logger)
        else:
            PipelineManagerClassicModels(seed, logger)
    logger.info("Done!")


