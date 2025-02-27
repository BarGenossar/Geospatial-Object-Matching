import config
import argparse
from utils import *
from pipelines import PipelineManager
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--evaluation_mode', type=str, default=config.Constants.evaluation_mode)
    parser.add_argument('--run_preparatory_phase', type=bool, default=config.TrainingPhase.run_preparatory_phase)
    parser.add_argument('--blocking_method', type=str, default=config.Blocking.blocking_method)
    parser.add_argument('--seed_num', type=int, default=config.Constants.seeds_num)
    args = parser.parse_args()

    logger = define_logger()
    print_config(logger)
    final_result_dict = {}
    for seed in range(1, args.seed_num+1):
        logger.info(f"Seed: {seed}")
        logger.info(3*'--------------------------')
        pipeline_manager_obj = PipelineManager(seed, logger, args)
        final_result_dict[seed] = pipeline_manager_obj.result_dict
    generate_final_result_csv(final_result_dict, args.evaluation_mode, logger)
    logger.info("Done!")


