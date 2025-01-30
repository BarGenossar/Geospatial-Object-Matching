import config
from utils import *
from pipelines import PipelineManager
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    logger = define_logger()
    print_config(logger)
    final_result_dict = {}
    run_prepahase = config.PreparatoryPhase.run_preparatory_phase
    for seed in range(1, config.Constants.seeds_num + 1):
        logger.info(f"Seed: {seed}")
        logger.info(3*'--------------------------')
        pipeline_manager_obj = PipelineManager(seed, logger, run_prepahase)
        if config.Constants.evaluation_mode == 'matching':
            final_result_dict[seed] = pipeline_manager_obj.flexible_classifier_obj.result_dict
    generate_final_result_csv(final_result_dict, logger)
    logger.info("Done!")


