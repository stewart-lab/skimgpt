import src.skim_and_km_api as skim
import copy
import sys
import os
from src.utils import Config

def main_workflow(combination, output_dir, config):
    try:
        if config.job_type == "skim_with_gpt":
            return skim.skim_with_gpt_workflow(
                term=combination,
                config=config,
                output_directory=output_dir
            )
        else:
            return skim.km_with_gpt_workflow(
                term=combination,
                config=config,
                output_directory=output_dir
            )
    except Exception as e:
        config.logger.error(f"Workflow failed for {combination}: {str(e)}")
        return None
