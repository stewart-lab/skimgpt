import src.skim_and_km_api as fastkm

def main_workflow(combination, output_dir, config):
    try:
        if config.job_type == "skim_with_gpt":
            return fastkm.skim_with_gpt_workflow(
                term=combination,
                config=config,
                output_directory=output_dir
            )
        elif config.job_type == "km_with_gpt_direct_comp":
            return fastkm.km_with_gpt_direct_comp_workflow(
                term=combination,
                config=config,
                output_directory=output_dir
            )
        else:
            return fastkm.km_with_gpt_workflow(
                term=combination,
                config=config,
                output_directory=output_dir
            )
    except Exception as e:
        config.logger.error(f"Workflow failed for {combination}: {str(e)}")
        return None
