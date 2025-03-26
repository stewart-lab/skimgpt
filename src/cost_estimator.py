import logging
from src.prompt_library import km_with_gpt, skim_with_gpt, skim_with_gpt_ac

def estimate_input_costs_km(config, combined_df, output_directory):
    logger = logging.getLogger("SKiM-GPT")
    
    try:
        total_tokens = 0
        post_n = config.post_n
        
        # Set cost per million tokens based on model
        model = config.model
        if model == "o1":
            input_cost_per_million = 15.00
        else:  # o1-mini or default
            input_cost_per_million = 1.10
        
        logger.info("=== KM Input Cost Estimation ===")
        logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            ab_count = row['ab_count']
            abstracts_to_process = min(ab_count, post_n)
            abstract_tokens = abstracts_to_process * 300
            
            a_term = row['a_term']
            b_term = row['b_term']
            hypothesis_template = config.km_hypothesis.format(a_term=a_term, b_term=b_term)
            
            prompt_text = km_with_gpt(b_term, a_term, hypothesis_template, "")
            prompt_text = prompt_text.replace("{consolidated_abstracts}", "")
            prompt_tokens = int(len(prompt_text.split()) * 4/3)
            
            row_total_tokens = abstract_tokens + prompt_tokens
            total_tokens += row_total_tokens
            
            # Log only total tokens for this row
            logger.info(f"Row {idx + 1} ({a_term}-{b_term}): Total input tokens: {row_total_tokens:,}")
        
        # Calculate and log total cost
        estimated_cost = (total_tokens / 1_000_000) * input_cost_per_million
        logger.info("=== Overall Summary ===")
        logger.info(f"Total estimated input tokens: {total_tokens:,}")
        logger.info(f"Estimated input cost (${input_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"Error calculating token estimation: {str(e)}", exc_info=True)
        raise

def estimate_input_costs_skim(config, combined_df, output_directory):
    logger = logging.getLogger("SKiM-GPT")
    
    try:
        total_tokens = 0
        post_n = config.post_n
        
        # Set cost per million tokens based on model
        model = config.model
        if model == "o1":
            input_cost_per_million = 15.00
        else:  # o1-mini or default
            input_cost_per_million = 1.10
        
        logger.info("=== SKIM Input Cost Estimation ===")
        logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            ab_count = row['ab_count']
            bc_count = row['bc_count']
            ab_abstracts_to_process = min(ab_count, post_n)
            bc_abstracts_to_process = min(bc_count, post_n)
            ab_tokens = ab_abstracts_to_process * 300
            bc_tokens = bc_abstracts_to_process * 300
            
            a_term = row['a_term']
            b_term = row['b_term']
            c_term = row['c_term']
            hypothesis_template = config.skim_hypotheses["ABC"].format(a_term=a_term, b_term=b_term, c_term=c_term)
            
            base_prompt_text = skim_with_gpt(b_term, a_term, hypothesis_template, "", c_term)
            base_prompt_text = base_prompt_text.replace("{consolidated_abstracts}", "")
            base_prompt_tokens = int(len(base_prompt_text.split()) * 4/3)
            
            row_total_tokens = ab_tokens + bc_tokens + base_prompt_tokens
            
            if 'ac_count' in row and row['ac_count'] > 0:
                ac_count = row['ac_count']
                ac_abstracts_to_process = min(ac_count, post_n)
                ac_tokens = ac_abstracts_to_process * 300
                
                ac_prompt_text = skim_with_gpt_ac(a_term, hypothesis_template, "", c_term)
                ac_prompt_text = ac_prompt_text.replace("{consolidated_abstracts}", "")
                ac_prompt_tokens = int(len(ac_prompt_text.split()) * 4/3)
                
                row_total_tokens += ac_tokens + ac_prompt_tokens
            
            total_tokens += row_total_tokens
            
            # Log only total tokens for this row
            logger.info(f"Row {idx + 1} ({a_term}-{b_term}-{c_term}): Total input tokens: {row_total_tokens:,}")
        
        # Calculate and log total cost
        estimated_cost = (total_tokens / 1_000_000) * input_cost_per_million
        logger.info("=== Overall Summary ===")
        logger.info(f"Total estimated input tokens: {total_tokens:,}")
        logger.info(f"Estimated input cost (${input_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"Error calculating token estimation: {str(e)}", exc_info=True)
        raise

def get_estimated_output_tokens_skim(post_n, config):
    logger = logging.getLogger("SKiM-GPT")
    
    # Set cost per million tokens based on model
    model = config.model
    if model == "o1":
        output_cost_per_million = 60.00
    else:  # o1-mini or default
        output_cost_per_million = 4.40
    
    # Define intervals based on the provided data
    intervals = [
        (0, 25, 8500),
        (25, 50, 9800),
        (50, 75, 11000),
        (75, 100, 12300),
        (100, 125, 13600),
        (125, 150, 14800),
        (150, 175, 16000),
        (175, 200, 17400)
    ]
    
    # Find the appropriate interval
    for start, end, tokens in intervals:
        if start <= post_n <= end:
            estimated_tokens = tokens
            estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
            logger.info(f"=== SKIM Output Token Estimation ===")
            logger.info(f"Estimated output tokens: {estimated_tokens:,}")
            logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
            return estimated_tokens
    
    # If POST_N is greater than the highest interval, use the highest value
    if post_n > 200:
        estimated_tokens = 17400
        estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
        logger.info(f"=== SKIM Output Token Estimation ===")
        logger.info(f"Estimated output tokens: {estimated_tokens:,}")
        logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        return estimated_tokens
    
    # Default fallback
    estimated_tokens = 8500
    estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
    logger.info(f"=== SKIM Output Token Estimation ===")
    logger.info(f"Estimated output tokens: {estimated_tokens:,}")
    logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
    return estimated_tokens

def get_estimated_output_tokens_km(post_n, config):
    logger = logging.getLogger("SKiM-GPT")
    
    # Set cost per million tokens based on model
    model = config.model
    if model == "o1":
        output_cost_per_million = 60.00
    else:  # o1-mini or default
        output_cost_per_million = 4.40
    
    # Define intervals based on the provided data
    intervals = [
        (0, 25, 4400),
        (25, 50, 4600),
        (50, 75, 4800),
        (75, 100, 5000),
        (100, 125, 5200),
        (125, 150, 5400),
        (150, 175, 5600),
        (175, 200, 5800)
    ]
    
    # Find the appropriate interval
    for start, end, tokens in intervals:
        if start <= post_n <= end:
            estimated_tokens = tokens
            estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
            logger.info(f"=== KM Output Token Estimation ===")
            logger.info(f"Estimated output tokens: {estimated_tokens:,}")
            logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
            return estimated_tokens
    
    # If POST_N is greater than the highest interval, use the highest value
    if post_n > 200:
        estimated_tokens = 5800
        estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
        logger.info(f"=== KM Output Token Estimation ===")
        logger.info(f"Estimated output tokens: {estimated_tokens:,}")
        logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        return estimated_tokens
    
    # Default fallback
    estimated_tokens = 4400
    estimated_cost = (estimated_tokens / 1_000_000) * output_cost_per_million
    logger.info(f"=== KM Output Token Estimation ===")
    logger.info(f"Estimated output tokens: {estimated_tokens:,}")
    logger.info(f"Estimated output cost (${output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
    return estimated_tokens

def calculate_total_cost_and_prompt(config, input_tokens, output_directory):
    logger = logging.getLogger("SKiM-GPT")
    
    # Get estimated output tokens based on POST_N and job type
    post_n = config.post_n
    if config.job_type == "skim_with_gpt":
        estimated_output_tokens = get_estimated_output_tokens_skim(post_n, config)
    elif config.job_type == "km_with_gpt":
        estimated_output_tokens = get_estimated_output_tokens_km(post_n, config)
    else:
        logger.info(f"Cost calculation not available for {config.job_type} jobs")
        return True
    
    # Calculate costs based on model
    model = config.model
    if model == "o1":
        input_cost_per_million = 15.00
        output_cost_per_million = 60.00
    elif model == "o1-mini":
        input_cost_per_million = 1.10
        output_cost_per_million = 4.40
    else:
        logger.warning(f"Unknown model: {model}, using o1-mini pricing")
        input_cost_per_million = 1.10
        output_cost_per_million = 4.40
    
    # Calculate costs
    estimated_input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    estimated_output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = estimated_input_cost + estimated_output_cost
    
    # Display cost information
    print("\n" + "="*50)
    print(f"COST ESTIMATION FOR {config.job_type.upper()} WITH {model.upper()}")
    print("="*50)
    print(f"POST_N value: {post_n}")
    print(f"Estimated input tokens:  {input_tokens:,}")
    print(f"Estimated input cost:    ${estimated_input_cost:.2f}")
    print(f"Estimated output tokens: {estimated_output_tokens:,}")
    print(f"Estimated output cost:   ${estimated_output_cost:.2f}")
    print("-"*50)
    print(f"TOTAL ESTIMATED COST:    ${total_cost:.2f}")
    print("="*50)
    
    # Prompt for confirmation
    while True:
        response = input("\nDo you want to continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            logger.info("User chose to abort the job due to cost concerns")
            return False
        else:
            print("Please enter 'y' or 'n'")