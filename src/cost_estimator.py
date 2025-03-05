import os
import logging
from src.prompt_library import km_with_gpt, skim_with_gpt, skim_with_gpt_ac
from openai import OpenAI
import pandas as pd

# O-1
# POST_N = 5, 50
# SKIM (5 rows) AND KM

# o1-mini
# POST_N = 5, 50 ,100
# SKIM  AND KM

def estimate_input_costs_km(config, combined_df, output_directory):
    logger = logging.getLogger("SKiM-GPT")
    
    try:
        total_abstract_tokens = 0
        post_n = config.post_n
        cost_details = []
        
        logger.info("=== KM Input Cost Estimation ===")
        logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            ab_count = row['ab_count']
            abstracts_to_process = min(ab_count, post_n)
            abstract_tokens = abstracts_to_process * 300  # 300 tokens per abstract
            
            # Get terms and format hypothesis
            a_term = row['a_term']
            b_term = row['b_term']
            hypothesis_template = config.km_hypothesis.format(a_term=a_term, b_term=b_term)
            
            # Calculate prompt tokens
            prompt_text = km_with_gpt(b_term, a_term, hypothesis_template, "")
            prompt_text = prompt_text.replace("{consolidated_abstracts}", "")
            prompt_tokens = int(len(prompt_text.split()) * 4/3)
            
            # Calculate total tokens
            total_tokens_row = abstract_tokens + prompt_tokens
            
            # Log only token counts, no cost information
            logger.info(f"Row {idx + 1} ({a_term}-{b_term}): Abstract tokens: {abstract_tokens:,}, Prompt tokens: {prompt_tokens:,}, Total tokens: {total_tokens_row:,}")
            
            cost_details.append({
                'row': idx + 1,
                'abstract_tokens': abstract_tokens,
                'prompt_tokens': prompt_tokens,
                'total_tokens': total_tokens_row
            })
            
            total_abstract_tokens += abstract_tokens

        # Calculate totals
        total_prompt_tokens = sum(detail['prompt_tokens'] for detail in cost_details)
        total_tokens = total_abstract_tokens + total_prompt_tokens
        
        # Log summary without cost information
        logger.info("=== Overall Summary ===")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Abstract tokens: {total_abstract_tokens:,}")
        logger.info(f"Prompt tokens: {total_prompt_tokens:,}")
        
    except Exception as e:
        logger.error(f"Error calculating token estimation: {str(e)}", exc_info=True)
        raise

def estimate_input_costs_skim(config, combined_df, output_directory):
    logger = logging.getLogger("SKiM-GPT")
    
    try:
        total_abstract_tokens = 0
        post_n = config.post_n
        cost_details = []
        
        logger.info("=== SKIM Input Cost Estimation ===")
        logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            # Calculate abstract tokens for ab and bc
            ab_count = row['ab_count']
            bc_count = row['bc_count']
            ab_abstracts_to_process = min(ab_count, post_n)
            bc_abstracts_to_process = min(bc_count, post_n)
            ab_tokens = ab_abstracts_to_process * 300
            bc_tokens = bc_abstracts_to_process * 300
            
            # Get terms and format hypothesis
            a_term = row['a_term']
            b_term = row['b_term']
            c_term = row['c_term']
            hypothesis_template = config.skim_hypotheses["ABC"].format(a_term=a_term, b_term=b_term, c_term=c_term)
            
            # Calculate base prompt tokens (skim_with_gpt)
            base_prompt_text = skim_with_gpt(b_term, a_term, hypothesis_template, "", c_term)
            base_prompt_text = base_prompt_text.replace("{consolidated_abstracts}", "")
            base_prompt_tokens = int(len(base_prompt_text.split()) * 4/3)
            
            # Initialize total tokens for this row
            abstract_tokens = ab_tokens + bc_tokens
            prompt_tokens = base_prompt_tokens
            
            # Add AC tokens if present
            if 'ac_count' in row and row['ac_count'] > 0:
                ac_count = row['ac_count']
                ac_abstracts_to_process = min(ac_count, post_n)
                ac_tokens = ac_abstracts_to_process * 300
                abstract_tokens += ac_tokens
                
                # Add AC prompt tokens
                ac_prompt_text = skim_with_gpt_ac(a_term, hypothesis_template, "", c_term)
                ac_prompt_text = ac_prompt_text.replace("{consolidated_abstracts}", "")
                ac_prompt_tokens = int(len(ac_prompt_text.split()) * 4/3)
                prompt_tokens += ac_prompt_tokens
            
            # Calculate total tokens
            total_tokens_row = abstract_tokens + prompt_tokens
            
            # Log only token counts, no cost information
            logger.info(f"Row {idx + 1} ({a_term}-{b_term}-{c_term}): Abstract tokens: {abstract_tokens:,}, Prompt tokens: {prompt_tokens:,}, Total tokens: {total_tokens_row:,}")
            
            cost_details.append({
                'row': idx + 1,
                'abstract_tokens': abstract_tokens,
                'prompt_tokens': prompt_tokens,
                'total_tokens': total_tokens_row
            })
            
            total_abstract_tokens += abstract_tokens

        # Calculate totals
        total_prompt_tokens = sum(detail['prompt_tokens'] for detail in cost_details)
        total_tokens = total_abstract_tokens + total_prompt_tokens
        
        # Log summary without cost information
        logger.info("=== Overall Summary ===")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Abstract tokens: {total_abstract_tokens:,}")
        logger.info(f"Prompt tokens: {total_prompt_tokens:,}")
        
    except Exception as e:
        logger.error(f"Error calculating token estimation: {str(e)}", exc_info=True)
        raise