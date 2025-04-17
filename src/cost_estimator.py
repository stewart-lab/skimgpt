import logging
from typing import List, Tuple
import pandas as pd
from src.prompt_library import km_with_gpt, skim_with_gpt, skim_with_gpt_ac, km_with_gpt_direct_comp
from src.utils import Config

class CostEstimator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("SKiM-GPT")
        self.model = config.model
        self.post_n = config.post_n
        
        # Set costs based on model
        if self.model == "o1":
            self.input_cost_per_million = 15.00
            self.output_cost_per_million = 60.00
        elif self.model == "o1-mini":
            self.input_cost_per_million = 1.10
            self.output_cost_per_million = 4.40
        elif self.model == "r1":
            self.input_cost_per_million = 0.55
            self.output_cost_per_million = 2.19
        else:  # default to o1-mini pricing
            self.input_cost_per_million = 1.10
            self.output_cost_per_million = 4.40
    
    def _calculate_cost(self, tokens: int, is_input: bool = True) -> float:
        """Calculate cost based on number of tokens."""
        cost_per_million = self.input_cost_per_million if is_input else self.output_cost_per_million
        return (tokens / 1_000_000) * cost_per_million
    
    def _get_output_tokens(self, intervals: List[Tuple[int, int, int]], job_type: str) -> int:
        """Get estimated output tokens based on POST_N value."""
        for start, end, tokens in intervals:
            if start <= self.post_n <= end:
                estimated_tokens = tokens
                estimated_cost = self._calculate_cost(estimated_tokens, is_input=False)
                self.logger.info(f"=== {job_type} Output Token Estimation ===")
                self.logger.info(f"Estimated output tokens: {estimated_tokens:,}")
                self.logger.info(f"Estimated output cost (${self.output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
                return estimated_tokens
        
        # Default to highest value if POST_N is out of range
        default_tokens = intervals[-1][2]
        estimated_cost = self._calculate_cost(default_tokens, is_input=False)
        self.logger.info(f"=== {job_type} Output Token Estimation ===")
        self.logger.info(f"Estimated output tokens: {default_tokens:,}")
        self.logger.info(f"Estimated output cost (${self.output_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        return default_tokens

class KMCostEstimator(CostEstimator):
    KM_INTERVALS = [
        (0, 25, 4400), (25, 50, 4600), (50, 75, 4800),
        (75, 100, 5000), (100, 125, 5200), (125, 150, 5400),
        (150, 175, 5600), (175, 200, 5800)
    ]
    
    DEEPSEEK_KM_INTERVALS = [
        (0, 25, 780), (25, 50, 850), (50, 75, 920),
        (75, 100, 990), (100, 125, 1060), (125, 150, 1130),
        (150, 175, 1200), (175, 200, 1270)
    ]
    
    def estimate_input_costs(self, combined_df: pd.DataFrame) -> int:
        """Calculate input token costs for KM jobs."""
        total_tokens = 0
        
        self.logger.info("=== KM Input Cost Estimation ===")
        self.logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            abstract_tokens = min(row['ab_count'], self.post_n) * 300
            
            # Get b_terms - either from array for direct_comp or single term for regular km
            if self.config.job_type == "km_with_gpt_direct_comp":
                # Split b_term string into array and take first two terms
                hypothesis_template = self.config.km_direct_comp_hypothesis.format(
                    a_term=row['a_term'],
                    b_term1=row['b_term'][0], 
                    b_term2=row['b_term'][1]  
                )
                # Use the correct prompt function for direct comparison
                prompt_text = km_with_gpt_direct_comp(row['b_term'][0], row['b_term'][1], row['a_term'], hypothesis_template, "")
            else:
                hypothesis_template = self.config.km_hypothesis.format(
                    a_term=row['a_term'], 
                    b_term=row['b_term']
                )
                prompt_text = km_with_gpt(row['b_term'], row['a_term'], hypothesis_template, "")
            
            prompt_tokens = int(len(prompt_text.replace("{consolidated_abstracts}", "").split()) * 4/3)
            
            row_total_tokens = abstract_tokens + prompt_tokens
            total_tokens += row_total_tokens
            
            if self.config.job_type == "km_with_gpt_direct_comp":
                self.logger.info(f"Row {idx + 1} ({row['a_term']}-{row['b_term'][0]}/{row['b_term'][1]}): Total input tokens: {row_total_tokens:,}")
            else:
                self.logger.info(f"Row {idx + 1} ({row['a_term']}-{row['b_term']}): Total input tokens: {row_total_tokens:,}")
        
        estimated_cost = self._calculate_cost(total_tokens)
        self.logger.info("=== Overall Summary ===")
        self.logger.info(f"Total estimated input tokens: {total_tokens:,}")
        self.logger.info(f"Estimated input cost (${self.input_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        
        return total_tokens
    
    def get_output_tokens(self) -> int:
        """Get estimated output tokens for KM jobs."""
        # Use appropriate intervals based on model
        if self.model == "r1":
            return self._get_output_tokens(self.DEEPSEEK_KM_INTERVALS, "KM")
        else:
            return self._get_output_tokens(self.KM_INTERVALS, "KM")

class SkimCostEstimator(CostEstimator):
    SKIM_INTERVALS = [
        (0, 25, 8500), (25, 50, 9800), (50, 75, 11000),
        (75, 100, 12300), (100, 125, 13600), (125, 150, 14800),
        (150, 175, 16000), (175, 200, 17400)
    ]
    
    DEEPSEEK_SKIM_INTERVALS = [
        (0, 25, 2690), (25, 50, 5710), (50, 75, 8730),
        (75, 100, 11750), (100, 125, 14770), (125, 150, 17790),
        (150, 175, 20810), (175, 200, 23830)
    ]
    
    def estimate_input_costs(self, combined_df: pd.DataFrame) -> int:
        """Calculate input token costs for SKIM jobs."""
        total_tokens = 0
        
        self.logger.info("=== SKIM Input Cost Estimation ===")
        self.logger.info("Row-by-row breakdown:")
        
        for idx, row in combined_df.iterrows():
            ab_tokens = min(row['ab_count'], self.post_n) * 300
            bc_tokens = min(row['bc_count'], self.post_n) * 300
            
            hypothesis_template = self.config.skim_hypotheses["ABC"].format(
                a_term=row['a_term'],
                b_term=row['b_term'],
                c_term=row['c_term']
            )
            
            base_prompt_text = skim_with_gpt(row['b_term'], row['a_term'], hypothesis_template, "", row['c_term'])
            base_prompt_tokens = int(len(base_prompt_text.replace("{consolidated_abstracts}", "").split()) * 4/3)
            
            row_total_tokens = ab_tokens + bc_tokens + base_prompt_tokens
            
            if 'ac_count' in row and row['ac_count'] > 0:
                ac_tokens = min(row['ac_count'], self.post_n) * 300
                ac_prompt_text = skim_with_gpt_ac(row['a_term'], hypothesis_template, "", row['c_term'])
                ac_prompt_tokens = int(len(ac_prompt_text.replace("{consolidated_abstracts}", "").split()) * 4/3)
                row_total_tokens += ac_tokens + ac_prompt_tokens
            
            total_tokens += row_total_tokens
            self.logger.info(f"Row {idx + 1} ({row['a_term']}-{row['b_term']}-{row['c_term']}): Total input tokens: {row_total_tokens:,}")
        
        estimated_cost = self._calculate_cost(total_tokens)
        self.logger.info("=== Overall Summary ===")
        self.logger.info(f"Total estimated input tokens: {total_tokens:,}")
        self.logger.info(f"Estimated input cost (${self.input_cost_per_million:.2f}/million tokens): ${estimated_cost:.2f}")
        
        return total_tokens
    
    def get_output_tokens(self) -> int:
        """Get estimated output tokens for SKIM jobs."""
        # Use appropriate intervals based on model
        if self.model == "r1":
            return self._get_output_tokens(self.DEEPSEEK_SKIM_INTERVALS, "SKIM")
        else:
            return self._get_output_tokens(self.SKIM_INTERVALS, "SKIM")

def calculate_total_cost_and_prompt(config: Config, input_tokens: int) -> bool:
    """Calculate total cost and prompt user for confirmation."""
    estimator = KMCostEstimator(config) if config.job_type in ["km_with_gpt", "km_with_gpt_direct_comp"] else SkimCostEstimator(config)
    
    if config.job_type not in ["km_with_gpt", "skim_with_gpt", "km_with_gpt_direct_comp"]:
        estimator.logger.info(f"Cost calculation not available for {config.job_type} jobs")
        return True
    
    estimated_output_tokens = estimator.get_output_tokens()
    estimated_input_cost = estimator._calculate_cost(input_tokens)
    estimated_output_cost = estimator._calculate_cost(estimated_output_tokens, is_input=False)
    
    # Calculate cost per iteration
    cost_per_iteration = estimated_input_cost + estimated_output_cost
    
    # Determine number of iterations
    num_iterations = 1
    if config.iterations:
        if isinstance(config.iterations, int) and config.iterations > 0:
            num_iterations = config.iterations
        elif isinstance(config.iterations, bool) and config.iterations:
            # If iterations is set to True but no number specified
            estimator.logger.warning("iterations is set to True but no number specified, assuming 1 iteration for cost")
    
    # Calculate total cost across all iterations
    total_cost = cost_per_iteration * num_iterations
    
    print("\n" + "="*50)
    print(f"COST ESTIMATION FOR {config.job_type.upper()} WITH {config.model.upper()}")
    print("="*50)
    print(f"POST_N value: {config.post_n}")
    print(f"Estimated input tokens:  {input_tokens:,}")
    print(f"Estimated input cost:    ${estimated_input_cost:.2f}")
    print(f"Estimated output tokens: {estimated_output_tokens:,}")
    print(f"Estimated output cost:   ${estimated_output_cost:.2f}")
    print(f"Cost per iteration:      ${cost_per_iteration:.2f}")
    
    if num_iterations > 1:
        print(f"Number of iterations:    {num_iterations}")
        print("-"*50)
        print(f"TOTAL ESTIMATED COST:    ${total_cost:.2f} ({num_iterations} iterations)")
    else:
        print("-"*50)
        print(f"TOTAL ESTIMATED COST:    ${total_cost:.2f}")
    
    print("="*50)
    
    while True:
        response = input("\nDo you want to continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            estimator.logger.info("User chose to abort the job due to cost concerns")
            return False
        else:
            print("Please enter 'y' or 'n'")