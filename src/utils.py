from __future__ import annotations

import numpy as np
import pandas as pd
import json
import os
import logging  
import re
import openai

# Compiled PMID pattern for reuse across modules
PMID_PATTERN = re.compile(r"PMID:\s*(\d+)")


def extract_pmid(text: str) -> str:
    """Extract the first PMID from text.
    
    Args:
        text: String containing PMID reference
        
    Returns:
        First PMID number as string, or empty string if not found
    """
    match = PMID_PATTERN.search(text)
    return match.group(1) if match else ""


def extract_pmids(text: str) -> list:
    """Extract all PMIDs from text.
    
    Args:
        text: String containing PMID references
        
    Returns:
        List of PMID numbers as strings
    """
    return PMID_PATTERN.findall(text)


class RaggedTensor:
    def __init__(self, data, break_point=[]):
        self.data = data
        self.break_point = break_point
        self.index = 0
        self.getShape()

    def getShape(self) -> None:
        if self.is2D():
            self.shape = [len(i) for i in self.data]
        else:
            self.shape = len(self.data)

    def is2D(self) -> bool:
        if not (len(self.data) == 0):
            return isinstance(self.data[0], list)
        else:
            return False

    # Duplicates each element in data according to the shape_list
    def expand(self, shape_list: list) -> None:
        assert (
            not self.is2D()
        ), "Data must be 1D before calling expand. Call flatten first?"
        assert self.shape == len(
            shape_list
        ), "The length of shape list must equal the length of data"

        expanded = []
        for idx, inp in enumerate(self.data):
            expanded.extend([inp] * shape_list[idx])

        return RaggedTensor(expanded)

    def flatten(self) -> RaggedTensor:
        if self.is2D():
            output = []
            for lst in self.data:
                output.extend(lst)
            return RaggedTensor(output)
        else:
            return self

    # Inverts the expand method
    def compress(self, shape_list: list):
        assert self.shape == sum(shape_list)
        self.data = list(set(self.data))
        self.getShape()

    # Splits the data depending on the index
    def split(self):
        if len(self.break_point) == 0:
            print("Warning: No breakpoint was specified.")
            return self, RaggedTensor([])
        past_break_point = 0
        output = []
        for break_point in self.break_point:
            output.append(RaggedTensor(self.data[past_break_point:break_point]))
            past_break_point = break_point
        output.append(RaggedTensor(self.data[past_break_point:]))
        return output

    # Reshapes the data depending on the input
    def reshape(self, shape: list) -> list:
        assert not self.is2D(), "Reshape only works with 1D tensors."
        assert self.shape == sum(
            shape
        ), "The shape of the tensor should be equal to the sum of the wanted shape."
        output = []
        running_length = 0
        for length in shape:
            output.append(self.data[running_length : running_length + length])
            running_length += length

        self.data = output
        self.getShape()

    # Applies a mask to the tensor
    def applyFilter(self, mask: RaggedTensor) -> None:
        assert (
            self.shape == mask.shape
        ), "Filtering only works when the shapes are the same"
        if self.is2D():
            for i in range(len(self.data)):
                boolean_mask = np.array(mask[i]) == 1
                self.data[i] = list(np.array(self.data[i])[boolean_mask])
        else:
            boolean_mask = np.array(mask) == 1
            self.data = list(np.array(self.data)[boolean_mask])

    # Applies a function to the tensor
    def map(self, func: callable, *args) -> RaggedTensor:
        assert not self.is2D(), "Map only works with 1D tensors"
        return RaggedTensor([func(i, *args) for i in self.data], self.break_point)

    # Simply concatenates two ragged tensors and appends to the break_point list
    def __add__(self, other: RaggedTensor) -> RaggedTensor:
        assert not self.is2D(), "Adding only works with flattened tensors"
        break_point = self.shape
        return RaggedTensor(self.data + other.data, self.break_point + [break_point])

    def __str__(self):
        return str(self.data)

    def __iter__(self):
        return self.flatten().data.__iter__()

    def __getitem__(self, index: int) -> any:
        return self.data[index]


def strip_pipe(term: str) -> str:
    """Collapse pipe-separated synonyms to the first option with robust handling.
    
    This function takes pipe-separated alternatives and returns only the first option,
    while preserving the overall structure and context of the text.
    
    Examples:
        "cancer|tumor" -> "cancer"
        "lung cancer|lung tumor|pulmonary cancer" -> "lung cancer"
        "HPV|human papillomavirus infection" -> "HPV"
        "chronic|persistent inflammation" -> "chronic"
        "diabetes|diabetes mellitus|diabetic" -> "diabetes"
        None -> ""
        "" -> ""
        "  " -> ""
        "term1||term2" -> "term1"
        "|term1|term2" -> "term1"
    
    Args:
        term: String that may contain pipe-separated alternatives, or None
        
    Returns:
        String with only the first option from each pipe-separated group,
        with normalized whitespace. Returns empty string for None or empty inputs.
    """
    # Handle None and non-string inputs
    if term is None:
        return ""
    
    if not isinstance(term, str):
        term = str(term)
    
    # Normalize whitespace and check if empty
    term = term.strip()
    if not term:
        return ""
    
    # Simple case: no pipe separator
    if '|' not in term:
        return ' '.join(term.split())
    
    # Extract first non-empty option from pipe-separated list
    for part in term.split('|'):
        cleaned_part = part.strip()
        if cleaned_part:
            # Normalize internal whitespace and return
            return ' '.join(cleaned_part.split())
    
    # All parts were empty
    return ""


def clean_term_for_display(term: str) -> str:
    """Clean a term for display and LLM processing by removing search operators.
    
    This function removes both pipe separators (keeping only the first synonym)
    and ampersands (replacing with spaces) to create clean, human-readable terms
    suitable for display in output JSON and prompts sent to LLMs.
    
    Examples:
        "cardiovascular&disease" -> "cardiovascular disease"
        "hormone&therapy|HT|hormone replacement therapy" -> "hormone therapy"
        "cancer|tumor" -> "cancer"
        "lung&cancer|lung&tumor" -> "lung cancer"
        None -> ""
        "" -> ""
    
    Args:
        term: String that may contain pipes and/or ampersands, or None
        
    Returns:
        Clean string with pipes collapsed to first option and ampersands 
        replaced with spaces. Returns empty string for None or empty inputs.
    """
    # First strip pipes (keeps only first synonym)
    term = strip_pipe(term)
    
    # Then replace ampersands with spaces
    if term:
        term = term.replace("&", " ")
        # Normalize whitespace after replacement
        term = ' '.join(term.split())
    
    return term


def apply_a_term_suffix(a_term: str, config: Config) -> str:
    """Apply configured A_TERM_SUFFIX to the given term if configured.
    
    Args:
        a_term: The A term to potentially modify
        config: Config object containing global_settings
        
    Returns:
        The a_term with suffix appended if configured, otherwise unchanged
    """
    suffix = config.global_settings.get("A_TERM_SUFFIX")
    if suffix:
        return a_term + suffix
    return a_term


def sanitize_term_for_filename(term: str, max_len: int = 80) -> str:
    """Canonicalize and truncate a term for safe filename usage across all job types.

    - Uses strip_pipe to canonicalize synonyms (token before '|')
    - Truncates to max_len characters
    - Replaces '/' with '_'
    """
    canonical = strip_pipe(term)
    if isinstance(canonical, str) and len(canonical) > max_len:
        canonical = canonical[:max_len]
    return canonical.replace("/", "_")


def normalize_entries(value):
    """Return a flat list of individual abstracts.

    Handles cases where input is a single concatenated string containing
    multiple abstracts separated by the '===END OF ABSTRACT===' sentinel.
    """
    segments = []
    # Normalize to list for iteration
    if isinstance(value, list):
        iterable = value
    elif isinstance(value, str):
        iterable = [value]
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, str):
            # Coerce non-strings defensively
            segments.append(str(item))
            continue

        text = item.strip()
        if not text:
            continue

        if '===END OF ABSTRACT===' in text:
            parts = [p.strip() for p in text.split('===END OF ABSTRACT===') if p.strip()]
            # Re-append sentinel to each piece to preserve downstream expectations
            segments.extend([f"{p}===END OF ABSTRACT===" for p in parts])
        else:
            # Fallback: treat as a single abstract entry
            segments.append(text)
    return segments


def extract_json_from_markdown(s: str) -> dict:
    """Extract JSON from markdown formatted text.
    
    Finds ```json ... ```; falls back to first {...}
    """
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if not m:
        m = re.search(r"(\{.*\})", s, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(1))


def write_to_json(data, file_path, output_directory, config):
    """Write data to JSON file with sanitized filename.
    
    Args:
        data: Data to write to JSON
        file_path: Base filename for the JSON file
        output_directory: Directory to write the file to
        config: Config object with logger
    """
    logger = config.logger
    # Sanitize file_path by replacing ',', '[', ']', and ' ' with '_'
    file_path = file_path.replace(",", "_").replace("[", "_").replace("]", "_").replace(" ", "_").replace("'", "_")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    file_path = os.path.join(output_directory, file_path)
    logger.debug(f" IN WRITE TO JSON   File path: {file_path}")
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


class Config:
    def __init__(self, config_path: str):
        self.job_config_path = config_path
        
        with open(self.job_config_path, "r") as config_file:
            self.job_config = json.load(config_file)
        self.global_settings = self.job_config["GLOBAL_SETTINGS"]
        
        # Add API configuration first
        self.model = self.global_settings["MODEL"]
        self.max_retries = self.global_settings["MAX_RETRIES"]
        self.retry_delay = self.global_settings["RETRY_DELAY"]
        
        self.log_level_str = self.global_settings.get("LOG_LEVEL", "INFO").upper()
        self.logger = self.setup_logger()

        self.secrets_path = os.path.join(os.path.dirname(config_path), "secrets.json")
        if not os.path.exists(self.secrets_path):
            self.create_secrets_file()
            
        self.secrets = self.load_secrets()
        self.validate_secrets()
        
        # Initialize OpenAI/DeepSeek client

        is_deepseek = self.model == "r1"
        key_name = "DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY"
        api_key = self.secrets[key_name]
        
        client_kwargs = {"api_key": api_key}
        if is_deepseek:
            client_kwargs["base_url"] = "https://api.deepseek.com/v1"
        
        self.llm_client = openai.OpenAI(**client_kwargs)
        self.logger.debug(f"Initialized {'DeepSeek' if is_deepseek else 'OpenAI'} client")
        
        self.km_output_dir = None
        self.km_output_base_name = None
        self.filtered_tsv_name = None
        self.debug_tsv_name = None
        
        # Hypotheses and job settings should be loaded BEFORE term lists
        self.km_hypothesis = self.job_config["KM_hypothesis"]
        self.skim_hypotheses = self.job_config["SKIM_hypotheses"]
        self.job_type = self.job_config.get("JOB_TYPE")
        self.filter_config = self.job_config["relevance_filter"]
        self.debug = self.filter_config["DEBUG"]
        self.test_leakage = self.filter_config["TEST_LEAKAGE"]
        self.post_n = self.global_settings["POST_N"]
        self.top_n_articles_most_cited = self.global_settings["TOP_N_ARTICLES_MOST_CITED"]
        self.top_n_articles_most_recent = self.global_settings["TOP_N_ARTICLES_MOST_RECENT"]
        self.outdir_suffix = self.global_settings["OUTDIR_SUFFIX"]
        self.min_word_count = self.global_settings["MIN_WORD_COUNT"]
        self.iterations = self.global_settings.get("iterations", False)
        self.current_iteration = 0
        self.temperature = self.filter_config["TEMPERATURE"]
        self.top_p = self.filter_config["TOP_P"]
        self.max_cot_tokens = self.filter_config["MAX_COT_TOKENS"]
        
        # Validate configuration settings
        self._validate_job_settings()
        self._validate_relevance_filter_settings()
        
    def _validate_job_settings(self):
        """Validate job-specific settings for conflicts"""
        # Check for mutually exclusive settings in km_with_gpt
        if self.is_km_with_gpt:
            position_enabled = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("position", False)
            is_dch_enabled = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("is_dch", False)
            
            if position_enabled and is_dch_enabled:
                raise ValueError(
                    "Configuration error: 'position' and 'is_dch' cannot both be true. "
                    "Position mode pairs terms by index (A1-B1, A2-B2), while DCH mode "
                    "compares exactly 2 B terms against each A term. Please set one to false."
                )
    
    def _validate_relevance_filter_settings(self):
        """Validate required relevance filter settings for Triton inference"""
        required_params = {
            "TEMPERATURE": self.temperature,
            "TOP_P": self.top_p,
            "MAX_COT_TOKENS": self.max_cot_tokens
        }
        
        missing = [name for name, value in required_params.items() if value is None]
        if missing:
            raise ValueError(
                f"Missing required relevance_filter parameters in config.json: {', '.join(missing)}"
            )
        
    def load_km_output(self, km_output_path: str):
        """Load TSV data directly from file path"""
        self.data = pd.read_csv(km_output_path, sep='\t')
        
        # Configure output paths
        self.km_output_dir = os.path.dirname(km_output_path)
        self.km_output_base_name = os.path.splitext(os.path.basename(km_output_path))[0]
        if not os.path.exists(self.km_output_dir) and self.km_output_dir != "":
            os.makedirs(self.km_output_dir)

        self.filtered_tsv_name = os.path.join(
            self.km_output_dir, f"filtered_{self.km_output_base_name}.tsv"
        )
        self.debug_tsv_name = os.path.join(
            self.km_output_dir, f"debug_{self.km_output_base_name}.tsv"
        )

        self.add_file_handler()
        self._validate_data_columns()

    def _validate_data_columns(self):
        # Additional checks for specific configurations
        self.has_ac = (
            "ac_pmid_intersection" in self.data.columns
            and len(self.data["ac_pmid_intersection"].value_counts()) > 0
        )

        print(f"Job type detected. Running {self.job_type}.")
        if self.is_skim_with_gpt:
            assert (
                "c_term" in self.data.columns
            ), "Input TSV must have c_term if running skim_with_gpt"
            assert (
                "bc_pmid_intersection" in self.data.columns
            ), "Input TSV must have a bc_pmid_intersection."

        assert (
            "ab_pmid_intersection" in self.data.columns
        ), "Input TSV must have an ab_pmid_intersection."
        assert "a_term" in self.data.columns, "Input TSV must have an a_term."
        assert "b_term" in self.data.columns, "Input TSV must have a b_term."

    def add_file_handler(self, output_dir=None):
        """Add file handler to logger after output directory is known"""
        # Use provided output directory or default to km_output_dir
        log_dir = output_dir if output_dir else self.km_output_dir
        
        if log_dir:
            # Remove existing file handlers to avoid duplicate logging
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
                    
            log_file = os.path.join(log_dir, "SKiM-GPT.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - SKiM-GPT - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)

    def setup_logger(self) -> logging.Logger:
        """Configure console logging initially"""
        logger = logging.getLogger("SKiM-GPT")
        logger.setLevel(getattr(logging, self.log_level_str, logging.INFO))
        logger.propagate = False
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers = []

        # Define a more detailed formatter that includes function and file information
        detailed_formatter = logging.Formatter(
            '%(asctime)s - SKiM-GPT - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler only initially
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        logger.addHandler(console_handler)
        
        return logger

    def __getstate__(self):
        """Prepare Config for pickling - exclude unpicklable objects"""
        state = self.__dict__.copy()
        # Remove unpicklable objects (logger has thread locks, llm_client has connections)
        state['logger'] = None
        state['llm_client'] = None
        return state

    def __setstate__(self, state):
        """Restore Config after unpickling - reconstruct unpicklable objects"""
        self.__dict__.update(state)
        
        # Reconstruct logger
        self.logger = self.setup_logger()
        
        # Reconstruct OpenAI/DeepSeek client
        is_deepseek = self.model == "r1"
        key_name = "DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY"
        api_key = self.secrets[key_name]
        
        client_kwargs = {"api_key": api_key}
        if is_deepseek:
            client_kwargs["base_url"] = "https://api.deepseek.com/v1"
        
        self.llm_client = openai.OpenAI(**client_kwargs)

    def _load_term_lists(self):
        """Load appropriate term lists based on job configuration"""
        if self.is_skim_with_gpt:
            self._load_skim_terms()
        elif self.job_type == "km_with_gpt":
            self._load_km_terms()
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

    def _load_skim_terms(self):
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]
        self.logger.debug(f"Loading skim terms for {self.job_type}")
        self.logger.debug(f"Job settings: {job_settings}")
        
        # Load C terms
        c_terms_file = job_settings["C_TERMS_FILE"]
        self.c_terms = self._read_terms_from_file(c_terms_file)
        
        # Load B terms
        b_terms_file = job_settings["B_TERMS_FILE"]
        self.b_terms = self._read_terms_from_file(b_terms_file)

        # Load A terms
        if job_settings.get("A_TERM_LIST", False):
            a_terms_file = job_settings["A_TERMS_FILE"]
            self.a_terms = self._read_terms_from_file(a_terms_file)
        else:
            self.a_terms = [self.global_settings["A_TERM"]]

    def _load_km_terms(self):
        """Load terms for km_with_gpt job type (supports DCH via is_dch flag)"""
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]
        
        # Load A terms
        if job_settings.get("A_TERM_LIST", False):
            a_terms_file = job_settings["A_TERMS_FILE"]
            self.a_terms = self._read_terms_from_file(a_terms_file)
        else:
            self.a_terms = [self.global_settings["A_TERM"]]

        # Load B terms for KM workflow
        b_terms_file = job_settings["B_TERMS_FILE"]
        self.b_terms = self._read_terms_from_file(b_terms_file)
        
        # DCH validation: must have exactly two B terms
        if self.is_dch:
            if len(self.b_terms) != 2:
                raise ValueError(
                    f"DCH mode requires exactly 2 B terms, found {len(self.b_terms)} in {b_terms_file}"
                )

    def _read_terms_from_file(self, file_path: str) -> list:
        """Read terms from a text file (one term per line)"""
        try:
            with open(file_path, "r") as f:
                terms = [line.strip() for line in f if line.strip()]
                self.logger.debug(f"Read {len(terms)} terms from {file_path}")
                return terms
        except FileNotFoundError:
            self.logger.error(f"Terms file not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading terms file {file_path}: {str(e)}")
            raise

    @property
    def job_specific_settings(self):
        return self.job_config["JOB_SPECIFIC_SETTINGS"][self.job_type]

    @property
    def sort_column(self):
        return self.job_specific_settings.get("SORT_COLUMN", "ab_sort_ratio")

    @property
    def position(self):
        return self.job_specific_settings.get("position", False)

    @property
    def fet_thresholds(self):
        if self.job_type == "skim_with_gpt":
            return {
                "ab": self.job_specific_settings["ab_fet_threshold"],
                "bc": self.job_specific_settings["bc_fet_threshold"]
            }
        else:
            return {
                "ab": self.job_specific_settings["ab_fet_threshold"]
            }

    @property
    def censor_year_upper(self):
        return self.job_specific_settings.get("censor_year_upper", self.job_specific_settings.get("censor_year", 2024))

    @property
    def censor_year_lower(self):
        # Default lower bound is zero (include all years)
        return self.job_specific_settings.get("censor_year_lower", 0)

    @property
    def is_km_with_gpt(self):
        """Check if job type is km_with_gpt"""
        return self.job_type == "km_with_gpt"

    @property
    def is_skim_with_gpt(self):
        """Check if job type is skim_with_gpt"""
        return self.job_type == "skim_with_gpt"

    @property
    def is_dch(self):
        """Check if in DCH (direct comparison hypothesis) mode"""
        return self.is_km_with_gpt and self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("is_dch", False)


    def create_secrets_file(self):
        """Create secrets.json from environment variables if missing"""
        secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PUBMED_API_KEY": os.getenv("PUBMED_API_KEY"),
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
        }
        
        # Only check for required keys based on model
        required_keys = ["PUBMED_API_KEY"]
        if self.model == "r1":
            required_keys.append("DEEPSEEK_API_KEY")
        else:
            required_keys.append("OPENAI_API_KEY")
        
        missing = [k for k in required_keys if not secrets.get(k)]
        if missing:
            raise ValueError(
                f"Cannot create secrets.json - missing environment variables: {', '.join(missing)}"
            )
        
        with open(self.secrets_path, "w") as f:
            json.dump(secrets, f, indent=2)
        
        os.chmod(self.secrets_path, 0o600)  # Restrict permissions
        self.logger.info(f"Created secrets file at {self.secrets_path}")

    def load_secrets(self) -> dict:
        """Load secrets from JSON file"""
        try:
            with open(self.secrets_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Secrets file missing at {self.secrets_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in {self.secrets_path}")

    def validate_secrets(self):
        """Validate required secrets exist"""
        required = ["PUBMED_API_KEY"]
        
        # Add API key requirement based on model
        if self.model == "r1":
            required.append("DEEPSEEK_API_KEY")
        else:
            required.append("OPENAI_API_KEY")
        
        missing = [key for key in required if not self.secrets.get(key)]
        if missing:
            raise ValueError(f"Missing secrets in {self.secrets_path}: {', '.join(missing)}")

    def set_iteration(self, iteration_number: int):
        """Set the current iteration and update output paths accordingly"""
        self.current_iteration = iteration_number
        # Update output paths for this iteration
        self._update_output_paths_for_iteration()
        
    def _update_output_paths_for_iteration(self):
        """Update output paths based on current iteration"""
        if self.iterations and self.km_output_dir and self.current_iteration > 0:
            # Create iteration-specific output directory
            iteration_dir = os.path.join(self.km_output_dir, f"iteration_{self.current_iteration}")
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
                
            # Update file paths to use iteration-specific directory
            self.filtered_tsv_name = os.path.join(
                iteration_dir, f"filtered_{self.km_output_base_name}.tsv"
            )
            self.debug_tsv_name = os.path.join(
                iteration_dir, f"debug_{self.km_output_base_name}.tsv"
            )
            
            # Update logger with new file handler for this iteration
            self.add_file_handler(iteration_dir)
        else:

            self.filtered_tsv_name = os.path.join(
                self.km_output_dir, f"filtered_{self.km_output_base_name}.tsv"
            )
            self.debug_tsv_name = os.path.join(
                self.km_output_dir, f"debug_{self.km_output_base_name}.tsv"
            )
            
            # Update logger with new file handler for the base directory
            self.add_file_handler(self.km_output_dir)
            
