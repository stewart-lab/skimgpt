from __future__ import annotations
import numpy as np
import pandas as pd
import json
import os
import logging  

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
    def split(self) -> list[RaggedTensor]:
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


class Config:
    def __init__(self, config_path: str):
        self.job_config_path = config_path
        
        with open(self.job_config_path, "r") as config_file:
            self.job_config = json.load(config_file)
        self.global_settings = self.job_config["GLOBAL_SETTINGS"]
        
        # Add API configuration first
        self.model = self.global_settings["MODEL"]
        self.rate_limit = self.global_settings["RATE_LIMIT"]
        self.delay = self.global_settings["DELAY"]
        self.max_retries = self.global_settings["MAX_RETRIES"]
        self.retry_delay = self.global_settings["RETRY_DELAY"]
        
        self.log_level_str = self.global_settings.get("LOG_LEVEL", "INFO").upper()
        self.logger = self.setup_logger()

        self.secrets_path = os.path.join(os.path.dirname(config_path), "secrets.json")
        if not os.path.exists(self.secrets_path):
            self.create_secrets_file()
            
        self.secrets = self.load_secrets()
        self.validate_secrets()
        
        self.km_output_dir = None
        self.km_output_base_name = None
        self.filtered_tsv_name = None
        self.debug_tsv_name = None
        
        # Hypotheses and job settings should be loaded BEFORE term lists
        self.km_hypothesis = self.job_config["KM_hypothesis"]
        self.km_direct_comp_hypothesis = self.job_config["KM_direct_comp_hypothesis"]
        self.skim_hypotheses = self.job_config["SKIM_hypotheses"]
        self.job_type = self.job_config.get("JOB_TYPE")
        self.filter_config = self.job_config["abstract_filter"]
        self.debug = self.filter_config["DEBUG"]
        self.test_leakage = self.filter_config["TEST_LEAKAGE"]
        self.test_leakage_type = self.filter_config["TEST_LEAKAGE_TYPE"]
        self.is_km_with_gpt = self.job_type == "km_with_gpt"
        self.is_skim_with_gpt = self.job_type == "skim_with_gpt"
        self.is_km_with_gpt_direct_comp = self.job_type == "km_with_gpt_direct_comp"
        self.evaluate_single_abstract = self.job_config["Evaluate_single_abstract"]
        self.post_n = self.global_settings["POST_N"]
        self.top_n_articles_most_cited = self.global_settings["TOP_N_ARTICLES_MOST_CITED"]
        self.top_n_articles_most_recent = self.global_settings["TOP_N_ARTICLES_MOST_RECENT"]
        
        self.outdir_suffix = self.global_settings["OUTDIR_SUFFIX"]
        self.min_word_count = self.global_settings["MIN_WORD_COUNT"]
        # Add API configuration
        self.km_api_url = self.global_settings["API_URL"]
        self.model = self.global_settings["MODEL"]
        self.rate_limit = self.global_settings["RATE_LIMIT"]
        self.delay = self.global_settings["DELAY"]
        self.max_retries = self.global_settings["MAX_RETRIES"]
        self.retry_delay = self.global_settings["RETRY_DELAY"]
        
        # Add iterations configuration
        self.iterations = self.global_settings.get("iterations", False)
        self.current_iteration = 0
        
        # Add HTCondor configuration
        self.htcondor_config = self.job_config.get("HTCONDOR", {})
        
        # Add abstract filter configuration
        self.temperature = self.filter_config["TEMPERATURE"]
        self.top_k = self.filter_config["TOP_K"]
        self.top_p = self.filter_config["TOP_P"]
        self.max_cot_tokens = self.filter_config["MAX_COT_TOKENS"]
        
        # Add HTCondor configuration validation
        self._validate_htcondor_config()
        
    def load_km_output(self, km_output_path: str):
        """Load TSV data and configure output paths when km_output is available"""
        self.data = self.read_tsv_to_dataframe_from_files_txt(km_output_path)
        
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

        # Add file handler now that we have output directory
        self.add_file_handler()
        
        # Validate data columns
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

    @staticmethod
    def read_tsv_to_dataframe_from_files_txt(files_txt_path: str) -> pd.DataFrame:
        """
        Read the first file path from files.txt and load that TSV into a pandas DataFrame.

        Args:
            files_txt_path: Path to the files.txt file.

        Returns:
            pandas.DataFrame: DataFrame loaded from the first TSV file listed in files.txt.
                            Returns an empty DataFrame if files.txt is empty or file not found.
        """
        try:
            with open(files_txt_path, "r") as f:
                file_paths = [line.strip() for line in f.readlines() if line.strip()]
                
                if len(file_paths) > 1:
                    logging.error(f"Multiple files detected in {files_txt_path}: {file_paths}")
                    return pd.DataFrame()
                
                if not file_paths:
                    logging.warning(f"{files_txt_path} is empty. Returning empty DataFrame.")
                    return pd.DataFrame()
                
                first_file_path = file_paths[0] if file_paths else ""

        except FileNotFoundError:
            logging.error(f"{files_txt_path} not found.")
            return pd.DataFrame()

        try:
            return pd.read_csv(first_file_path, sep="\t")
        
        except FileNotFoundError:
            logging.error(f"File path '{first_file_path}' from {files_txt_path} not found.")
            return pd.DataFrame()

    def _load_term_lists(self):
        """Load appropriate term lists based on job configuration"""
        if self.is_skim_with_gpt:
            self._load_skim_terms()
        elif self.is_km_with_gpt_direct_comp:
            self._load_km_direct_comp_terms()
        elif not self.is_skim_with_gpt and not self.is_km_with_gpt_direct_comp:
            self._load_km_terms()
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

    def _load_skim_terms(self):
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]
        self.logger.debug(f"Loading skim terms for {self.job_type}")
        self.logger.debug(f"Job settings: {job_settings}")
        
        self.position = job_settings["position"]
        
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
        """Load terms for km_with_gpt job type"""
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]
        self.position = job_settings["position"]
        
        # Load A terms
        if job_settings.get("A_TERM_LIST", False):
            a_terms_file = job_settings["A_TERMS_FILE"]
            self.a_terms = self._read_terms_from_file(a_terms_file)
        else:
            self.a_terms = [self.global_settings["A_TERM"]]

        # Load B terms for KM workflow
        b_terms_file = job_settings["B_TERMS_FILE"]
        self.b_terms = self._read_terms_from_file(b_terms_file)

    def _load_km_direct_comp_terms(self):
        """Load terms for km_with_gpt_direct_comp job type"""
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt_direct_comp"]
        self.position = job_settings["position"]
        
        # Load A terms
        if job_settings.get("A_TERM_LIST", False):
            a_terms_file = job_settings["A_TERMS_FILE"]
            self.a_terms = self._read_terms_from_file(a_terms_file)
        else:
            self.a_terms = [self.global_settings["A_TERM"]]

        # Load B terms for KM direct comp workflow
        b_terms_file = job_settings["B_TERMS_FILE"]
        self.b_terms = self._read_terms_from_file_for_km_direct_comp(b_terms_file)

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
         
    def _read_terms_from_file_for_km_direct_comp(self, file_path: str) -> list:
        """Read terms from a text file (only one line allowed.)"""
        try:
            with open(file_path, "r") as f:
                line = f.readline().strip()
                if not line:
                    self.logger.error(f"Empty file: {file_path}")
                    raise ValueError(f"Empty file: {file_path}")
                terms = [line]
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
    def fet_thresholds(self):
        if self.job_type == "skim_with_gpt":
            return {
                "ab": self.job_specific_settings["skim"]["ab_fet_threshold"],
                "bc": self.job_specific_settings["skim"]["bc_fet_threshold"]
            }
        elif self.is_km_with_gpt_direct_comp:
            return {
                "ab": self.job_specific_settings["km_with_gpt_direct_comp"]["ab_fet_threshold"]
            }
        else:
            return {
                "ab": self.job_specific_settings["km_with_gpt"]["ab_fet_threshold"]
            }

    @property
    def censor_year(self):
        return self.job_specific_settings.get("censor_year", 2024)

    def _validate_htcondor_config(self):
        """Validate required HTCondor settings"""
        if not self.htcondor_config:
            self.logger.warning("No HTCondor configuration found in config file")
            return
            
        required_keys = [
            "collector_host", 
            "submit_host",
            "docker_image",
            "request_gpus",
            "request_cpus",
            "request_memory",
            "request_disk"
        ]
        
        missing = [key for key in required_keys if key not in self.htcondor_config]
        if missing:
            raise ValueError(f"Missing required HTCondor config keys: {', '.join(missing)}")

    @property
    def using_htcondor(self):
        """Check if HTCondor configuration is present and valid"""
        return bool(self.htcondor_config) and all([
            self.collector_host,
            self.submit_host,
            self.docker_image
        ])

    @property
    def collector_host(self):
        return self.htcondor_config.get("collector_host")

    @property
    def submit_host(self):
        return self.htcondor_config.get("submit_host")

    @property
    def docker_image(self):
        return self.htcondor_config.get("docker_image")

    @property
    def request_gpus(self):
        return self.htcondor_config.get("request_gpus", "1")

    @property
    def request_cpus(self):
        return self.htcondor_config.get("request_cpus", "1")

    @property
    def request_memory(self):
        return self.htcondor_config.get("request_memory", "24GB")

    @property
    def request_disk(self):
        return self.htcondor_config.get("request_disk", "50GB")

    def create_secrets_file(self):
        """Create secrets.json from environment variables if missing"""
        secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PUBMED_API_KEY": os.getenv("PUBMED_API_KEY"),
            "HTCONDOR_TOKEN": os.getenv("HTCONDOR_TOKEN"),
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
        }
        
        # Only check for required keys based on model
        required_keys = ["PUBMED_API_KEY", "HTCONDOR_TOKEN"]
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
        required = ["PUBMED_API_KEY", "HTCONDOR_TOKEN"]
        
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
            # When iterations is False or current_iteration is 0,
            # update file paths to use the base output directory
            self.filtered_tsv_name = os.path.join(
                self.km_output_dir, f"filtered_{self.km_output_base_name}.tsv"
            )
            self.debug_tsv_name = os.path.join(
                self.km_output_dir, f"debug_{self.km_output_base_name}.tsv"
            )
            
            # Update logger with new file handler for the base directory
            self.add_file_handler(self.km_output_dir)
            
