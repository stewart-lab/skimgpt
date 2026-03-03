from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable

import numpy as np
import openai
import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logger = logging.getLogger(__name__)


def configure_logging(level_str: str = "INFO") -> None:
    """Configure the root logger with a console handler (idempotent)."""
    root = logging.getLogger()
    level = getattr(logging, level_str.upper(), logging.INFO)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        root.addHandler(handler)


def setup_wrapper_logger(parent_dir: str, job_type: str) -> None:
    """Configure the root logger with console and file handlers for wrapper runs.

    Clears all existing handlers, then delegates to :func:`configure_logging`
    and :func:`add_file_handler` to set up a fresh console + file pair.
    """
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    configure_logging("INFO")
    add_file_handler(parent_dir, filename=f"{job_type}_wrapper.log")


def add_file_handler(log_dir: str, filename: str = "SKiM-GPT.log") -> None:
    """Add a FileHandler to the root logger (replaces any existing FileHandler)."""
    root = logging.getLogger()
    # Remove existing file handlers to avoid duplicate logging
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    log_file = os.path.join(log_dir, filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(file_handler)


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


def extract_pmids(text: str) -> list[str]:
    """Extract all PMIDs from text.
    
    Args:
        text: String containing PMID references
        
    Returns:
        List of PMID numbers as strings
    """
    return PMID_PATTERN.findall(text)


class RaggedTensor:
    def __init__(self, data: list, break_point: list[int] | None = None):
        self.data = data
        self.break_point = break_point if break_point is not None else []
        self.index = 0
        self.get_shape()

    def get_shape(self) -> None:
        if self.is_2d():
            self.shape = [len(i) for i in self.data]
        else:
            self.shape = len(self.data)

    def is_2d(self) -> bool:
        return bool(self.data) and isinstance(self.data[0], list)

    # Duplicates each element in data according to the shape_list
    def expand(self, shape_list: list) -> RaggedTensor:
        assert (
            not self.is_2d()
        ), "Data must be 1D before calling expand. Call flatten first?"
        assert self.shape == len(
            shape_list
        ), "The length of shape list must equal the length of data"

        expanded = []
        for item, count in zip(self.data, shape_list):
            expanded.extend([item] * count)

        return RaggedTensor(expanded)

    def flatten(self) -> RaggedTensor:
        if self.is_2d():
            return RaggedTensor([item for lst in self.data for item in lst])
        return self

    # Splits the data depending on the index
    def split(self) -> list[RaggedTensor] | tuple[RaggedTensor, RaggedTensor]:
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
    def reshape(self, shape: list) -> None:
        assert not self.is_2d(), "Reshape only works with 1D tensors."
        assert self.shape == sum(
            shape
        ), "The shape of the tensor should be equal to the sum of the wanted shape."
        output = []
        running_length = 0
        for length in shape:
            output.append(self.data[running_length : running_length + length])
            running_length += length

        self.data = output
        self.get_shape()

    # Applies a mask to the tensor
    def apply_filter(self, mask: RaggedTensor) -> None:
        assert (
            self.shape == mask.shape
        ), "Filtering only works when the shapes are the same"
        if self.is_2d():
            for i, (data_row, mask_row) in enumerate(zip(self.data, mask.data)):
                boolean_mask = np.array(mask_row) == 1
                self.data[i] = list(np.array(data_row)[boolean_mask])
        else:
            boolean_mask = np.array(mask.data) == 1
            self.data = list(np.array(self.data)[boolean_mask])

    # Applies a function to the tensor
    def map(self, func: Callable, *args) -> RaggedTensor:
        assert not self.is_2d(), "Map only works with 1D tensors"
        return RaggedTensor([func(i, *args) for i in self.data], self.break_point)

    # Simply concatenates two ragged tensors and appends to the break_point list
    def __add__(self, other: RaggedTensor) -> RaggedTensor:
        assert not self.is_2d(), "Adding only works with flattened tensors"
        break_point = self.shape
        return RaggedTensor(self.data + other.data, self.break_point + [break_point])

    def __str__(self) -> str:
        return str(self.data)

    def __iter__(self):
        return iter(self.flatten().data)

    def __getitem__(self, index: int):
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
    if len(canonical) > max_len:
        canonical = canonical[:max_len]
    return canonical.replace("/", "_")


ABSTRACT_DELIMITER = "===END OF ABSTRACT==="


def split_abstract_entries(value: str | list, *, keep_delimiter: bool = False) -> list[str]:
    """Split abstract text on the ``===END OF ABSTRACT===`` delimiter.

    Args:
        value: A concatenated string (or list of strings) containing abstracts
            separated by the delimiter.
        keep_delimiter: If *True*, re-append the delimiter to each entry
            (needed by the relevance pipeline).

    Returns:
        A flat list of individual abstract entries, stripped of excess whitespace.
    """
    segments: list[str] = []
    if isinstance(value, list):
        iterable = value
    elif isinstance(value, str):
        iterable = [value]
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, str):
            segments.append(str(item))
            continue

        text = item.strip()
        if not text:
            continue

        if ABSTRACT_DELIMITER in text:
            parts = [p.strip() for p in text.split(ABSTRACT_DELIMITER) if p.strip()]
            if keep_delimiter:
                segments.extend([f"{p}{ABSTRACT_DELIMITER}" for p in parts])
            else:
                segments.extend(parts)
        else:
            segments.append(text)
    return segments


def join_abstract_entries(entries: list[str]) -> str:
    """Rejoin abstract entries with the standard delimiter and newlines."""
    return f"{ABSTRACT_DELIMITER}\n\n".join(entries) + f"{ABSTRACT_DELIMITER}\n\n"


def normalize_entries(value: str | list) -> list[str]:
    """Return a flat list of individual abstracts with the delimiter preserved.

    Thin wrapper around :func:`split_abstract_entries` kept for backward
    compatibility with the relevance pipeline.
    """
    return split_abstract_entries(value, keep_delimiter=True)


def extract_json_from_markdown(s: str) -> dict:
    """Extract JSON from markdown formatted text.
    
    Finds ```json ... ```; falls back to first {...}
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if not m:
        m = re.search(r"(\{.*\})", s, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(1))


def write_to_json(data: dict | list, file_path: str, output_directory: str) -> None:
    """Write data to JSON file with sanitized filename.

    Args:
        data: Data to write to JSON
        file_path: Base filename for the JSON file
        output_directory: Directory to write the file to
    """
    # Sanitize file_path by replacing special characters with '_'
    file_path = re.sub(r"[,\[\] ']", "_", file_path)
    os.makedirs(output_directory, exist_ok=True)

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
        configure_logging(self.log_level_str)

        self.secrets_path = os.path.join(os.path.dirname(config_path), "secrets.json")
        if not os.path.exists(self.secrets_path):
            self.create_secrets_file()
            
        self.secrets = self.load_secrets()
        self.validate_secrets()
        
        # Initialize OpenAI/DeepSeek client
        self.llm_client = self._build_llm_client()
        
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
        
        # HTCondor fallback configuration (optional — used when Triton is unavailable)
        htcondor_config = self.job_config.get("HTCONDOR", {})
        self.using_htcondor = bool(htcondor_config)
        self.collector_host = htcondor_config.get("collector_host")
        self.submit_host = htcondor_config.get("submit_host")
        self.docker_image = htcondor_config.get("docker_image")
        
        # Validate configuration settings
        self._validate_job_settings()
        self._validate_relevance_filter_settings()
        
    def _validate_job_settings(self) -> None:
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
    
    def _validate_relevance_filter_settings(self) -> None:
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
        if self.km_output_dir:
            os.makedirs(self.km_output_dir, exist_ok=True)

        self.filtered_tsv_name = os.path.join(
            self.km_output_dir, f"filtered_{self.km_output_base_name}.tsv"
        )
        self.debug_tsv_name = os.path.join(
            self.km_output_dir, f"debug_{self.km_output_base_name}.tsv"
        )

        if self.km_output_dir:
            add_file_handler(self.km_output_dir)
        self._validate_data_columns()

    def _validate_data_columns(self) -> None:
        # Additional checks for specific configurations
        self.has_ac = (
            "ac_pmid_intersection" in self.data.columns
            and len(self.data["ac_pmid_intersection"].value_counts()) > 0
        )

        logger.info(f"Job type detected. Running {self.job_type}.")
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

    def __getstate__(self) -> dict:
        """Prepare Config for pickling - exclude unpicklable objects"""
        state = self.__dict__.copy()
        # Remove unpicklable objects (llm_client has connections)
        state['llm_client'] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore Config after unpickling - reconstruct unpicklable objects"""
        self.__dict__.update(state)

        # Re-apply logging configuration (handlers aren't pickled)
        configure_logging(self.log_level_str)

        # Reconstruct OpenAI/DeepSeek client
        self.llm_client = self._build_llm_client()

    def _build_llm_client(self) -> openai.OpenAI:
        """Build and return an OpenAI/DeepSeek client based on model config."""
        is_deepseek = self.model == "r1"
        key_name = "DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY"
        api_key = self.secrets[key_name]

        client_kwargs = {"api_key": api_key}
        if is_deepseek:
            client_kwargs["base_url"] = "https://api.deepseek.com/v1"

        client = openai.OpenAI(**client_kwargs)
        logger.debug(f"Initialized {'DeepSeek' if is_deepseek else 'OpenAI'} client")
        return client

    def load_term_lists(self) -> None:
        """Load appropriate term lists based on job configuration."""
        if self.is_skim_with_gpt:
            self._load_skim_terms()
        elif self.job_type == "km_with_gpt":
            self._load_km_terms()
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

    def _load_a_terms(self, job_settings: dict) -> None:
        """Load A terms from file or single-term config (shared by skim and km loaders)."""
        if job_settings.get("A_TERM_LIST", False):
            self.a_terms = self._read_terms_from_file(job_settings["A_TERMS_FILE"])
        else:
            self.a_terms = [self.global_settings["A_TERM"]]

    def _load_skim_terms(self) -> None:
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]
        logger.debug(f"Loading skim terms for {self.job_type}")
        logger.debug(f"Job settings: {job_settings}")

        self.c_terms = self._read_terms_from_file(job_settings["C_TERMS_FILE"])
        self.b_terms = self._read_terms_from_file(job_settings["B_TERMS_FILE"])
        self._load_a_terms(job_settings)

    def _load_km_terms(self) -> None:
        """Load terms for km_with_gpt job type (supports DCH via is_dch flag)"""
        job_settings = self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]

        self._load_a_terms(job_settings)
        self.b_terms = self._read_terms_from_file(job_settings["B_TERMS_FILE"])

        # DCH validation: must have exactly two B terms
        if self.is_dch and len(self.b_terms) != 2:
            raise ValueError(
                f"DCH mode requires exactly 2 B terms, found {len(self.b_terms)} "
                f"in {job_settings['B_TERMS_FILE']}"
            )

    def _read_terms_from_file(self, file_path: str) -> list[str]:
        """Read terms from a text file (one term per line)"""
        try:
            with open(file_path, "r") as f:
                terms = [line.strip() for line in f if line.strip()]
                logger.debug(f"Read {len(terms)} terms from {file_path}")
                return terms
        except FileNotFoundError:
            logger.error(f"Terms file not found: {file_path}")
            raise

    @property
    def job_specific_settings(self) -> dict:
        return self.job_config["JOB_SPECIFIC_SETTINGS"][self.job_type]

    @property
    def sort_column(self) -> str:
        return self.job_specific_settings.get("SORT_COLUMN", "ab_sort_ratio")

    @property
    def position(self) -> bool:
        return self.job_specific_settings.get("position", False)

    @property
    def censor_year_upper(self) -> int:
        return self.job_specific_settings.get(
            "censor_year_upper",
            self.job_specific_settings.get("censor_year", 2024),
        )

    @property
    def censor_year_lower(self) -> int:
        # Default lower bound is zero (include all years)
        return self.job_specific_settings.get("censor_year_lower", 0)

    @property
    def is_km_with_gpt(self) -> bool:
        return self.job_type == "km_with_gpt"

    @property
    def is_skim_with_gpt(self) -> bool:
        return self.job_type == "skim_with_gpt"

    @property
    def is_dch(self) -> bool:
        return (
            self.is_km_with_gpt
            and self.job_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("is_dch", False)
        )

    @property
    def _required_secret_keys(self) -> list[str]:
        """Return list of required secret key names based on model config."""
        keys = ["PUBMED_API_KEY"]
        if self.model == "r1":
            keys.append("DEEPSEEK_API_KEY")
        else:
            keys.append("OPENAI_API_KEY")
        return keys

    def create_secrets_file(self) -> None:
        """Create secrets.json from environment variables if missing"""
        secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PUBMED_API_KEY": os.getenv("PUBMED_API_KEY"),
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
        }

        missing = [k for k in self._required_secret_keys if not secrets.get(k)]
        if missing:
            raise ValueError(
                f"Cannot create secrets.json - missing environment variables: {', '.join(missing)}"
            )
        
        with open(self.secrets_path, "w") as f:
            json.dump(secrets, f, indent=2)
        
        os.chmod(self.secrets_path, 0o600)  # Restrict permissions
        logger.info(f"Created secrets file at {self.secrets_path}")

    def load_secrets(self) -> dict:
        """Load secrets from JSON file"""
        try:
            with open(self.secrets_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Secrets file missing at {self.secrets_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in {self.secrets_path}")

    def validate_secrets(self) -> None:
        """Validate required secrets exist"""
        missing = [key for key in self._required_secret_keys if not self.secrets.get(key)]
        if missing:
            raise ValueError(f"Missing secrets in {self.secrets_path}: {', '.join(missing)}")

    def set_iteration(self, iteration_number: int) -> None:
        """Set the current iteration and update output paths accordingly"""
        self.current_iteration = iteration_number
        # Update output paths for this iteration
        self._update_output_paths_for_iteration()
        
    def _update_output_paths_for_iteration(self) -> None:
        """Update output paths based on current iteration"""
        if self.iterations and self.km_output_dir and self.current_iteration > 0:
            output_dir = os.path.join(self.km_output_dir, f"iteration_{self.current_iteration}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.km_output_dir

        self.filtered_tsv_name = os.path.join(
            output_dir, f"filtered_{self.km_output_base_name}.tsv"
        )
        self.debug_tsv_name = os.path.join(
            output_dir, f"debug_{self.km_output_base_name}.tsv"
        )
        add_file_handler(output_dir)


def get_hypothesis(
    config: Config,
    a_term: str = None,
    b_term: str = None,
    c_term: str = None,
    *,
    relationship_type: str = None,
    clean_terms: bool = True,
) -> str:
    """Build a hypothesis string from config templates.

    Two calling conventions:

    1. **Relevance pipeline** (``relationship_type`` omitted): infers the
       relationship from which terms are non-None.
    2. **Classifier pipeline** (``relationship_type`` provided): selects the
       template directly.  Use ``"A_B"``, ``"A_B_C"``, or ``"A_C"``.

    Args:
        config: Config object holding hypothesis templates.
        a_term, b_term, c_term: Terms to substitute.
        relationship_type: Explicit relationship selector for the classifier
            path.  One of ``"A_B"``, ``"A_B_C"``, ``"A_C"``.
        clean_terms: When *True* (default) the terms are cleaned for display
            (pipes/ampersands removed).  Set to *False* when the caller has
            already cleaned them.
    """
    if clean_terms:
        if a_term:
            a_term = clean_term_for_display(a_term)
        if b_term:
            b_term = clean_term_for_display(b_term)
        if c_term:
            c_term = clean_term_for_display(c_term)

    fmt = {"a_term": a_term or "", "b_term": b_term or "", "c_term": c_term or ""}

    # --- Classifier path: explicit relationship_type -----------------------
    if relationship_type is not None:
        if config.is_km_with_gpt and relationship_type == "A_B":
            return config.km_hypothesis.format(**fmt)
        template_map = {
            "A_B_C": config.skim_hypotheses.get("ABC"),
            "A_C":   config.skim_hypotheses.get("AC"),
            "A_B":   config.skim_hypotheses.get("AB"),
        }
        template = template_map.get(relationship_type)
        if template:
            return template.format(**fmt)
        logger.warning("Unknown relationship_type %r", relationship_type)
        return ""

    # --- Relevance path: infer from which terms are provided ---------------
    if config.is_km_with_gpt:
        return config.km_hypothesis.format(**fmt)

    if config.is_skim_with_gpt:
        if a_term and b_term and not c_term:
            return config.skim_hypotheses["AB"].format(**fmt)
        if b_term and c_term and not a_term:
            return config.skim_hypotheses["BC"].format(**fmt)
        if a_term and c_term and not b_term:
            return config.skim_hypotheses["rel_AC"].format(**fmt)

    return f"No valid hypothesis for the provided {config.job_type}."
