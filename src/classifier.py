import openai
import time
import re
from typing import Any
from src.utils import Config, strip_pipe, clean_term_for_display, extract_json_from_markdown, extract_pmids
from src import prompt_library as prompts_module  

# Constants
SCORE_NA_RESPONSE = {"score": "N/A", "decision": "N/A"}


def calculate_relevance_ratios(out_df, config: Config):
    logger = config.logger
    logger.debug(f" Complete Out df: {out_df.to_string()}")
    mask_columns = ["ab_mask", "bc_mask", "ac_mask"]
    logger.debug(f"  Mask columns: {mask_columns}")
    logger.debug(f"  Out df columns: {out_df.columns}")
    for col in mask_columns:
        if col in out_df.columns:
            prefix = col.split("_")[0]
            ratio_col = f"{prefix}_relevance_ratio"
            fraction_col = f"{prefix}_relevance_fraction"
            logger.debug(f"  Ratio col: {ratio_col}")
            logger.debug(f"  Fraction col: {fraction_col}")
            out_df[[ratio_col, fraction_col]] = (
                out_df[col]
                .apply(
                    lambda x: (
                        (sum(x) / len(x), f"{sum(x)}/{len(x)}")
                        if isinstance(x, list) and len(x) > 0
                        else (0, "0/0")
                    )
                )
                .tolist()
            )

    return out_df


def process_single_row(row, config: Config):
    logger = config.logger

    processed_results = {}

    if config.is_skim_with_gpt:
        a_term = row.get("a_term")
        b_term = row.get("b_term")
        c_term = row.get("c_term")
        
        # Clean terms for display and LLM processing
        a_term = clean_term_for_display(a_term)
        b_term = clean_term_for_display(b_term)
        c_term = clean_term_for_display(c_term)
        
        # Update row with cleaned terms so downstream functions use them
        row["a_term"] = a_term
        row["b_term"] = b_term
        row["c_term"] = c_term

        abc_result, abc_prompt, abc_urls = perform_analysis(
            row=row, config=config, relationship_type="A_B_C"
        )
        if abc_result or abc_prompt:
            processed_results["A_B_C_Relationship"] = {
                "a_term": a_term,
                "b_term": b_term,
                "c_term": c_term,
                "Relationship": f"{a_term} - {b_term} - {c_term}",
                "Hypothesis": config.skim_hypotheses["ABC"].format(
                    a_term=a_term, b_term=b_term, c_term=c_term
                ),
                "Result": abc_result,
                "Prompt": abc_prompt,
                "URLS": {
                    "AB": abc_urls.get("AB", []),
                    "BC": abc_urls.get("BC", []),
                },
            }

        ac_result, ac_prompt, ac_urls = perform_analysis(
            row=row, config=config, relationship_type="A_C"
        )
        if ac_result or ac_prompt:
            processed_results["A_C_Relationship"] = {
                "a_term": a_term,
                "b_term": b_term,
                "c_term": c_term,
                "Relationship": f"{a_term} - {c_term}",
                "Hypothesis": config.skim_hypotheses["AC"].format(
                    a_term=a_term, c_term=c_term
                ),
                "Result": ac_result,
                "Prompt": ac_prompt,
                "URLS": {
                    "AC": ac_urls.get("AC", []),
                },
            }

    elif config.is_dch:
        # DCH mode: direct hypothesis comparison (is_dch can only be True if is_km_with_gpt is True)
        hypothesis1 = row.get("hypothesis1")
        hypothesis2 = row.get("hypothesis2")
        # Hypotheses are already cleaned (strip_pipe applied to terms before formatting)
        result, prompt, urls = perform_analysis(
            row=row, config=config, relationship_type="A_B"
        )
        if result or prompt:
            processed_results["Hypothesis_Comparison"] = {
                "hypothesis1": hypothesis1,
                "hypothesis2": hypothesis2,
                "Result": result,
                "Prompt": prompt,
                "URLS": urls,
            }
        # For DCH we only require the two hypotheses; PMIDs are optional
        if not all([hypothesis1, hypothesis2]):
            logger.error(f"Missing hypotheses for DCH row.")
            return None
    elif config.is_km_with_gpt:
        # Standard km_with_gpt case (not DCH)
        a_term = row.get("a_term")
        b_term = row.get("b_term")

        if not all([a_term, b_term]):
            logger.error(f"Missing 'a_term' or 'b_term' in row.")
            return None

        # Clean terms for display and LLM processing
        a_term = clean_term_for_display(a_term)
        b_term = clean_term_for_display(b_term)
        
        # Update row with cleaned terms so downstream functions use them
        row["a_term"] = a_term
        row["b_term"] = b_term

        ab_result, ab_prompt, ab_urls = perform_analysis(
            row=row, config=config, relationship_type="A_B"
        )
        if ab_result or ab_prompt:
            processed_results["A_B_Relationship"] = {
                "a_term": a_term,
                "b_term": b_term,
                "Relationship": f"{a_term} - {b_term}",
                "Hypothesis": config.km_hypothesis.format(a_term=a_term, b_term=b_term),
                "Result": ab_result,
                "Prompt": ab_prompt,
                "URLS": {"AB": ab_urls.get("AB", [])},
            }
    else:
        logger.warning(f"Job type '{config.job_type}' is not specifically handled.")
    logger.debug(f"Processed results: {processed_results}")
    return processed_results if any(processed_results.values()) else None


def extract_pmids_and_generate_urls(text: Any, config: Config) -> list:
    logger = config.logger
    if not isinstance(text, str):
        logger.error(
            f"Expected string for 'text', but got {type(text)}. Converting to string."
        )
        text = str(text)

    # Find all PMIDs in the text
    pmids = extract_pmids(text)

    # Generate PubMed URLs
    pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]

    return pubmed_urls


def perform_analysis(row: dict, config: Config, relationship_type: str) -> tuple:
    logger = config.logger
    if relationship_type == "A_B_C":
        b_term = row.get("b_term", "")
        a_term = row.get("a_term", "")
        c_term = row.get("c_term", "")

        ab_abstracts = row.get("ab_abstracts", "")
        bc_abstracts = row.get("bc_abstracts", "")
    elif relationship_type == "A_C":
        a_term = row.get("a_term", "")
        c_term = row.get("c_term", "")
        b_term = ""  # Initialize b_term for A_C relationship

        ab_abstracts = ""  # Not used for A-C relationship
        bc_abstracts = ""  # Not used for A-C relationship
        ac_abstracts = row.get("ac_abstracts", "")
    elif relationship_type == "A_B":
        if config.is_dch:
            # DCH: direct comparison, use consolidated text from row if provided
            a_term = ""
            b_term = ""
            c_term = ""
            ab_abstracts = row.get("ab_abstracts", "")
            bc_abstracts = ""
            ac_abstracts = ""
        else:
            # Preserve pipes through relevance; strip only just before LLM call
            a_term = row.get("a_term", "")
            b_term = row.get("b_term", "")
            c_term = ""  # Not used for A-B relationship

            ab_abstracts = row.get("ab_abstracts", "")
            bc_abstracts = ""  # Not used
            ac_abstracts = ""  # Not used
    
    else:
        logger.error(f"Unknown relationship type: {relationship_type}")
        return [SCORE_NA_RESPONSE], "", {}

    # Compute expected per-abstract count based on available abstracts
    if relationship_type == "A_B_C":
        expected_count = (
            (len(ab_abstracts) if isinstance(ab_abstracts, list) else 0) +
            (len(bc_abstracts) if isinstance(bc_abstracts, list) else 0)
        )
    elif relationship_type == "A_C":
        expected_count = (len(ac_abstracts) if isinstance(ac_abstracts, list) else 0)
    elif relationship_type == "A_B":
        if config.is_dch:
            expected_count = row.get("expected_per_abstract_count")
        else:
            expected_count = (len(ab_abstracts) if isinstance(ab_abstracts, list) else 0)

    # Define URLs based on relationship type
    if relationship_type == "A_B_C":
        urls = {
            "AB": extract_pmids_and_generate_urls(ab_abstracts, config) if ab_abstracts else [],
            "BC": extract_pmids_and_generate_urls(bc_abstracts, config) if bc_abstracts else [],
            # Exclude AC URLs from ABC section
        }
    elif relationship_type == "A_C":
        urls = {
            "AC": extract_pmids_and_generate_urls(ac_abstracts, config) if ac_abstracts else [],
        }
    elif relationship_type == "A_B":
        urls = {
            "AB": extract_pmids_and_generate_urls(ab_abstracts, config) if ab_abstracts else [],
        }

    # Conditions for early exit based on abstracts availability
    if relationship_type == "A_B_C":
        if not ab_abstracts or not bc_abstracts:
            logger.error(
                f"Early exit for job_type '{config.job_type}' and relationship_type '{relationship_type}': Missing 'ab_abstracts' or 'bc_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
    elif relationship_type == "A_C":
        if not ac_abstracts:
            logger.error(
                f"Early exit for job_type '{config.job_type}' and relationship_type '{relationship_type}': Missing 'ac_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
    elif relationship_type == "A_B":
        if not config.is_dch and not ab_abstracts:
            logger.error(
                f"Early exit for relationship_type '{relationship_type}': Missing 'ab_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
    

    # Consolidate abstracts based on relationship type
    if relationship_type == "A_B_C":
        consolidated_abstracts = ab_abstracts + bc_abstracts
    elif relationship_type == "A_C":
        consolidated_abstracts = ac_abstracts
    elif relationship_type == "A_B":
        consolidated_abstracts = ab_abstracts
    

    try:
        logger.debug(f"Consolidated abstracts: {consolidated_abstracts}")
        logger.debug(f"Job type: {config.job_type}")
        logger.debug(f"Relationship type: {relationship_type}")
        logger.debug(f"A term: {a_term}")
        logger.debug(f"B term: {b_term}")
        
        logger.debug(f"C term: {c_term}")
        # Terms are already cleaned in process_single_row() before being stored in row
        result, prompt_text = analyze_abstract_with_frontier_LLM(
            a_term=a_term,
            b_term=b_term,
            c_term=c_term,
            consolidated_abstracts=consolidated_abstracts,
            config=config,
            relationship_type=relationship_type,
            hypothesis1=row.get("hypothesis1"),
            hypothesis2=row.get("hypothesis2"),
            expected_per_abstract_count=expected_count,
        )
        logger.debug(f" IN ANALYZE ABSTRACT   Result: {result}")
        logger.debug(f" IN ANALYZE ABSTRACT   Prompt text: {prompt_text}")
        logger.debug(f" IN ANALYZE ABSTRACT   B term: {b_term}")
        logger.debug(f" IN ANALYZE ABSTRACT   URLs: {urls}")
        return result, prompt_text, urls
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return [SCORE_NA_RESPONSE], "", urls


def analyze_abstract_with_frontier_LLM(
    a_term: str,
    b_term: str,
    c_term: str,
    consolidated_abstracts: str,
    config: Config,
    relationship_type: str,
    hypothesis1: str | None = None,
    hypothesis2: str | None = None,
    expected_per_abstract_count: int | None = None,
) -> tuple:
    logger = config.logger
    if not a_term and not config.is_dch:
        logger.error("A term is empty.")
        return [SCORE_NA_RESPONSE], ""

    # Use pre-initialized LLM client from config
    client = config.llm_client

    responses = []
    prompt_text = ""

    # Hypotheses are already cleaned (strip_pipe applied to terms before formatting in relevance.py)
    prompt_text = generate_prompt(
        a_term=a_term,
        b_term=b_term,
        c_term=c_term,
        content=consolidated_abstracts,
        config=config,
        relationship_type=relationship_type,
        hypothesis1=hypothesis1,
        hypothesis2=hypothesis2,
    )
    if not prompt_text:
        logger.error("Failed to generate prompt.")
        return ["Score: N/A"], prompt_text
    logger.debug(f" IN ANALYZE ABSTRACT   Prompt text: {prompt_text}")

    # Derive expected per-abstract count from the prompt; prefer this over any provided value
    final_expected_count = expected_per_abstract_count
    try:
        m = re.search(r"Available PMIDs for (?:[Cc]itation|[Cc]itation):\s*([0-9,\s]+)", prompt_text)
        if m:
            numbers = [n for n in re.findall(r"\d+", m.group(1))]
            if numbers:
                final_expected_count = len(numbers)
                logger.info(f"Derived expected_per_abstract_count from prompt: {final_expected_count}")
    except Exception as _e:
        logger.debug(f"Could not derive expected count from prompt: {_e}")

    response = call_openai_json(
        client,
        prompt_text,
        config,
        expected_per_abstract_count=final_expected_count,
    )
    logger.debug(f" IN ANALYZE ABSTRACT   Response: {response}")
    if response:
        responses.append(response)
    logger.debug(f" IN ANALYZE ABSTRACT   Responses: {responses}")
    return responses, prompt_text


def generate_prompt(
    a_term: str,
    b_term: str,
    c_term: str,
    content: str,
    config: Config,
    relationship_type: str,
    hypothesis1: str | None = None,
    hypothesis2: str | None = None,
) -> str:
    logger = config.logger
  
    if config.is_dch:
        # DCH mode: direct hypothesis comparison (is_dch implies is_km_with_gpt)
        if not (hypothesis1 and hypothesis2):
            logger.error("DCH requires hypothesis1 and hypothesis2.")
            return ""
        prompt_function = getattr(prompts_module, "km_with_gpt_direct_comp", None)
        if not prompt_function:
            logger.error("Prompt function for DCH not found.")
            return ""
        return prompt_function(
            hypothesis_1=hypothesis1,
            hypothesis_2=hypothesis2,
            consolidated_abstracts=content,
        )
    elif config.is_km_with_gpt:
        # Standard km_with_gpt (non-DCH)
        hypothesis_template = config.km_hypothesis.format(b_term=b_term, a_term=a_term)
    elif config.is_skim_with_gpt:
        if relationship_type == "A_B_C":
            hypothesis_template = config.skim_hypotheses["ABC"].format(
                a_term=a_term, b_term=b_term, c_term=c_term
            )
        elif relationship_type == "A_C":
            hypothesis_template = config.skim_hypotheses["AC"].format(
                a_term=a_term, c_term=c_term
            )
        elif relationship_type == "A_B":
            hypothesis_template = config.skim_hypotheses["AB"].format(
                a_term=a_term, b_term=b_term
            )
        else:
            logger.error(
                f"Unknown relationship type for skim_with_gpt: {relationship_type}"
            )
            return ""
    else:
        logger.error(f"Unknown job type: {config.job_type}")
        return ""

    # Select the appropriate prompt function based on relationship_type
    if config.is_skim_with_gpt and relationship_type == "A_C":
        prompt_function = getattr(prompts_module, "skim_with_gpt_ac", None)
    else:
        prompt_function = getattr(prompts_module, config.job_type, None)

    if not prompt_function:
        raise ValueError(
            f"Prompt function for relationship type '{relationship_type}' and job type '{config.job_type}' not found in the prompts module."
        )

    # Prepare arguments for the prompt function
    if relationship_type == "A_B_C":
        return prompt_function(
            a_term=a_term,
            b_term=b_term,
            c_term=c_term,
            hypothesis_template=hypothesis_template,
            consolidated_abstracts=content,
        )
    elif relationship_type == "A_C":
        return prompt_function(
            a_term=a_term,
            hypothesis_template=hypothesis_template,
            consolidated_abstracts=content,
            c_term=c_term,
        )
    elif relationship_type == "A_B":
        return prompt_function(
            a_term=a_term,
            b_term=b_term,
            hypothesis_template=hypothesis_template,
            consolidated_abstracts=content,
        )
    
    else:
        raise ValueError("Invalid relationship type specified.")


def call_openai_json(client, prompt, config, expected_per_abstract_count: int | None = None):
    # Build messages with system + user for all models
    if config.is_dch:
        system_instructions = prompts_module.km_with_gpt_direct_comp_system_instructions()
        irrelevant_label = "neither"
    elif config.is_km_with_gpt:
        system_instructions = prompts_module.km_with_gpt_system_instructions()
        irrelevant_label = "inconclusive"
    elif config.is_skim_with_gpt:
        system_instructions = prompts_module.skim_with_gpt_system_instructions()
        irrelevant_label = "inconclusive"
    else:
        raise ValueError(f"Unknown job type: {config.job_type}")

    messages = [
        {"role": "system", "content": system_instructions},
    ]
    if expected_per_abstract_count is not None:
        messages.append({
            "role": "system",
            "content": (
                "You MUST return exactly "
                f"{expected_per_abstract_count} items in the per_abstract array. "
                "Include one entry per PMID listed. If an abstract is irrelevant, "
                f"label it '{irrelevant_label}' but still include it. Do not omit any."
            )
        })
    messages.append({"role": "user", "content": prompt})

    params = {
        "messages": messages,
        "model": ("deepseek-reasoner" if config.model == "r1" else config.model),
    }

    # Retry and error handling (migrated from call_openai)
    retry_delay = config.retry_delay
    max_retries = config.max_retries

    for _ in range(1, max_retries + 1):
        try:
            # Time the API call
            api_start_time = time.time()
            resp = client.chat.completions.create(**params)
            api_duration = time.time() - api_start_time
            
            # Extract token usage and log timing
            total_tokens = resp.usage.total_tokens if hasattr(resp, 'usage') and resp.usage else 'N/A'
            model_name = params.get('model', 'unknown')
            abstracts_count = expected_per_abstract_count if expected_per_abstract_count else 'N/A'
            
            config.logger.info(
                f"OpenAI API call completed in {api_duration:.2f}s "
                f"(model={model_name}, abstracts={abstracts_count}, tokens={total_tokens})"
            )
            
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from API.")

            payload = extract_json_from_markdown(content)

            # Validate required fields
            for k in ["per_abstract","score_rationale","tallies","score","decision"]:
                if k not in payload:
                    raise ValueError(f"Missing required field '{k}' in model output.")

            # Enforce per_abstract length when provided
            if expected_per_abstract_count is not None:
                per_abstract = payload.get("per_abstract", [])
                if not isinstance(per_abstract, list) or len(per_abstract) != expected_per_abstract_count:
                    raise ValueError(
                        f"per_abstract length {len(per_abstract) if isinstance(per_abstract, list) else 'N/A'} "
                        f"does not match expected {expected_per_abstract_count}."
                    )

            # Validate tallies depending on mode
            tallies = payload.get("tallies", {})
            if config.is_dch:
                for tk in ["support_H1","support_H2","both","neither_or_inconclusive"]:
                    if tk not in tallies:
                        raise ValueError(f"Missing required tally '{tk}' in model output.")
            else:
                for tk in ["support","refute","inconclusive"]:
                    if tk not in tallies:
                        raise ValueError(f"Missing required tally '{tk}' in model output.")

            return payload

        except openai.AuthenticationError as e:
            config.logger.error(
                "AuthenticationError: Your API key or token was invalid, expired, or revoked."
            )
            config.logger.debug(str(e))
            break
        except openai.BadRequestError as e:
            config.logger.warning("BadRequestError: request malformed – retrying after delay …")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.PermissionDeniedError as e:
            config.logger.error("PermissionDeniedError: Access denied for the requested resource.")
            config.logger.debug(str(e))
            break
        except openai.NotFoundError as e:
            config.logger.error("NotFoundError: The requested resource does not exist.")
            config.logger.debug(str(e))
            break
        except openai.ConflictError as e:
            config.logger.error("ConflictError: Resource updated by another request; retrying …")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.UnprocessableEntityError as e:
            if config.model == "r1":
                config.logger.error("UnprocessableEntityError: Invalid parameters for DeepSeek API.")
            else:
                config.logger.error("UnprocessableEntityError: Unable to process the request.")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.RateLimitError as e:
            if config.model == "r1":
                config.logger.warning("Rate Limit Reached: Please slow down requests.")
            else:
                config.logger.warning("429 received; backing off …")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.APITimeoutError as e:
            config.logger.error("APITimeoutError: Request timed out; retrying …")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.APIConnectionError as e:
            config.logger.error("APIConnectionError: Issue connecting to OpenAI services.")
            config.logger.debug(str(e))
            time.sleep(retry_delay)
            continue
        except openai.APIStatusError as e:
            if config.model == "r1" and e.status_code == 500:
                config.logger.error("Server Error: DeepSeek server issue; retrying later …")
            else:
                config.logger.error(
                    f"APIStatusError: Non-200 status. Status Code: {e.status_code}, Response: {e.response}"
                )
            time.sleep(retry_delay)
            continue
        except openai.APIError as e:
            if config.model == "r1":
                status_code = getattr(e, 'status_code', None)
                if status_code == 402:
                    config.logger.error("Insufficient Balance: Please add funds.")
                    break
                elif status_code == 503:
                    config.logger.error("Server Overloaded: Retrying after delay …")
                    time.sleep(retry_delay)
                    continue
            raise
        except Exception as e:
            config.logger.error("Unexpected error during API call.")
            config.logger.info(str(e))
            time.sleep(retry_delay)
            continue

    return None
