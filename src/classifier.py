import openai
import time
import json
import re
from typing import Any
from src.utils import Config
from src import prompt_library as prompts_module 



def write_to_json(data, file_path, config: Config):
    logger = config.logger
    # Sanitize file_path by replacing ',', '[', ']', and ' ' with '_'
    file_path = file_path.replace(",", "_").replace("[", "_").replace("]", "_").replace(" ", "_").replace("'", "_")
    logger.debug(f" IN WRITE TO JSON   File path: {file_path}")
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def calculate_relevance_ratios(out_df, config: Config):
    logger = config.logger
    logger.debug(f" IN CALCULATE RELEVANCE RATIOS   Out df: {out_df}")
    mask_columns = ["ab_mask", "bc_mask", "ac_mask"]
    logger.debug(f" IN CALCULATE RELEVANCE RATIOS   Mask columns: {mask_columns}")
    logger.debug(f" IN CALCULATE RELEVANCE RATIOS   Out df columns: {out_df.columns}")
    for col in mask_columns:
        if col in out_df.columns:
            prefix = col.split("_")[0]
            ratio_col = f"{prefix}_relevance_ratio"
            fraction_col = f"{prefix}_relevance_fraction"
            logger.debug(f" IN CALCULATE RELEVANCE RATIOS   Ratio col: {ratio_col}")
            logger.debug(f" IN CALCULATE RELEVANCE RATIOS   Fraction col: {fraction_col}")
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

        abc_result, abc_prompt, abc_urls = perform_analysis(
            job_type=config.job_type, row=row, config=config, relationship_type="A_B_C"
        )
        if abc_result or abc_prompt:
            processed_results["A_B_C_Relationship"] = {
                "Relationship": f"{a_term} - {b_term} - {c_term}",
                "Result": abc_result,
                "Prompt": abc_prompt,
                "URLS": {
                    "AB": abc_urls.get("AB", []),
                    "BC": abc_urls.get("BC", []),
                },
            }

        ac_result, ac_prompt, ac_urls = perform_analysis(
            job_type=config.job_type, row=row, config=config, relationship_type="A_C"
        )
        if ac_result or ac_prompt:
            processed_results["A_C_Relationship"] = {
                "Relationship": f"{a_term} - {c_term}",
                "Result": ac_result,
                "Prompt": ac_prompt,
                "URLS": {
                    "AC": ac_urls.get("AC", []),
                },
            }

    elif config.job_type == "km_with_gpt":
        # Handle km_with_gpt job type
        a_term = row.get("a_term")
        b_term = row.get("b_term")

        if not all([a_term, b_term]):
            logger.error(
                f"Missing 'a_term' or 'b_term' in row {row.get('id', 'Unknown')}."
            )
            return None

        # Process A-B relationship
        ab_result, ab_prompt, ab_urls = perform_analysis(
            job_type=config.job_type, row=row, config=config, relationship_type="A_B"
        )
        if ab_result or ab_prompt:
            processed_results["A_B_Relationship"] = {
                "Relationship": f"{a_term} - {b_term}",
                "Result": ab_result,
                "Prompt": ab_prompt,
                "URLS": {
                    "AB": ab_urls.get("AB", []),
                },
            }
    elif config.is_km_with_gpt_direct_comp:
         # Handle km_with_gpt_direct_comp job type
        logger.debug(f"Processing row: {row}")
        logger.debug(f"Row type: {type(row)}")
        logger.debug(f"Row keys: {row.keys()}")

        a_term = row.get("a_term")
        b_term = row.get("b_term")
        logger.debug(f" IN PROCESS SINGLE ROW   B term: {b_term}")
         # Remove brackets and split by comma
        b_term_str = b_term.strip("[]")  # Remove brackets
        b_term_list = [item.strip() for item in b_term_str.split(',')] # Split by comma and strip whitespace
        # Filter out any empty strings that might result from the split
        b_term = [item for item in b_term_list if item]
        assert len(b_term) == 2
        b_term1 = b_term[0]
        b_term2 = b_term[1] if len(b_term) > 1 else ""
        logger.debug(f" IN PROCESS SINGLE ROW   B term1: {b_term1}")
        logger.debug(f" IN PROCESS SINGLE ROW   B term2: {b_term2}")
        ab_result, ab_prompt, ab_urls = perform_analysis(
            job_type=config.job_type, row=row, config=config, relationship_type="A_B1_B2"
        )
        logger.debug(f" IN PROCESS SINGLE ROW   A_B1_B2_Relationship: {ab_result}")
        # config.logger.debug(f" IN PROCESS SINGLE ROW   A_B1_B2_Prompt: {ab_prompt}")
        # config.logger.debug(f" IN PROCESS SINGLE ROW   A_B1_B2_URLS: {ab_urls}")
        if ab_result or ab_prompt:
            processed_results["A_B1_B2_Relationship"] = {
                "Relationship": f"{a_term} - {b_term1} - {b_term2}",
                "Result": ab_result,
                "Prompt": ab_prompt,
                "URLS": {
                    "AB1B2": ab_urls.get("AB1B2", []),
                },
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

    # Regular expression to find PMIDs
    pmid_pattern = r"PMID:\s*(\d+)"

    # Find all PMIDs in the text
    pmids = re.findall(pmid_pattern, text)

    # Generate PubMed URLs
    pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]

    return pubmed_urls


def perform_analysis(job_type: str, row: dict, config: Config, relationship_type: str) -> tuple:
    logger = config.logger
    if relationship_type == "A_B_C":
        b_term = row.get("b_term", "")
        a_term = row.get("a_term", "")
        c_term = row.get("c_term", "")

        ab_abstracts = row.get("ab_pmid_intersection", "")
        bc_abstracts = row.get("bc_pmid_intersection", "")
    elif relationship_type == "A_C":
        a_term = row.get("a_term", "")
        c_term = row.get("c_term", "")

        ab_abstracts = ""  # Not used for A-C relationship
        bc_abstracts = ""  # Not used for A-C relationship
        ac_abstracts = row.get("ac_pmid_intersection", "")
    elif relationship_type == "A_B":
        a_term = row.get("a_term", "")
        b_term = row.get("b_term", "")
        c_term = ""  # Not used for A-B relationship

        ab_abstracts = row.get("ab_pmid_intersection", "")
        bc_abstracts = ""  # Not used
        ac_abstracts = ""  # Not used
    elif relationship_type == "A_B1_B2":
        a_term = row.get("a_term", "")
        b_term = row.get("b_term", "")
        b_term1 = b_term[0]
        b_term2 = b_term[1]
        ab_abstracts = row.get("ab_pmid_intersection", "")
        bc_abstracts = ""  # Not used
        ac_abstracts = ""  # Not used
        c_term = ""  # Not used for A-B1-B2 relationship
    else:
        logger.error(f"Unknown relationship type: {relationship_type}")
        return ["Score: N/A"], "", {}

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
    elif relationship_type == "A_B1_B2":
        urls = {
            "AB1B2": extract_pmids_and_generate_urls(ab_abstracts, config) if ab_abstracts else [],
        }

    # Conditions for early exit based on abstracts availability
    if relationship_type == "A_B_C":
        if not ab_abstracts or not bc_abstracts:
            logger.error(
                f"Early exit for job_type '{job_type}' and relationship_type '{relationship_type}': Missing 'ab_abstracts' or 'bc_abstracts'."
            )
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_C":
        if not ac_abstracts:
            logger.error(
                f"Early exit for job_type '{job_type}' and relationship_type '{relationship_type}': Missing 'ac_abstracts'."
            )
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_B" or relationship_type == "A_B1_B2":
        if not ab_abstracts:
            logger.error(
                f"Early exit for relationship_type '{relationship_type}': Missing 'ab_abstracts'."
            )
            return ["Score: N/A"], "", urls

    # Consolidate abstracts based on relationship type
    if relationship_type == "A_B_C":
        consolidated_abstracts = ab_abstracts + bc_abstracts
    elif relationship_type == "A_C":
        consolidated_abstracts = ac_abstracts
    elif relationship_type == "A_B" or relationship_type == "A_B1_B2":
        consolidated_abstracts = ab_abstracts

    try:
        logger.debug(f"Consolidated abstracts: {consolidated_abstracts}")
        logger.debug(f"Job type: {job_type}")
        logger.debug(f"Relationship type: {relationship_type}")
        logger.debug(f"A term: {a_term}")
        logger.debug(f"B term: {b_term}")
        if relationship_type == "A_B1_B2":
            logger.debug(f"B term1: {b_term1}")
            logger.debug(f"B term2: {b_term2}")
        logger.debug(f"C term: {c_term}")
        result, prompt_text = analyze_abstract_with_frontier_LLM(
            a_term=a_term,
            b_term=row.get("b_term", ""),
            c_term=c_term,
            consolidated_abstracts=consolidated_abstracts,
            job_type=job_type,
            config=config,
            relationship_type=relationship_type,
        )
        logger.debug(f" IN ANALYZE ABSTRACT   Result: {result}")
        logger.debug(f" IN ANALYZE ABSTRACT   Prompt text: {prompt_text}")
        logger.debug(f" IN ANALYZE ABSTRACT   URLs: {urls}")
        return result, prompt_text, urls
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return ["Score: N/A"], "", urls


def analyze_abstract_with_frontier_LLM(
    a_term: str,
    b_term: str,
    c_term: str,
    consolidated_abstracts: str,
    job_type: str,
    config: Config,
    relationship_type: str,
) -> tuple:
    logger = config.logger
    if not a_term:
        logger.error("A term is empty.")
        return [], ""

    api_key = config.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OpenAI API key is not set.")

    openai_client = openai.OpenAI(api_key=api_key)

    responses = []
    prompt_text = ""

    # Generate prompt based on relationship type and job type
    prompt_text = generate_prompt(
        job_type=job_type,
        a_term=a_term,
        b_term=b_term,
        c_term=c_term,
        content=consolidated_abstracts,
        config=config,
        relationship_type=relationship_type,
    )
    if not prompt_text:
        logger.error("Failed to generate prompt.")
        return ["Score: N/A"], prompt_text
    logger.debug(f" IN ANALYZE ABSTRACT   Prompt text: {prompt_text}")
    response = call_openai(openai_client, prompt_text, config)
    if response:
        responses.append(response)
    logger.debug(f" IN ANALYZE ABSTRACT   Responses: {responses}")
    return responses, prompt_text


def generate_prompt(
    job_type: str,
    a_term: str,
    b_term: str,
    c_term: str,
    content: str,
    config: Config,
    relationship_type: str,
) -> str:
    logger = config.logger
    job_type_lower = job_type.lower()   
  
    if job_type == "km_with_gpt_direct_comp":
        # Remove brackets and split by comma
        b_term_str = b_term.strip("[]")  # Remove brackets
        b_term_list = [item.strip() for item in b_term_str.split(',')] # Split by comma and strip whitespace
        # Filter out any empty strings that might result from the split
        b_term = [item for item in b_term_list if item]
        assert len(b_term) == 2
        b_term1 = b_term[0]
        b_term2 = b_term[1]
        logger.debug(f" IN GENERATE PROMPT   B term: {b_term}")
        logger.debug(f" IN GENERATE PROMPT B term1: {b_term1}")
        logger.debug(f" IN GENERATE PROMPT B term2: {b_term2}")
    # Determine the correct hypothesis template from config
    if job_type == "km_with_gpt":
        hypothesis_template = config.km_hypothesis.format(b_term=b_term, a_term=a_term)
    elif job_type == "km_with_gpt_direct_comp":
        hypothesis_template = config.km_direct_comp_hypothesis.format(b_term1=b_term1, b_term2=b_term2, a_term=a_term)
        logger.debug(f" IN GENERATE PROMPT   Hypothesis template: {hypothesis_template}")
    elif job_type == "skim_with_gpt":
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
        logger.error(f"Unknown job type: {job_type}")
        return ""

    # Select the appropriate prompt function based on relationship_type
    if job_type == "skim_with_gpt" and relationship_type == "A_C":
        prompt_function = getattr(prompts_module, "skim_with_gpt_ac", None)
    else:
        prompt_function = getattr(prompts_module, job_type_lower, None)

    if not prompt_function:
        raise ValueError(
            f"Prompt function for relationship type '{relationship_type}' and job type '{job_type}' not found in the prompts module."
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
    elif relationship_type == "A_B1_B2":
        return prompt_function(
            a_term=a_term,
            b_term1=b_term1,
            b_term2=b_term2,
            hypothesis_template=hypothesis_template,
            consolidated_abstracts=content,
        )
    else:
        raise ValueError("Invalid relationship type specified.")


def call_openai(client, prompt, config: Config):
    logger = config.logger
    retry_delay = config.global_settings["RETRY_DELAY"]
    max_retries = config.global_settings["MAX_RETRIES"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                logger.warning("Empty response received from OpenAI API.")
                time.sleep(retry_delay)

        # Specific Exceptions First
        except openai.AuthenticationError as e:
            logger.error(
                "AuthenticationError: Your API key or token was invalid, expired, or revoked.\n"
                "Solution: Check your API key or token and make sure it is correct and active. "
                "You may need to generate a new one from your account dashboard."
            )
            logger.debug(str(e))
            break  # Authentication issues won't resolve with retries

        except openai.BadRequestError as e:
            logger.error(
                "BadRequestError: Your request was malformed or missing some required parameters.\n"
                "Solution: Check the error message for specifics, ensure all required parameters "
                "are provided, and verify the format and size of your request data."
            )
            logger.debug(str(e))
            break  # Bad requests won't resolve with retries

        except openai.PermissionDeniedError as e:
            logger.error(
                "PermissionDeniedError: You don't have access to the requested resource.\n"
                "Solution: Ensure you are using the correct API key, organization ID, and resource ID."
            )
            logger.debug(str(e))
            break  # Permission issues won't resolve with retries

        except openai.NotFoundError as e:
            logger.error(
                "NotFoundError: The requested resource does not exist.\n"
                "Solution: Ensure you are using the correct resource identifier."
            )
            logger.debug(str(e))
            break  # Not found errors won't resolve with retries

        except openai.ConflictError as e:
            logger.error(
                "ConflictError: The resource was updated by another request.\n"
                "Solution: Try to update the resource again and ensure no other requests "
                "are attempting to update it simultaneously."
            )
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.UnprocessableEntityError as e:
            logger.error(
                "UnprocessableEntityError: Unable to process the request despite the format being correct.\n"
                "Solution: Please try the request again."
            )
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.RateLimitError as e:
            logger.warning("A 429 status code was received; we should back off a bit.")
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.APITimeoutError as e:
            logger.error(
                "APITimeoutError: The request timed out.\n"
                "Solution: Retry your request after a brief wait and contact OpenAI if the issue persists."
            )
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.APIConnectionError as e:
            logger.error(
                "APIConnectionError: Issue connecting to OpenAI services.\n"
                "Solution: Check your network settings, proxy configuration, SSL certificates, or firewall rules."
            )
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.APIStatusError as e:
            # General APIStatusError should come after specific APIStatusError subclasses
            logger.error(
                f"APIStatusError: Another non-200-range status code was received.\n"
                f"Status Code: {e.status_code}\n"
                f"Response: {e.response}\n"
                f"Cause: {e.__cause__}"
            )
            time.sleep(retry_delay)

        except Exception as e:
            logger.error("An unexpected error occurred.")
            logger.debug(str(e))
            time.sleep(retry_delay)

    return None
