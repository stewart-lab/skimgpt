import openai
import time
import json
import re
from typing import Any
from src.utils import Config, strip_pipe
from src import prompt_library as prompts_module  
import os


def write_to_json(data, file_path, output_directory, config: Config):
    logger = config.logger
    # Sanitize file_path by replacing ',', '[', ']', and ' ' with '_'
    file_path = file_path.replace(",", "_").replace("[", "_").replace("]", "_").replace(" ", "_").replace("'", "_")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    file_path = os.path.join(output_directory, file_path)
    logger.debug(f" IN WRITE TO JSON   File path: {file_path}")
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def calculate_relevance_ratios(out_df, config: Config):
    logger = config.logger
    logger.debug(f" IN CALCULATE RELEVANCE RATIOS  Complete Out df: {out_df.to_string()}")
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
            row=row, config=config, relationship_type="A_B_C"
        )
        if abc_result or abc_prompt:
            processed_results["A_B_C_Relationship"] = {
                "a_term": a_term,
                "b_term": b_term,
                "c_term": c_term,
                "Relationship": f"{a_term} - {b_term} - {c_term}",
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
                "Result": ac_result,
                "Prompt": ac_prompt,
                "URLS": {
                    "AC": ac_urls.get("AC", []),
                },
            }

    elif config.is_km_with_gpt and not config.is_dch:
        a_term = row.get("a_term")
        b_term = row.get("b_term")

        if config.is_dch and isinstance(b_term, list):
            # DCH case where b_term is a list of two; defer branching to prompt-building using config.is_dch
            relationship_type = "A_B"
            result, prompt, urls = perform_analysis(
                job_type=config.job_type, row=row, config=config, relationship_type=relationship_type
            )

            if result or prompt:
                processed_results = json.loads(result)

        else:
            # Standard km_with_gpt case
            if not all([a_term, b_term]):
                logger.error(f"Missing 'a_term' or 'b_term' in row.")
                return None

        ab_result, ab_prompt, ab_urls = perform_analysis(
            row=row, config=config, relationship_type="A_B"
        )
        if ab_result or ab_prompt:
            processed_results["A_B_Relationship"] = {
                "a_term": a_term,
                "b_term": strip_pipe(b_term),
                "Relationship": f"{strip_pipe(a_term)} - {strip_pipe(b_term)}",
                "Result": ab_result,
                "Prompt": ab_prompt,
                "URLS": {"AB": ab_urls.get("AB", [])},
            }
    elif config.is_km_with_gpt and config.is_dch:
        hypothesis1 = row.get("hypothesis1")
        hypothesis2 = row.get("hypothesis2")
        hypothesis1_pmids = row.get("hypothesis1_pmids")
        hypothesis2_pmids = row.get("hypothesis2_pmids")
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


def perform_analysis(row: dict, config: Config, relationship_type: str) -> tuple:
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
        b_term = ""  # Initialize b_term for A_C relationship

        ab_abstracts = ""  # Not used for A-C relationship
        bc_abstracts = ""  # Not used for A-C relationship
        ac_abstracts = row.get("ac_pmid_intersection", "")
    elif relationship_type == "A_B":
        if config.is_dch:
            # DCH: direct comparison, use consolidated text from row if provided
            a_term = ""
            b_term = ""
            c_term = ""
            ab_abstracts = row.get("ab_pmid_intersection", "")
            bc_abstracts = ""
            ac_abstracts = ""
        else:
            a_term = row.get("a_term", "")
            b_term = row.get("b_term", "")
            c_term = ""  # Not used for A-B relationship

            ab_abstracts = row.get("ab_pmid_intersection", "")
            bc_abstracts = ""  # Not used
            ac_abstracts = ""  # Not used
    
    else:
        logger.error(f"Unknown relationship type: {relationship_type}")
        return ["Score: N/A"], "", {}

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
    else:
        expected_count = None

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
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_C":
        if not ac_abstracts:
            logger.error(
                f"Early exit for job_type '{config.job_type}' and relationship_type '{relationship_type}': Missing 'ac_abstracts'."
            )
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_B":
        if not config.is_dch and not ab_abstracts:
            logger.error(
                f"Early exit for relationship_type '{relationship_type}': Missing 'ab_abstracts'."
            )
            return ["Score: N/A"], "", urls
    

    # Consolidate abstracts based on relationship type
    if relationship_type == "A_B_C":
        consolidated_abstracts = ab_abstracts + bc_abstracts
    elif relationship_type == "A_C":
        consolidated_abstracts = ac_abstracts
    elif relationship_type == "A_B":
        consolidated_abstracts = ab_abstracts
    logger.info(f" IN PERFORM ANALYSIS   Consolidated abstracts: {consolidated_abstracts}")
    

    try:
        logger.debug(f"Consolidated abstracts: {consolidated_abstracts}")
        logger.debug(f"Job type: {config.job_type}")
        logger.debug(f"Relationship type: {relationship_type}")
        logger.debug(f"A term: {a_term}")
        logger.debug(f"B term: {b_term}")
        
        logger.debug(f"C term: {c_term}")
        b_term = row.get("b_term", "")
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
        return ["Score: N/A"], "", urls


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
        return [], ""

    # Initialize client based on model type
    if config.model == "r1":
        api_key = config.secrets["DEEPSEEK_API_KEY"]
        if not api_key:
            raise ValueError("Deepseek API key is not set.")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    else:
        api_key = config.secrets["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OpenAI API key is not set.")
        client = openai.OpenAI(api_key=api_key)

    responses = []
    prompt_text = ""

    # Generate prompt based on relationship type and current config.job_type
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

    # Derive expected per-abstract count from the prompt when in DCH; prefer this over any provided value
    final_expected_count = expected_per_abstract_count
    if config.is_dch:
        try:
            import re as _re
            m = _re.search(r"Available PMIDs for citation:\s*([0-9,\s]+)", prompt_text)
            if m:
                numbers = [n for n in _re.findall(r"\d+", m.group(1))]
                if numbers:
                    final_expected_count = len(numbers)
                    logger.info(f"Derived expected_per_abstract_count from prompt: {final_expected_count}")
        except Exception as _e:
            logger.debug(f"Could not derive expected count from prompt: {_e}")

    if config.is_dch:
        response = call_openai_json(
            client,
            prompt_text,
            config,
            expected_per_abstract_count=final_expected_count,
        )
        logger.info(f" IN ANALYZE ABSTRACT   Response: {response}")
    else:
        response = call_openai(client, prompt_text, config)
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
  
    if config.is_km_with_gpt:
        # Handle DCH here based on global config flag; relationship_type is not used to branch
        if config.is_dch:
            # Require direct hypotheses for DCH
            if not (hypothesis1 and hypothesis2):
                logger.error("DCH requires hypothesis1 and hypothesis2.")
                return ""
            h1 = hypothesis1
            h2 = hypothesis2
            prompt_function = getattr(prompts_module, "km_with_gpt_direct_comp", None)
            if not prompt_function:
                logger.error("Prompt function for DCH not found.")
                return ""
            return prompt_function(
                hypothesis_1=h1,
                hypothesis_2=h2,
                consolidated_abstracts=content,
            )
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


def extract_json_from_markdown(s: str) -> dict:
    # Finds ```json ... ```; falls back to first {...}
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if not m:
        m = re.search(r"(\{.*\})", s, flags=re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(1))

def call_openai_json(client, prompt, config, expected_per_abstract_count: int | None = None):
    # Build messages with system + user for all models (o1, o3, gpt-5 compatible)
    messages = [
        {"role": "system", "content": prompts_module.km_with_gpt_direct_comp_system_instructions()},
    ]
    if expected_per_abstract_count is not None:
        messages.append({
            "role": "system",
            "content": (
                "You MUST return exactly "
                f"{expected_per_abstract_count} items in the per_abstract array. "
                "Include one entry per PMID listed. If an abstract is irrelevant, "
                "label it 'neither' but still include it. Do not omit any."
            )
        })
    messages.append({"role": "user", "content": prompt})

    params = {
        "messages": messages,
        "model": ("deepseek-reasoner" if config.model == "r1" else config.model),
    }

    resp = client.chat.completions.create(**params)
    content = resp.choices[0].message.content
    # Parse strict JSON block
    payload = extract_json_from_markdown(content)

    # (Optional) light validation against your schema fields
    for k in ["per_abstract","score_rationale","tallies","score","decision","used_pmids"]:
        if k not in payload:
            raise ValueError(f"Missing required field '{k}' in model output.")

    # Enforce the number of per_abstract labels to match expected count when provided
    if expected_per_abstract_count is not None:
        per_abstract = payload.get("per_abstract", [])
        if not isinstance(per_abstract, list) or len(per_abstract) != expected_per_abstract_count:
            raise ValueError(
                f"per_abstract length {len(per_abstract) if isinstance(per_abstract, list) else 'N/A'} "
                f"does not match expected {expected_per_abstract_count}."
            )

    # Validate tallies include 'both' key
    tallies = payload.get("tallies", {})
    for tk in ["support_H1","support_H2","both","neither_or_inconclusive"]:
        if tk not in tallies:
            raise ValueError(f"Missing required tally '{tk}' in model output.")

    return payload

def call_openai(client, prompt, config: Config):
    logger = config.logger
    retry_delay = config.global_settings["RETRY_DELAY"]
    max_retries = config.global_settings["MAX_RETRIES"]

    for attempt in range(1, max_retries + 1):
        try:
            # Create parameters dictionary
            params = {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Set the correct model name based on config.model
            if config.model == "r1":
                # Use the proper DeepSeek model name instead of "r1"
                params["model"] = "deepseek-reasoner"
            else:
                params["model"] = config.model
            
            # Make the API call with the parameters
            response = client.chat.completions.create(**params)

            content = response.choices[0].message.content
            usage = response.usage
            logger.info(
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}, "
                f"total_tokens={usage.total_tokens}"
            )

            if content:
                return content
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
            logger.warning("BadRequestError: request malformed – retrying after delay …")
            logger.debug(str(e))
            time.sleep(retry_delay)
            continue  # simple retry without altering prompt

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

        # Catch conflict before generic APIError
        except openai.ConflictError as e:
            logger.error(
                "ConflictError: The resource was updated by another request.\n"
                "Solution: Try to update the resource again and ensure no other requests "
                "are attempting to update it simultaneously."
            )
            logger.debug(str(e))
            time.sleep(retry_delay)

        # The following must also come before openai.APIError or they'll remain unreachable
        except openai.UnprocessableEntityError as e:
            # This handles both OpenAI and Deepseek 422 errors
            if config.model == "r1":
                logger.error(
                    "UnprocessableEntityError: Your request contains invalid parameters.\n"
                    "Solution: Please modify your request parameters according to the DeepSeek API format."
                )
            else:
                logger.error(
                    "UnprocessableEntityError: Unable to process the request despite the format being correct.\n"
                    "Solution: Please try the request again."
                )
            logger.debug(str(e))
            time.sleep(retry_delay)

        except openai.RateLimitError as e:
            # This handles both OpenAI and Deepseek 429 errors
            if config.model == "r1":
                logger.warning(
                    "Rate Limit Reached: You are sending requests too quickly.\n"
                    "Solution: Please pace your requests reasonably. Consider switching to alternative providers temporarily."
                )
            else:
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
            # Handle both OpenAI and Deepseek general status errors
            if config.model == "r1" and e.status_code == 500:
                logger.error(
                    "Server Error: The DeepSeek server encountered an issue.\n"
                    "Solution: Please retry your request after a brief wait and contact support if the issue persists."
                )
            else:
                logger.error(
                    f"APIStatusError: Another non-200-range status code was received.\n"
                    f"Status Code: {e.status_code}\n"
                    f"Response: {e.response}\n"
                    f"Cause: {e.__cause__}"
                )
            time.sleep(retry_delay)

        # Finally, catch other API errors not covered by the above
        except openai.APIError as e:
            if config.model == "r1":
                status_code = getattr(e, 'status_code', None)
                if status_code == 402:
                    logger.error(
                        "Insufficient Balance: You have run out of balance.\n"
                        "Solution: Please check your account's balance and add funds if necessary."
                    )
                    break  # Balance issues won't resolve with retries
                elif status_code == 503:
                    logger.error(
                        "Server Overloaded: The server is experiencing high traffic.\n"
                        "Solution: Please retry your request after a brief wait."
                    )
                    time.sleep(retry_delay)
                    continue  # Continue to next retry attempt
            raise  # Re-raise so it's handled by outer layers if needed

        except Exception as e:
            logger.error("An unexpected error occurred.")
            logger.info(str(e))
            time.sleep(retry_delay)

    return None
