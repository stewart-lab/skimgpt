# /w5home/jfreeman/kmGPT/src/classifier.py
import os
import openai
import time
import json
import importlib
import inspect
import re
import logging
from typing import Tuple, List, Dict
from utils import Config  # **Import Config**
from Bio import Entrez

# Configure logging at the beginning of your script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("classifier.log"), logging.StreamHandler()],
)


def write_to_json(data, file_path):
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def calculate_relevance_ratios(out_df):
    mask_columns = ["ab_mask", "bc_mask", "ac_mask"]

    for col in mask_columns:
        if col in out_df.columns:
            prefix = col.split("_")[0]
            ratio_col = f"{prefix}_relevance_ratio"
            fraction_col = f"{prefix}_relevance_fraction"

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


def process_single_row(row, config):
    job_type = config.job_type

    if job_type not in [
        "km_with_gpt",
        "position_km_with_gpt",
        "skim_with_gpt",
    ]:
        logging.error("Invalid job type (caught in process_single_row)")
        return None

    # Initialize the result dictionary
    processed_results = {}

    if job_type == "skim_with_gpt":
        # Ensure all three terms are present
        a_term = row.get("a_term")
        b_term = row.get("b_term")
        c_term = row.get("c_term")
        if not all([a_term, b_term, c_term]):
            logging.error(
                f"Missing one or more terms in row {row.get('id', 'Unknown')}."
            )
            return None

        # Process A-B-C relationship
        abc_result, abc_prompt, abc_urls = perform_analysis(
            job_type=job_type, row=row, config=config, relationship_type="A_B_C"
        )
        if abc_result or abc_prompt:
            processed_results["A_B_C_Relationship"] = {
                "Relationship": f"{a_term} - {b_term} - {c_term}",
                "Result": abc_result,
                "Prompt": abc_prompt,
                "URLS": {
                    "AB": abc_urls.get("AB", []),
                    "BC": abc_urls.get("BC", []),
                    # Exclude AC URLs from ABC section
                },
            }

        # Process A-C relationship using 'skim_with_gpt_ac' prompt
        ac_result, ac_prompt, ac_urls = perform_analysis(
            job_type=job_type, row=row, config=config, relationship_type="A_C"
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

    elif job_type == "km_with_gpt":
        # Handle km_with_gpt job type
        a_term = row.get("a_term")
        b_term = row.get("b_term")

        if not all([a_term, b_term]):
            logging.error(
                f"Missing 'a_term' or 'b_term' in row {row.get('id', 'Unknown')}."
            )
            return None

        # Process A-B relationship
        ab_result, ab_prompt, ab_urls = perform_analysis(
            job_type=job_type, row=row, config=config, relationship_type="A_B"
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

    else:
        # Existing logic for other job types can be added here
        logging.warning(f"Job type '{job_type}' is not specifically handled.")

    return processed_results if any(processed_results.values()) else None


def extract_pmids_and_generate_urls(text: str) -> list:
    # Regular expression to find PMIDs
    pmid_pattern = r"PMID:\s*(\d+)"

    # Find all PMIDs in the text
    pmids = re.findall(pmid_pattern, text)

    # Generate PubMed URLs
    pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]

    return pubmed_urls


def perform_analysis(job_type: str, row: dict, config, relationship_type: str) -> tuple:
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
    else:
        logging.error(f"Unknown relationship type: {relationship_type}")
        return ["Score: N/A"], "", {}

    # Define URLs based on relationship type
    if relationship_type == "A_B_C":
        urls = {
            "AB": extract_pmids_and_generate_urls(ab_abstracts) if ab_abstracts else [],
            "BC": extract_pmids_and_generate_urls(bc_abstracts) if bc_abstracts else [],
            # Exclude AC URLs from ABC section
        }
    elif relationship_type == "A_C":
        urls = {
            "AC": extract_pmids_and_generate_urls(ac_abstracts) if ac_abstracts else [],
        }
    elif relationship_type == "A_B":
        urls = {
            "AB": extract_pmids_and_generate_urls(ab_abstracts) if ab_abstracts else [],
        }

    # Conditions for early exit based on abstracts availability
    if relationship_type == "A_B_C":
        if not ab_abstracts or not bc_abstracts:
            logging.info(
                f"Early exit for job_type '{job_type}' and relationship_type '{relationship_type}': Missing 'ab_abstracts' or 'bc_abstracts'."
            )
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_C":
        if not ac_abstracts:
            logging.info(
                f"Early exit for job_type '{job_type}' and relationship_type '{relationship_type}': Missing 'ac_abstracts'."
            )
            return ["Score: N/A"], "", urls
    elif relationship_type == "A_B":
        if not ab_abstracts:
            logging.info(
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

    try:
        result, prompt_text = analyze_abstract_with_gpt4(
            a_term=a_term,
            b_term=row.get("b_term", ""),
            c_term=c_term,
            consolidated_abstracts=consolidated_abstracts,
            job_type=job_type,
            config=config,
            relationship_type=relationship_type,
        )
        return result, prompt_text, urls
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return ["Score: N/A"], "", urls


def analyze_abstract_with_gpt4(
    a_term: str,
    b_term: str,
    c_term: str,
    consolidated_abstracts: str,
    job_type: str,
    config,
    relationship_type: str,
) -> tuple:
    if not a_term:
        logging.error("A term is empty.")
        return [], ""

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
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
        logging.error("Failed to generate prompt.")
        return ["Score: N/A"], prompt_text

    response = call_openai(openai_client, prompt_text, config)
    if response:
        responses.append(response)

    return responses, prompt_text


def generate_prompt(
    job_type: str,
    a_term: str,
    b_term: str,
    c_term: str,
    content: str,
    config,
    relationship_type: str,
) -> str:
    job_type_lower = job_type.lower()

    # Dynamically import the prompts module (assuming this module contains relevant prompt generation functions)
    prompts_module = importlib.import_module("prompt_library")
    assert prompts_module, "Failed to import the prompts module."

    # Determine the correct hypothesis template from config
    if job_type == "km_with_gpt":
        hypothesis_template = config.km_hypothesis.format(b_term=b_term, a_term=a_term)
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
            logging.error(
                f"Unknown relationship type for skim_with_gpt: {relationship_type}"
            )
            return ""
    else:
        logging.error(f"Unknown job type: {job_type}")
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
    else:
        raise ValueError("Invalid relationship type specified.")


def call_openai(client, prompt, config):
    retry_delay = config.global_settings["RETRY_DELAY"]
    max_retries = config.global_settings["MAX_RETRIES"]
    model = config.global_settings["MODEL"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                print("Empty response received from OpenAI API.")
                time.sleep(retry_delay)

        # Specific Exceptions First

        except openai.AuthenticationError as e:
            print(
                "AuthenticationError: Your API key or token was invalid, expired, or revoked."
            )
            print(
                "Solution: Check your API key or token and make sure it is correct and active. You may need to generate a new one from your account dashboard."
            )
            print(e)
            break  # Authentication issues won't resolve with retries

        except openai.BadRequestError as e:
            print(
                "BadRequestError: Your request was malformed or missing some required parameters."
            )
            print(
                "Solution: Check the error message for specifics, ensure all required parameters are provided, and verify the format and size of your request data."
            )
            print(e)
            break  # Bad requests won't resolve with retries

        except openai.PermissionDeniedError as e:
            print(
                "PermissionDeniedError: You don't have access to the requested resource."
            )
            print(
                "Solution: Ensure you are using the correct API key, organization ID, and resource ID."
            )
            print(e)
            break  # Permission issues won't resolve with retries

        except openai.NotFoundError as e:
            print("NotFoundError: The requested resource does not exist.")
            print("Solution: Ensure you are using the correct resource identifier.")
            print(e)
            break  # Not found errors won't resolve with retries

        except openai.ConflictError as e:
            print("ConflictError: The resource was updated by another request.")
            print(
                "Solution: Try to update the resource again and ensure no other requests are attempting to update it simultaneously."
            )
            print(e)
            time.sleep(retry_delay)

        except openai.UnprocessableEntityError as e:
            print(
                "UnprocessableEntityError: Unable to process the request despite the format being correct."
            )
            print("Solution: Please try the request again.")
            print(e)
            time.sleep(retry_delay)

        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            print(e)
            time.sleep(retry_delay)

        except openai.APITimeoutError as e:
            print("APITimeoutError: The request timed out.")
            print(
                "Solution: Retry your request after a brief wait and contact OpenAI if the issue persists."
            )
            print(e)
            time.sleep(retry_delay)

        except openai.APIConnectionError as e:
            print("APIConnectionError: Issue connecting to OpenAI services.")
            print(
                "Solution: Check your network settings, proxy configuration, SSL certificates, or firewall rules."
            )
            print(e)
            time.sleep(retry_delay)

        except openai.APIStatusError as e:
            # General APIStatusError should come after specific APIStatusError subclasses
            print("APIStatusError: Another non-200-range status code was received.")
            print(f"Status Code: {e.status_code}")
            print(f"Response: {e.response}")
            print(f"Cause: {e.__cause__}")
            time.sleep(retry_delay)

        except Exception as e:
            print("An unexpected error occurred.")
            print(e)
            time.sleep(retry_delay)

    return None
