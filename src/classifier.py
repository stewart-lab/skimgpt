import os
import openai
import time
import json
import importlib
import inspect
import re


def write_to_json(data, file_path):
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def process_single_row(row, config):
    job_type = config.job_type

    if job_type not in [
        "km_with_gpt",
        "position_km_with_gpt",
        "skim_with_gpt",
    ]:
        print("Invalid job type (caught in process_single_row)")
        return None

    (result, prompt, urls) = perform_analysis(job_type, row, config, {})

    # if everything is empty, then we have no data to process
    if not result and not prompt:
        return None
    return {
        "Relationship": f"{row['a_term']} - {row['b_term']}"
        + (f" - {row['c_term']}" if "c_term" in row else ""),
        "Result": result,
        "Prompt": prompt,
        "URLS": urls,
    }


def extract_pmids_and_generate_urls(text):
    # Regular expression to find PMIDs
    pmid_pattern = r"PMID:\s*(\d+)"

    # Find all PMIDs in the text
    pmids = re.findall(pmid_pattern, text)

    # Generate PubMed URLs
    pubmed_urls = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]

    return pubmed_urls


def perform_analysis(job_type, row, config, abstract_data):
    b_term = row["b_term"]
    a_term = row["a_term"]
    c_term = row.get("c_term", None) if job_type == "skim_with_gpt" else None
    consolidated_abstracts = (
        row["ab_pmid_intersection"]
        + row.get("bc_pmid_intersection", "")
        + row.get("ac_pmid_intersection", "")
    )
    urls = extract_pmids_and_generate_urls(consolidated_abstracts)
    result, prompt = analyze_abstract_with_gpt4(
        consolidated_abstracts, b_term, a_term, config, c_term=c_term
    )
    return result, prompt, urls


def analyze_abstract_with_gpt4(
    consolidated_abstracts, b_term, a_term, config, c_term=None
):
    if not b_term or not a_term:
        print("B term or A term is empty.")
        return []

    api_key = config.api_key
    if not api_key:
        raise ValueError("OpenAI API key is not set.")
    openai_client = openai.OpenAI(api_key=api_key)
    responses = []
    if not config.evaluate_single_abstract:
        prompt = generate_prompt(
            b_term=b_term,
            a_term=a_term,
            content=consolidated_abstracts,
            config=config,
            c_term=c_term if c_term is not None else None,
        )
        response = call_openai(openai_client, prompt, config)
        if response:
            responses.append(response)
    elif config.evaluate_single_abstract:
        for abstract in consolidated_abstracts:
            # Pass c_term if it is not None
            prompt = generate_prompt(
                b_term,
                a_term,
                abstract,
                config,
                c_term=c_term if c_term is not None else None,
            )
            response = call_openai(openai_client, prompt, config)
            if response:
                responses.append(response)
    else:
        raise ValueError("Please set True or False for evaluate_single_abstract.")

    return responses, prompt


def generate_prompt(b_term, a_term, content, config, c_term=None):
    job_type = config.job_type.lower()
    b_term = b_term.replace("&", " ")  # Clean up the term if necessary

    # Choose the correct hypothesis template based on the job type
    hypothesis_templates = {
        "km_with_gpt": config.km_hypothesis.format(b_term=b_term, a_term=a_term),
        "position_km_with_gpt": config.position_km_hypothesis.format(
            b_term=b_term, a_term=a_term
        ),
        "skim_with_gpt": config.skim_hypotheses["ABC"].format(
            c_term=c_term, a_term=a_term, b_term=b_term
        ),  # Assuming c_term is needed
    }

    hypothesis_template = hypothesis_templates.get(
        job_type, "No valid hypothesis for the provided JOB_TYPE."
    )
    if hypothesis_template.startswith("No valid"):
        return hypothesis_template

    # Dynamically import the prompts module (assuming this module contains relevant prompt generation functions)
    prompts_module = importlib.import_module("prompt_library")
    assert prompts_module, "Failed to import the prompts module."

    # Use job_type to fetch the corresponding prompt function
    prompt_function = getattr(prompts_module, job_type, None)
    if not prompt_function:
        raise ValueError(
            f"Prompt function for '{job_type}' not found in the prompts module."
        )

    # Prepare arguments for the prompt function
    prompt_args = (b_term, a_term, hypothesis_template, content)
    if "c_term" in inspect.signature(prompt_function).parameters and c_term:
        return prompt_function(*prompt_args, c_term=c_term)
    else:
        return prompt_function(*prompt_args)


def call_openai(client, prompt, config):
    retry_delay = config.global_settings["RETRY_DELAY"]
    max_retries = config.global_settings["MAX_RETRIES"]
    model = config.global_settings["MODEL"]
    max_tokens = config.global_settings["MAX_TOKENS"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical research analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                print("Empty response received from OpenAI API.")
                time.sleep(retry_delay)

        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            time.sleep(retry_delay)
            print(e.__cause__)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            print(e.__cause__)
    return None
