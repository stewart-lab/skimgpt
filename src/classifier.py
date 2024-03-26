import os
import openai
import time
import json
import importlib
import inspect
import skim_and_km_api as skim

def write_to_json(data, file_path, output_directory):
    full_path = os.path.join(output_directory, file_path)
    with open(full_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def analyze_abstract_with_gpt4(
    consolidated_abstracts, b_term, a_term, config, c_term=None
):
    if not b_term or not a_term:
        print("B term or A term is empty.")
        return []

    api_key = config.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is not set.")
    openai_client = openai.OpenAI(api_key=api_key)
    responses = []
    if not config["Evaluate_single_abstract"]:
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
    elif config["Evaluate_single_abstract"]:
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
    job_type = config.get("JOB_TYPE", "").lower()
    
    # Define hypothesis templates directly based on job_type
    abc_hypothesis = config.get("SKIM_hypotheses")["ABC"].format(c_term=c_term, a_term=a_term, b_term=b_term)

    # Incorporate this into your hypothesis_templates dictionary
    hypothesis_templates = {
        "km_with_gpt": config.get("KM_hypothesis", "").format(b_term=b_term, a_term=a_term),
        "position_km_with_gpt": config.get("POSITION_KM_hypothesis", "").format(b_term=b_term, a_term=a_term),
        "skim_with_gpt": abc_hypothesis  # Using the formatted ABC hypothesis directly
    }

    # Fetch the hypothesis template for the given job_type
    hypothesis_template = hypothesis_templates.get(job_type)
    if not hypothesis_template:
        return "No valid hypothesis for the provided JOB_TYPE."
    
    # Dynamically import the prompts module
    prompts_module = importlib.import_module("prompt_library")
    assert prompts_module, "Failed to import the prompts module."

    # Use job_type to fetch the corresponding prompt function
    prompt_function = getattr(prompts_module, job_type, None)
    if not prompt_function:
        raise ValueError(f"Prompt function for '{job_type}' not found in the prompts module.")

    prompt_args = (b_term, a_term, content, config)
    if "hypothesis_template" in inspect.signature(prompt_function).parameters:
        prompt_args = (b_term, a_term, hypothesis_template, content)
        return prompt_function(*prompt_args, c_term=c_term) if c_term is not None else prompt_function(*prompt_args)
    else:
        return prompt_function(*prompt_args, c_term=c_term) if "c_term" in inspect.signature(prompt_function).parameters and c_term is not None else prompt_function(*prompt_args)

def perform_analysis(job_type, row, config):
    if job_type == "km_with_gpt" or job_type == "position_km_with_gpt":
        consolidated_abstracts = row["ab_pmid_intersection"]
    elif job_type == "skim_with_gpt":
        if "ac_pmid_intersection" in row:
            consolidated_abstracts = row["ac_pmid_intersection"] + row["bc_pmid_intersection"] + row["ab_pmid_intersection"]
        consolidated_abstracts = row["bc_pmid_intersection"] + row["ab_pmid_intersection"]
    else:
        print("Invalid job type (caught in perform_analysis)")
        return None, None, None
    b_term = row["b_term"]
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    c_term = row.get("c_term", None)  # Handle c_term dynamically

    # If there are no abstracts, return None for all fields
    if not consolidated_abstracts:
        return None, None, None

    # Call analyze_abstract_with_gpt4 or a similar function you have defined
    result, prompt = analyze_abstract_with_gpt4(
        consolidated_abstracts, b_term, a_term, config, c_term=c_term
    )

    return result, prompt, consolidated_abstracts

def process_single_row(row, config):
    job_type = config.get("JOB_TYPE")

    # Validate job_type
    if job_type not in [
        "km_with_gpt",
        "position_km_with_gpt",
        "skim_with_gpt",
    ]:
        print("Invalid job type (caught in process_single_row)")
        return None

    result, prompt, consolidated_abstracts = perform_analysis(job_type, row, config)

    # If all fields are None, there's no data to process
    if not result and not prompt and not consolidated_abstracts:
        return None

    return {
        "Term": row["b_term"],
        "Result": result,
        "Prompt": prompt,
        "Abstracts": consolidated_abstracts,
    }

def test_openai_connection(config):
    openai.api_key = config["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model=config["GLOBAL_SETTINGS"]["MODEL"],
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": "Test connection to OpenAI."},
            ],
        )
        print("Successfully connected to OpenAI!")
    except Exception as e:
        print(f"Failed to connect to OpenAI. Error: {e}")

def call_openai(client, prompt, config):
    retry_delay = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]
    max_retries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]
    model = config["GLOBAL_SETTINGS"]["MODEL"]
    max_tokens = config["GLOBAL_SETTINGS"]["MAX_TOKENS"]

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


def save_to_json(data, config, output_directory):
    output_filename = os.path.join(
        output_directory, config["OUTPUT_JSON"] + "_filtered.json"
    )
    # Adding "_filtered" to the filename
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Filtered results have been saved to {output_filename}")


def api_cost_estimator(df, config):
    job_type = config.get("JOB_TYPE", "")
    max_abstracts = config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"]
    estimated_cost = 0
    total_calls = 0

    def read_terms_length(file_path):
        return len(skim.read_terms_from_file(file_path))

    if job_type in ["drug_discovery_validation", "pathway_augmentation"]:
        estimated_cost = max_abstracts * len(df) * 0.006
    elif job_type == "km_with_gpt":
        if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERM_LIST"]:
            term_length = read_terms_length(
                config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
            )
        else:
            term_length = 1  # Default value if A_TERM_LIST is not set

        num_b_terms = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"]
        total_calls = max_abstracts * term_length * num_b_terms
        estimated_cost = total_calls * 0.006
    elif job_type == "post_km_analysis":
        robust_setting = (
            config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"]
            .get("robust", "False")
            .lower()
        )

        if robust_setting == "true":
            total_calls = sum(
                len(eval(row["panc & ggp & kras-mapk set"])) // (max_abstracts // 2)
                + len(eval(row["brd & ggp set"])) // (max_abstracts // 2)
                for _, row in df.iterrows()
            )
        else:
            total_calls = df.apply(
                lambda row: min(
                    len(eval(row["panc & ggp & kras-mapk set"]))
                    + len(eval(row["brd & ggp set"])),
                    max_abstracts,
                ),
                axis=1,
            ).sum()

        estimated_cost = total_calls * 0.06

    user_input = input(
        f"The following job consists of {total_calls} abstracts and will cost roughly ${estimated_cost:.2f} in GPT-4 API calls. Do you wish to proceed? [Y/n]: "
    )
    if user_input.lower() != "y":
        print("Exiting workflow.")
        return False
    return True

