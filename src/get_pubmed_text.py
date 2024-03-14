import ast
import logging
import time
from xml.etree import ElementTree
import requests
from tqdm import tqdm


def extract_text_from_xml(element):
    text = element.text or ""
    for child in element:
        text += extract_text_from_xml(child)
        if child.tail:
            text += child.tail
    return text


def fetch_abstract_from_pubmed(config, pmid):
    global_settings = config.get("GLOBAL_SETTINGS", {})
    max_retries = global_settings.get("MAX_RETRIES", 3)
    retry_delay = global_settings.get("RETRY_DELAY", 5)
    base_url = global_settings.get(
        "BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    )
    pubmed_params = global_settings.get("PUBMED_PARAMS", {})

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params={**pubmed_params, "id": pmid})
            response.raise_for_status()  # Check if the request was successful
            tree = ElementTree.fromstring(response.content)
            break
        except (ElementTree.ParseError, requests.exceptions.RequestException) as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Skipping this PMID.")
                raise

    abstract_texts = [
        extract_text_from_xml(abstract) for abstract in tree.findall(".//AbstractText")
    ]
    abstract_text = " ".join(abstract_texts)
    # Append the PMID to the beginning of the abstract
    abstract_text = f"PMID: {pmid} Text: {abstract_text}"

    year = next((year.text for year in tree.findall(".//PubDate/Year")), None)
    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return abstract_text, paper_url, year


def abstract_quality_control(config, pmids, rate_limit, delay):
    min_word_count = config["GLOBAL_SETTINGS"].get(
        "MIN_WORD_COUNT", 100
    )  # Default to 100 if not specified
    if not pmids:
        return None
    results = {}

    pmid_batches = [pmids[i : i + rate_limit] for i in range(0, len(pmids), rate_limit)]

    for batch in tqdm(pmid_batches, desc = "Getting PMIDs"):
        for pmid in batch:
            try:
                abstract, url, year = fetch_abstract_from_pubmed(config, pmid)
                if not all([abstract, url, year]):
                    logging.warning(
                        f"PMID {pmid} has missing data and will be removed from pool."
                    )
                    continue

                if len(abstract.split()) < min_word_count:
                    logging.warning(
                        f"Abstract for PMID {pmid} is {len(abstract.split())} words. Removing from pool."
                    )
                    continue

                results[pmid] = (abstract, url, year)
            except Exception as e:  # Replace with specific exceptions
                logging.error(f"Error processing PMID {pmid}: {e}")
        time.sleep(delay)

    return results


def process_abstracts_data(config, pmids):
    abstracts_data = abstract_quality_control(
        config,
        pmids,
        config["GLOBAL_SETTINGS"]["RATE_LIMIT"],
        config["GLOBAL_SETTINGS"]["DELAY"],
    )
    if not abstracts_data:
        return None, None, None
    successful_pmids = get_successful_pmids(pmids, abstracts_data)
    consolidated_abstracts = [abstracts_data[pmid][0] for pmid in successful_pmids]
    paper_urls = [abstracts_data[pmid][1] for pmid in successful_pmids]
    publication_years = [abstracts_data[pmid][2] for pmid in successful_pmids]
    return consolidated_abstracts, paper_urls, publication_years


def parse_pmids(row, key):
    return ast.literal_eval(row[key])


def get_successful_pmids(pmids, abstracts_data):
    return [pmid for pmid in pmids if pmid in abstracts_data]


def handle_rate_limit(e, retry_delay):
    # Extract the retry-after value from the error message if available0-
    retry_after = int(e.response.headers.get("Retry-After", retry_delay))
    logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
    time.sleep(retry_after)
