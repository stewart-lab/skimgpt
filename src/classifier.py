import logging
import openai
import re
import time
from typing import Any

from src import prompt_library as prompts_module
from src.utils import Config, clean_term_for_display, extract_json_from_markdown, extract_pmids

logger = logging.getLogger(__name__)

# Constants
SCORE_NA_RESPONSE = {"score": "N/A", "decision": "N/A"}

# OpenAI errors that are non-retryable (should break the retry loop immediately)
_NON_RETRYABLE_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
)

# OpenAI errors that are retryable (should sleep then continue)
_RETRYABLE_ERRORS = (
    openai.BadRequestError,
    openai.ConflictError,
    openai.UnprocessableEntityError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIStatusError,
)


def _list_len(value) -> int:
    """Return the length of value if it is a list, otherwise 0."""
    return len(value) if isinstance(value, list) else 0


def _generate_pubmed_urls(text: Any) -> list[str]:
    """Extract PMIDs from text and return corresponding PubMed URLs."""
    if not isinstance(text, str):
        logger.error(
            f"Expected string for 'text', but got {type(text)}. Converting to string."
        )
        text = str(text)
    return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in extract_pmids(text)]


def _build_relationship_result(
    result, prompt_text, urls, *, a_term="", b_term="", c_term="",
    relationship_label="", hypothesis="", url_keys=None,
) -> dict:
    """Build a standardised relationship result dict.

    Args:
        result: The LLM analysis result.
        prompt_text: The prompt that was sent.
        urls: Dict mapping relationship keys to URL lists.
        a_term/b_term/c_term: Cleaned display terms.
        relationship_label: Human-readable relationship string (e.g. "A - B - C").
        hypothesis: Formatted hypothesis string.
        url_keys: Which keys from *urls* to include. If None, include all.
    """
    if url_keys is not None:
        urls = {k: urls.get(k, []) for k in url_keys}
    return {
        "a_term": a_term,
        "b_term": b_term,
        "c_term": c_term,
        "Relationship": relationship_label,
        "Hypothesis": hypothesis,
        "Result": result,
        "Prompt": prompt_text,
        "URLS": urls,
    }


def calculate_relevance_ratios(out_df, config: Config):
    logger.debug(f" Complete Out df: {out_df.to_string()}")

    for col in ["ab_mask", "bc_mask", "ac_mask"]:
        if col not in out_df.columns:
            continue
        prefix = col.split("_")[0]
        ratio_col = f"{prefix}_relevance_ratio"
        fraction_col = f"{prefix}_relevance_fraction"
        logger.debug(f"  Computing {ratio_col} and {fraction_col}")
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
    processed_results = {}

    if config.is_skim_with_gpt:
        a_term = clean_term_for_display(row.get("a_term"))
        b_term = clean_term_for_display(row.get("b_term"))
        c_term = clean_term_for_display(row.get("c_term"))

        # Update row with cleaned terms so downstream functions use them
        row["a_term"] = a_term
        row["b_term"] = b_term
        row["c_term"] = c_term

        abc_result, abc_prompt, abc_urls = perform_analysis(
            row=row, config=config, relationship_type="A_B_C"
        )
        if abc_result or abc_prompt:
            processed_results["A_B_C_Relationship"] = _build_relationship_result(
                abc_result, abc_prompt, abc_urls,
                a_term=a_term, b_term=b_term, c_term=c_term,
                relationship_label=f"{a_term} - {b_term} - {c_term}",
                hypothesis=config.skim_hypotheses["ABC"].format(
                    a_term=a_term, b_term=b_term, c_term=c_term
                ),
                url_keys=["AB", "BC"],
            )

        ac_result, ac_prompt, ac_urls = perform_analysis(
            row=row, config=config, relationship_type="A_C"
        )
        if ac_result or ac_prompt:
            processed_results["A_C_Relationship"] = _build_relationship_result(
                ac_result, ac_prompt, ac_urls,
                a_term=a_term, b_term=b_term, c_term=c_term,
                relationship_label=f"{a_term} - {c_term}",
                hypothesis=config.skim_hypotheses["AC"].format(
                    a_term=a_term, c_term=c_term
                ),
                url_keys=["AC"],
            )

    elif config.is_dch:
        # DCH mode: direct hypothesis comparison (is_dch can only be True if is_km_with_gpt is True)
        hypothesis1 = row.get("hypothesis1")
        hypothesis2 = row.get("hypothesis2")

        if not (hypothesis1 and hypothesis2):
            logger.error("Missing hypotheses for DCH row.")
            return None

        result, prompt_text, urls = perform_analysis(
            row=row, config=config, relationship_type="A_B"
        )
        if result or prompt_text:
            processed_results["Hypothesis_Comparison"] = {
                "hypothesis1": hypothesis1,
                "hypothesis2": hypothesis2,
                "Result": result,
                "Prompt": prompt_text,
                "URLS": urls,
            }

    elif config.is_km_with_gpt:
        a_term = row.get("a_term")
        b_term = row.get("b_term")

        if not (a_term and b_term):
            logger.error("Missing 'a_term' or 'b_term' in row.")
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
            processed_results["A_B_Relationship"] = _build_relationship_result(
                ab_result, ab_prompt, ab_urls,
                a_term=a_term, b_term=b_term,
                relationship_label=f"{a_term} - {b_term}",
                hypothesis=config.km_hypothesis.format(a_term=a_term, b_term=b_term),
                url_keys=["AB"],
            )
    else:
        logger.warning(f"Job type '{config.job_type}' is not specifically handled.")

    logger.debug(f"Processed results: {processed_results}")
    return processed_results if any(processed_results.values()) else None


def perform_analysis(row: dict, config: Config, relationship_type: str) -> tuple:
    """Perform LLM analysis for a given relationship type.

    Extracts the relevant abstracts and terms from *row*, builds PubMed URLs,
    validates that required abstracts are present, then delegates to the
    frontier LLM for scoring.

    Returns:
        (result_list, prompt_text, urls_dict)
    """
    # -- Extract terms, abstracts, URLs, and expected counts per relationship type
    if relationship_type == "A_B_C":
        a_term = row.get("a_term", "")
        b_term = row.get("b_term", "")
        c_term = row.get("c_term", "")
        ab_abstracts = row.get("ab_abstracts", "")
        bc_abstracts = row.get("bc_abstracts", "")
        ac_abstracts = ""

        expected_count = _list_len(ab_abstracts) + _list_len(bc_abstracts)
        urls = {
            "AB": _generate_pubmed_urls(ab_abstracts) if ab_abstracts else [],
            "BC": _generate_pubmed_urls(bc_abstracts) if bc_abstracts else [],
        }
        if not ab_abstracts or not bc_abstracts:
            logger.error(
                f"Early exit for '{config.job_type}' / '{relationship_type}': "
                "Missing 'ab_abstracts' or 'bc_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
        consolidated_abstracts = ab_abstracts + bc_abstracts

    elif relationship_type == "A_C":
        a_term = row.get("a_term", "")
        b_term = ""
        c_term = row.get("c_term", "")
        ac_abstracts = row.get("ac_abstracts", "")

        expected_count = _list_len(ac_abstracts)
        urls = {
            "AC": _generate_pubmed_urls(ac_abstracts) if ac_abstracts else [],
        }
        if not ac_abstracts:
            logger.error(
                f"Early exit for '{config.job_type}' / '{relationship_type}': "
                "Missing 'ac_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
        consolidated_abstracts = ac_abstracts

    elif relationship_type == "A_B":
        if config.is_dch:
            a_term, b_term, c_term = "", "", ""
            ab_abstracts = row.get("ab_abstracts", "")
            expected_count = row.get("expected_per_abstract_count")
        else:
            a_term = row.get("a_term", "")
            b_term = row.get("b_term", "")
            c_term = ""
            ab_abstracts = row.get("ab_abstracts", "")
            expected_count = _list_len(ab_abstracts)

        urls = {
            "AB": _generate_pubmed_urls(ab_abstracts) if ab_abstracts else [],
        }
        if not config.is_dch and not ab_abstracts:
            logger.error(
                f"Early exit for '{relationship_type}': Missing 'ab_abstracts'."
            )
            return [SCORE_NA_RESPONSE], "", urls
        consolidated_abstracts = ab_abstracts

    else:
        logger.error(f"Unknown relationship type: {relationship_type}")
        return [SCORE_NA_RESPONSE], "", {}

    # -- Call the frontier LLM ------------------------------------------------
    try:
        logger.debug(f"Consolidated abstracts: {consolidated_abstracts}")
        logger.debug(
            f"Job={config.job_type}  Relationship={relationship_type}  "
            f"A={a_term}  B={b_term}  C={c_term}"
        )
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
        logger.debug(f"Analysis result: {result}")
        logger.debug(f"Prompt text: {prompt_text}")
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
    if not a_term and not config.is_dch:
        logger.error("A term is empty.")
        return [SCORE_NA_RESPONSE], ""

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
    logger.debug(f"Generated prompt text: {prompt_text}")

    # Derive expected per-abstract count from the prompt; prefer over provided value
    final_expected_count = expected_per_abstract_count
    try:
        m = re.search(r"Available PMIDs for [Cc]itation:\s*([0-9,\s]+)", prompt_text)
        if m:
            numbers = re.findall(r"\d+", m.group(1))
            if numbers:
                final_expected_count = len(numbers)
                logger.info(f"Derived expected_per_abstract_count from prompt: {final_expected_count}")
    except Exception as exc:
        logger.debug(f"Could not derive expected count from prompt: {exc}")

    response = call_openai_json(
        config.llm_client,
        prompt_text,
        config,
        expected_per_abstract_count=final_expected_count,
    )
    logger.debug(f"LLM response: {response}")
    result = [response] if response else []
    return result, prompt_text


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
    if config.is_dch:
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

    # Determine hypothesis template
    if config.is_km_with_gpt:
        hypothesis_template = config.km_hypothesis.format(b_term=b_term, a_term=a_term)
    elif config.is_skim_with_gpt:
        hypothesis_map = {
            "A_B_C": lambda: config.skim_hypotheses["ABC"].format(
                a_term=a_term, b_term=b_term, c_term=c_term
            ),
            "A_C": lambda: config.skim_hypotheses["AC"].format(
                a_term=a_term, c_term=c_term
            ),
            "A_B": lambda: config.skim_hypotheses["AB"].format(
                a_term=a_term, b_term=b_term
            ),
        }
        generator = hypothesis_map.get(relationship_type)
        if not generator:
            logger.error(f"Unknown relationship type for skim_with_gpt: {relationship_type}")
            return ""
        hypothesis_template = generator()
    else:
        logger.error(f"Unknown job type: {config.job_type}")
        return ""

    # Select the appropriate prompt function
    if config.is_skim_with_gpt and relationship_type == "A_C":
        prompt_function = getattr(prompts_module, "skim_with_gpt_ac", None)
    else:
        prompt_function = getattr(prompts_module, config.job_type, None)

    if not prompt_function:
        raise ValueError(
            f"Prompt function for '{relationship_type}' / '{config.job_type}' "
            "not found in prompts module."
        )

    # Build keyword arguments -- all prompt functions share these common args
    kwargs = {
        "a_term": a_term,
        "hypothesis_template": hypothesis_template,
        "consolidated_abstracts": content,
    }
    if relationship_type == "A_B_C":
        kwargs["b_term"] = b_term
        kwargs["c_term"] = c_term
    elif relationship_type == "A_C":
        kwargs["c_term"] = c_term
    elif relationship_type == "A_B":
        kwargs["b_term"] = b_term
    else:
        raise ValueError("Invalid relationship type specified.")

    return prompt_function(**kwargs)


def call_openai_json(client, prompt, config, expected_per_abstract_count: int | None = None):
    """Call the OpenAI-compatible API, validate the JSON response, and retry on transient errors."""
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

    messages = [{"role": "system", "content": system_instructions}]
    if expected_per_abstract_count is not None:
        messages.append({
            "role": "system",
            "content": (
                f"You MUST return exactly {expected_per_abstract_count} items in the "
                "per_abstract array. Include one entry per PMID listed. If an abstract "
                f"is irrelevant, label it '{irrelevant_label}' but still include it. "
                "Do not omit any."
            ),
        })
    messages.append({"role": "user", "content": prompt})

    params = {
        "messages": messages,
        "model": "deepseek-reasoner" if config.model == "r1" else config.model,
    }

    retry_delay = config.retry_delay

    for _ in range(config.max_retries):
        try:
            api_start = time.time()
            resp = client.chat.completions.create(**params)
            api_duration = time.time() - api_start

            total_tokens = resp.usage.total_tokens if getattr(resp, "usage", None) else "N/A"
            abstracts_count = expected_per_abstract_count or "N/A"
            logger.info(
                f"OpenAI API call completed in {api_duration:.2f}s "
                f"(model={params['model']}, abstracts={abstracts_count}, tokens={total_tokens})"
            )

            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from API.")

            payload = extract_json_from_markdown(content)
            _validate_payload(payload, config, expected_per_abstract_count)
            return payload

        except _NON_RETRYABLE_ERRORS as e:
            logger.error(f"{type(e).__name__}: {e}")
            # DeepSeek-specific: 402 (insufficient balance) is also non-retryable
            break

        except _RETRYABLE_ERRORS as e:
            logger.warning(f"{type(e).__name__}: {e}")
            time.sleep(retry_delay)
            continue

        except openai.APIError as e:
            # Catch-all for remaining API errors not in the tuples above
            status_code = getattr(e, "status_code", None)
            if config.model == "r1" and status_code == 402:
                logger.error("Insufficient Balance: Please add funds.")
                break
            if config.model == "r1" and status_code == 503:
                logger.error("Server Overloaded: Retrying after delay.")
                time.sleep(retry_delay)
                continue
            raise

        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            time.sleep(retry_delay)
            continue

    return None


def _validate_payload(
    payload: dict, config: Config, expected_per_abstract_count: int | None
) -> None:
    """Validate that the parsed LLM payload contains all required fields.

    Raises ValueError on validation failure so the retry loop can catch it.
    """
    required_fields = ["per_abstract", "score_rationale", "tallies", "score", "decision"]
    for key in required_fields:
        if key not in payload:
            raise ValueError(f"Missing required field '{key}' in model output.")

    # Enforce per_abstract length when expected count is provided
    if expected_per_abstract_count is not None:
        per_abstract = payload.get("per_abstract", [])
        if not isinstance(per_abstract, list) or len(per_abstract) != expected_per_abstract_count:
            actual = len(per_abstract) if isinstance(per_abstract, list) else "N/A"
            raise ValueError(
                f"per_abstract length {actual} does not match expected {expected_per_abstract_count}."
            )

    # Validate tallies depending on mode
    tallies = payload.get("tallies", {})
    if config.is_dch:
        required_tallies = ["support_H1", "support_H2", "both", "neither_or_inconclusive"]
    else:
        required_tallies = ["support", "refute", "inconclusive"]

    for tally_key in required_tallies:
        if tally_key not in tallies:
            raise ValueError(f"Missing required tally '{tally_key}' in model output.")
