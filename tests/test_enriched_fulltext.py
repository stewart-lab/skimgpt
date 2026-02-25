#!/usr/bin/env python3
"""
Test script to fetch and enrich full text for a given PMID.

Takes a PMID as an argument and returns the enriched full text,
including figure transcriptions reinjected into the text.

Usage:
    python tests/test_enriched_fulltext.py <PMID> [options]

Examples:
    python tests/test_enriched_fulltext.py 34567890
    python tests/test_enriched_fulltext.py 34567890 --save-xml --output result.json
    python tests/test_enriched_fulltext.py 34567890 --save-xml --output result.json --hypothesis "Does RBM20 antisense oligo treatment improve Hfpef mouse disease severity?"
    python tests/test_enriched_fulltext.py 34567890 --skip-figures --show-text
"""

import argparse
import json
import logging
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pubmed_fetcher import PubMedFetcher
from src.image_analyzer import ImageAnalyzer
from src.utils import Config

try:
    import tiktoken
except ImportError:
    tiktoken = None


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging for the test."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("test_enriched_fulltext")


def load_secrets() -> dict:
    """Load secrets from secrets.json file."""
    secrets_path = os.path.join(os.path.dirname(__file__), "..", "secrets.json")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(
            f"secrets.json not found at {secrets_path}. "
            "Please create one with PUBMED_API_KEY and GEMINI_API_KEY."
        )
    with open(secrets_path, "r") as f:
        return json.load(f)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for
        encoding_name: The tiktoken encoding name to use

    Returns:
        Number of tokens, or -1 if tiktoken is not available
    """
    if not tiktoken:
        return -1

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return -1


def fetch_enriched_fulltext(
    pmid: str,
    logger: logging.Logger,
    secrets: dict,
    skip_figures: bool = False,
    save_xml_dir: str = None,
    hypothesis: str = "",
) -> dict:
    """
    Fetch and enrich full text for a given PMID.

    Args:
        pmid: PubMed ID to fetch
        logger: Logger instance
        secrets: Dictionary containing API keys
        skip_figures: If True, skip figure processing
        save_xml_dir: Optional directory to save raw PMC XML for debugging

    Returns:
        Dictionary containing:
        - pmid: The PMID
        - pmcid: The PMCID (if available)
        - raw_data: The raw data from PMC
        - enriched_text: The fully enriched text with figure transcriptions
        - figures: List of processed figures with transcriptions
        - token_count: Total number of tokens in the enriched text
        - success: Boolean indicating success
        - error: Error message if any
    """
    result = {
        "pmid": pmid,
        "pmcid": None,
        "raw_data": None,
        "enriched_text": None,
        "figures": [],
        "token_count": None,
        "success": False,
        "error": None,
    }

    # Create a minimal config for PubMedFetcher
    class MinimalConfig:
        def __init__(self, logger):
            self.logger = logger

    config = MinimalConfig(logger)

    # Initialize PubMedFetcher
    logger.info(f"Initializing PubMedFetcher for PMID: {pmid}")
    fetcher = PubMedFetcher(
        config=config,
        email="test@example.com",
        api_key=secrets.get("PUBMED_API_KEY", ""),
    )

    # Fetch raw full text data
    logger.info("Fetching full text from PMC...")
    raw_data_map = fetcher.fetch_full_text_context(
        [pmid], return_raw=True, save_xml_dir=save_xml_dir
    )

    if pmid not in raw_data_map:
        result["error"] = (
            f"No PMC data found for PMID {pmid}. Article may not be in PMC."
        )
        logger.warning(result["error"])
        return result

    raw_data = raw_data_map[pmid]
    result["raw_data"] = raw_data
    result["pmcid"] = raw_data.get("pmcid")

    logger.info(f"Found PMCID: {result['pmcid']}")
    logger.info(f"Title: {raw_data.get('title', 'N/A')}")
    logger.info(f"Sections: {list(raw_data.get('sections', {}).keys())}")
    logger.info(f"Tables: {len(raw_data.get('tables', []))}")
    logger.info(f"Figures: {len(raw_data.get('figures', []))}")

    figures = raw_data.get("figures", [])
    pmcid = raw_data.get("pmcid")

    # Format initial full text (before figure enrichment)
    full_text_body = fetcher._format_fulltext_complete(raw_data)

    if figures and not skip_figures and pmcid:
        logger.info(f"Processing {len(figures)} figures...")

        # Initialize ImageAnalyzer
        try:
            image_analyzer = ImageAnalyzer(
                secrets=secrets, model_name="gemini-3-flash-preview", logger=logger
            )
        except Exception as e:
            logger.error(f"Failed to initialize ImageAnalyzer: {e}")
            image_analyzer = None

        if image_analyzer:
            # Create temporary directory for figures
            temp_fig_dir = tempfile.mkdtemp(prefix=f"pmid_{pmid}_figures_")
            logger.info(f"Downloading figures to: {temp_fig_dir}")

            try:
                # 1. Download figures from OA package
                figures = fetcher._download_figures_from_package(
                    pmcid, figures, temp_fig_dir
                )

                # 2. Filter to downloaded figures and analyze
                downloaded_figures = [f for f in figures if "local_path" in f]
                logger.info(f"Downloaded {len(downloaded_figures)} figures")

                if downloaded_figures:
                    analyzed_figures = image_analyzer.enhance_figure_descriptions(
                        downloaded_figures, full_text_body, hypothesis
                    )

                    # Update figures with analysis results
                    fig_map = {f["id"]: f for f in analyzed_figures}
                    for f in figures:
                        if f["id"] in fig_map:
                            f.update(fig_map[f["id"]])

                    result["figures"] = figures

            except Exception as e:
                logger.error(f"Error processing figures: {e}")
                result["error"] = f"Figure processing error: {e}"

        # Inject figures and tables using the source method
        enriched_text = fetcher.inject_figures_and_tables(raw_data, figures=figures)
    else:
        # No figure processing - just inject tables
        enriched_text = fetcher.inject_figures_and_tables(raw_data)

    result["enriched_text"] = (
        f"PMID: {pmid}\n[FULL-TEXT]\n{enriched_text}\n\n===END OF FULL TEXT===\n\n"
    )

    # Count tokens
    token_count = count_tokens(result["enriched_text"])
    result["token_count"] = token_count
    if token_count > 0:
        logger.info(f"Total tokens: {token_count:,}")

    result["success"] = True

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and enrich full text for a given PMID"
    )
    parser.add_argument("pmid", type=str, help="PubMed ID to fetch")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure processing (faster, but no transcriptions)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the enriched text (JSON format)",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        help="Hypothesis for the figure transcription",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the full enriched text to stdout",
    )
    parser.add_argument(
        "--save-xml", action="store_true", help="Save the raw PMC XML for debugging"
    )

    args = parser.parse_args()

    logger = setup_logging(args.debug)

    try:
        secrets = load_secrets()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Fetching enriched full text for PMID: {args.pmid}")

    # Set up XML save directory if requested
    save_xml_dir = None
    if args.save_xml:
        save_xml_dir = os.path.join(os.path.dirname(__file__), "..", "debug_xml")
        logger.info(f"Raw XML will be saved to: {save_xml_dir}")

    result = fetch_enriched_fulltext(
        pmid=args.pmid,
        logger=logger,
        secrets=secrets,
        skip_figures=args.skip_figures,
        save_xml_dir=save_xml_dir,
        hypothesis=args.hypothesis,
    )

    if result["success"]:
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"PMID: {result['pmid']}")
        logger.info(f"PMCID: {result['pmcid']}")
        logger.info(f"Figures processed: {len(result['figures'])}")
        logger.info(f"Enriched text length: {len(result['enriched_text'])} characters")
        if result["token_count"] and result["token_count"] > 0:
            logger.info(f"Total tokens: {result['token_count']:,}")

        if args.show_text:
            print("\n" + "=" * 60)
            print("ENRICHED TEXT:")
            print("=" * 60)
            print(result["enriched_text"])

        if args.output:
            # Save result to JSON file
            output_data = {
                "pmid": result["pmid"],
                "pmcid": result["pmcid"],
                "enriched_text": result["enriched_text"],
                "token_count": result["token_count"],
                "figures": [
                    {
                        "id": f.get("id"),
                        "label": f.get("label"),
                        "caption": f.get("caption"),
                        "enhanced_content": f.get("enhanced_content"),
                    }
                    for f in result["figures"]
                ],
                "success": result["success"],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Output saved to: {args.output}")

        # Print summary of figures
        if result["figures"]:
            print("\n" + "=" * 60)
            print("FIGURE TRANSCRIPTIONS:")
            print("=" * 60)
            for fig in result["figures"]:
                fig_id = fig.get("id", "unknown")
                label = fig.get("label", "")
                enhanced = fig.get("enhanced_content", "")[:200]
                print(f"\n{label} ({fig_id}):")
                print(f"  {enhanced}...")
    else:
        logger.error(f"Failed to fetch enriched text: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
