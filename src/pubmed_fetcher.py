from __future__ import annotations

import logging
import os
import re
import time
import warnings
from pathlib import Path

from Bio import Entrez

# Configure tiktoken cache directory before import to avoid permission issues
# in shared computing environments
tiktoken_cache_dir = Path.home() / ".cache" / "tiktoken"
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

import tiktoken

from src.utils import Config, extract_pmid

logger = logging.getLogger(__name__)

# Silence the specific Bio.Entrez.Parser warning about DTD files
warnings.filterwarnings("ignore", message="Failed to save .* at .*")

ABSTRACT_DELIMITER = "===END OF ABSTRACT==="


def _split_entries(text: str) -> list[str]:
    """Split delimited abstract text into a list of stripped, non-empty entries."""
    return [e.strip() for e in text.split(ABSTRACT_DELIMITER) if e.strip()]


def _join_entries(entries: list[str]) -> str:
    """Rejoin abstract entries with the standard delimiter."""
    return f"{ABSTRACT_DELIMITER}\n\n".join(entries) + f"{ABSTRACT_DELIMITER}\n\n"


class PubMedFetcher:
    def __init__(
        self,
        config: Config,
        email: str,
        api_key: str,
        max_retries: int = 10,
        backoff_factor: float = 0.5,
    ):
        """Initialize PubMed fetcher with credentials and retry settings."""
        self.config = config
        self.email = email
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.pmid_years: dict[str, int] = {}
        self._setup_entrez()
        logger.info("PubMedFetcher initialized")

    def _setup_entrez(self) -> None:
        """Configure Entrez with credentials."""
        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def validate_pmids(self, pmids: list) -> list[str]:
        """Validate PMIDs to ensure they are numeric."""
        valid_pmids = []
        for pmid in pmids:
            pmid_str = str(pmid)
            if pmid_str.isdigit():
                valid_pmids.append(pmid_str)
            else:
                logger.warning(f"Invalid PMID detected and skipped: {pmid}")
        return valid_pmids

    def _batch_pmids(self, pmids: list[str], batch_size: int = 200) -> list[list[str]]:
        """Split PMIDs into batches."""
        return [pmids[i : i + batch_size] for i in range(0, len(pmids), batch_size)]

    def _extract_publication_year(self, paper: dict) -> str:
        """Extract publication year from PubMed article data."""
        article = paper["MedlineCitation"]["Article"]

        # 1. Try ArticleDate
        pub_date = article.get("ArticleDate", [])
        if pub_date and "Year" in pub_date[0]:
            return pub_date[0]["Year"]

        # 2. Try Journal PubDate
        journal_pub_date = (
            article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        )
        if "Year" in journal_pub_date:
            return journal_pub_date["Year"]
        if "MedlineDate" in journal_pub_date:
            year_match = re.search(r"\d{4}", journal_pub_date["MedlineDate"])
            if year_match:
                return year_match.group(0)

        # 3. Try MedlineCitation DateCompleted
        medline_date = paper["MedlineCitation"].get("DateCompleted", {})
        if "Year" in medline_date:
            return medline_date["Year"]

        # Default to "0000" if no year found
        pmid = str(paper["MedlineCitation"]["PMID"])
        logger.warning(f"No publication year found for PMID {pmid}")
        return "0000"

    def _fetch_batch(self, batch: list[str]) -> dict:
        """Fetch a single batch of PMIDs with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                with Entrez.efetch(
                    db="pubmed", id=batch, retmode="xml", rettype="abstract"
                ) as efetch:
                    output = Entrez.read(efetch)

                returned_pmids = []
                returned_contents = []
                delimiter = f"\n\n{ABSTRACT_DELIMITER}\n\n"
                skipped_min_wc_pmids = []

                for paper in output.get("PubmedArticle", []):
                    pmid = str(paper["MedlineCitation"]["PMID"])
                    article = paper["MedlineCitation"]["Article"]
                    pub_year = self._extract_publication_year(paper)

                    title = article.get("ArticleTitle", "No title available")
                    abstract_text = " ".join(
                        article.get("Abstract", {}).get(
                            "AbstractText", ["No abstract available"]
                        )
                    )

                    if len(abstract_text.split()) >= self.config.min_word_count:
                        returned_pmids.append(pmid)
                        self.pmid_years[pmid] = int(pub_year)
                        content = f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract_text}{delimiter}"
                        returned_contents.append(content)
                    else:
                        skipped_min_wc_pmids.append(pmid)

                if skipped_min_wc_pmids:
                    logger.debug(
                        f"Excluded {len(skipped_min_wc_pmids)} PMIDs due to MIN_WORD_COUNT="
                        f"{self.config.min_word_count}. Example: {skipped_min_wc_pmids[:5]}"
                    )

                return {
                    "pmids": returned_pmids,
                    "contents": returned_contents,
                }

            except Exception as e:
                logger.error(
                    f"Attempt {attempt} - Error fetching abstracts for batch: {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached for batch. Skipping.")
                    return {}

    def fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """Fetch abstracts for a list of PMIDs."""
        pmids = self.validate_pmids(pmids)
        if not pmids:
            logger.error("No valid PMIDs to fetch.")
            return {}

        batches = self._batch_pmids(pmids)
        abstract_dict = {}

        for batch in batches:
            batch_result = self._fetch_batch(batch)
            if batch_result:
                abstract_dict.update(
                    dict(zip(batch_result["pmids"], batch_result["contents"]))
                )
            time.sleep(0.34)  # Rate limiting

        if not abstract_dict:
            logger.error("No abstracts fetched successfully.")
            return {}

        logger.info(
            f"Successfully fetched abstracts for {len(abstract_dict)} PMIDs."
        )

        # Filter abstracts by publication year bounds
        lower = self.config.censor_year_lower
        upper = self.config.censor_year_upper
        filtered_dict = {
            pmid: content
            for pmid, content in abstract_dict.items()
            if lower <= self.pmid_years.get(pmid, 0) <= upper
        }
        logger.info(
            f"Filtered {len(filtered_dict)}/{len(abstract_dict)} abstracts "
            f"by publication year bounds ({lower}-{upper})"
        )

        return filtered_dict

    def _get_year_for_entry(self, entry: str) -> int:
        """Extract PMID from an entry and look up its publication year."""
        pmid = extract_pmid(entry)
        if pmid:
            return self.pmid_years.get(pmid, 0)
        return 0

    def interleave_abstracts(
        self,
        text: str,
        n: int | None = None,
        top_n_most_cited: int = 0,
        top_n_most_recent: int = 0,
    ) -> str:
        """Interleave abstracts based on citation count and recency.

        If top_n_most_cited is 0, only returns most recent abstracts.
        If top_n_most_recent is 0, only returns most cited abstracts.
        Otherwise, interleaves based on the ratio of cited:recent.
        """
        if not isinstance(text, str) or text == "[]":
            return ""

        entries = _split_entries(text)
        if not entries:
            return ""

        # Original order represents most-cited ranking
        cited_entries = entries.copy()

        logger.debug("Original order (most cited):")
        for entry in cited_entries[:3]:
            pmid = extract_pmid(entry)
            if pmid:
                logger.debug(f"PMID: {pmid}, Year: {self.pmid_years.get(pmid, 0)}")

        # Create year-sorted version (newest first)
        recent_entries = sorted(
            [e for e in entries if extract_pmid(e)],
            key=self._get_year_for_entry,
            reverse=True,
        )

        logger.debug("\nYear-sorted order (most recent):")
        logged_pmids: set[str] = set()
        for entry in recent_entries[:15]:
            pmid = extract_pmid(entry)
            if pmid and pmid not in logged_pmids:
                logger.debug(f"PMID: {pmid}, Year: {self.pmid_years.get(pmid, 0)}")
                logged_pmids.add(pmid)

        # Calculate interleaving ratio
        if top_n_most_recent > 0:
            ratio = top_n_most_cited / top_n_most_recent
        elif top_n_most_cited > 0:
            ratio = float("inf")
        else:
            ratio = 1.0
        logger.debug(
            f"Interleaving ratio (cited:recent) = "
            f"{ratio if ratio != float('inf') else 'inf'}"
        )

        # Interleave based on ratio
        result: list[str] = []
        cited_idx = 0
        recent_idx = 0
        used_pmids: set[str] = set()

        def add_entry(entry: str) -> bool:
            """Add entry if not already present and limit not reached."""
            if n is not None and len(result) >= n:
                return False
            pmid = extract_pmid(entry)
            if pmid and pmid not in used_pmids:
                result.append(entry)
                used_pmids.add(pmid)
                return True
            return False

        while (
            cited_idx < len(cited_entries) or recent_idx < len(recent_entries)
        ) and (n is None or len(result) < n):
            made_progress = False

            if ratio > 1 and cited_idx < len(cited_entries):
                # More cited than recent
                expected_cited = round(recent_idx * ratio)
                while cited_idx < min(expected_cited, len(cited_entries)):
                    if add_entry(cited_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
                if recent_idx < len(recent_entries):
                    if add_entry(recent_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1
                elif cited_idx < len(cited_entries):
                    if add_entry(cited_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1

            elif ratio < 1 and recent_idx < len(recent_entries):
                # More recent than cited
                if ratio > 0:
                    expected_recent = round(cited_idx / ratio)
                else:
                    expected_recent = len(recent_entries)
                while recent_idx < min(expected_recent, len(recent_entries)):
                    if add_entry(recent_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1
                if cited_idx < len(cited_entries):
                    if add_entry(cited_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
                elif recent_idx < len(recent_entries):
                    if add_entry(recent_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1

            elif ratio == 1:
                # Equal numbers of cited and recent
                if cited_idx < len(cited_entries):
                    if add_entry(cited_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
                if recent_idx < len(recent_entries):
                    if add_entry(recent_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1

            if not made_progress:
                break

            if cited_idx >= len(cited_entries) and recent_idx >= len(recent_entries):
                break

            logger.debug(
                f"cited_idx: {cited_idx}, recent_idx: {recent_idx}, n: {n}"
            )
            logger.debug(
                f"Cited entries used: {cited_idx}, Recent entries used: {recent_idx}"
            )
            logger.debug(f"Unique PMIDs in result: {len(used_pmids)}")
            logger.debug(f"Final interleaved list contains {len(result)} entries")

        if not result:
            return ""

        logger.debug(
            f"Returning {len(result)} abstracts from interleave_abstracts"
        )
        return _join_entries(result)

    def optimize_text_length(
        self,
        text: str | list,
        max_tokens: int = 110000000,
        encoding_name: str = "cl100k_base",
        num_intersections: int = 1,
    ) -> str:
        """Truncate abstracts to fit within a token budget."""
        if isinstance(text, list):
            text = f"\n{ABSTRACT_DELIMITER}\n".join(text) if text else ""

        if not text or max_tokens <= 0:
            return ""

        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.error("tiktoken not installed. Required for token counting.")
            return text

        tokens_per_intersection = max_tokens // num_intersections

        entries = _split_entries(text)
        optimized_entries = []
        current_tokens = 0

        for entry in entries:
            entry_tokens = len(encoding.encode(entry))
            logger.debug(f"Entry tokens: {entry_tokens}")
            logger.debug(f"Current tokens: {current_tokens}")
            logger.debug(f"entry: {entry}")
            if current_tokens + entry_tokens <= tokens_per_intersection:
                optimized_entries.append(entry)
                current_tokens += entry_tokens
                logger.debug(f"adding entry: {entry}")
            else:
                logger.debug(f"breaking at entry: {entry}")
                break

        if not optimized_entries:
            return ""

        logger.debug(
            f"Returning {len(optimized_entries)} abstracts from optimize_text_length"
        )
        return _join_entries(optimized_entries)
