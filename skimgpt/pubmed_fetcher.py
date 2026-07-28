from __future__ import annotations

import io
import logging
import os
import re
import shutil
import socket
import tarfile
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List

import requests
from Bio import Entrez

# Configure tiktoken cache directory before import to avoid permission issues
# in shared computing environments
tiktoken_cache_dir = Path.home() / ".cache" / "tiktoken"
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

import tiktoken

from skimgpt.retry import retry_call
from skimgpt.utils import (
    ABSTRACT_DELIMITER,
    Config,
    extract_pmid,
    join_abstract_entries,
    split_abstract_entries,
)

logger = logging.getLogger(__name__)

# Silence the specific Bio.Entrez.Parser warning about DTD files
warnings.filterwarnings("ignore", message="Failed to save .* at .*")

EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"


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
        # Full-text methods (ported from the fulltext feature) log via self.logger
        # and throttle PMC/OA requests via self._rate_limit_delay.
        self.logger = logger
        self._rate_limit_delay = 0.1 if api_key else 0.34
        self._setup_entrez()
        self._check_connectivity()
        logger.info("PubMedFetcher initialized")

    def _setup_entrez(self) -> None:
        """Configure Entrez with credentials."""
        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def _check_connectivity(self) -> None:
        """Verify DNS resolution for the PubMed eutils host.

        Raises:
            socket.gaierror: If DNS resolution fails.
        """
        try:
            ip_address = socket.gethostbyname(EUTILS_HOST)
            logger.debug(f"DNS resolution OK: {EUTILS_HOST} -> {ip_address}")
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for '{EUTILS_HOST}': {e}")
            raise

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

        def _attempt() -> dict:
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

        def _on_retryable(exc: BaseException, attempt: int) -> None:
            logger.error(f"Attempt {attempt} - Error fetching abstracts for batch: {exc}")

        return retry_call(
            _attempt,
            max_retries=self.max_retries,
            delay=self.backoff_factor,
            backoff_factor=2.0,
            on_retryable=_on_retryable,
            default={},
        )

    def fetch_abstracts_iter(self, pmids: list[str]) -> Iterator[dict[str, str]]:
        """Yield {pmid: content} dicts batch-by-batch as PubMed responds.

        Used by pipelined callers that want to start downstream work (e.g.
        Triton inference submission) as soon as the first batch lands rather
        than waiting for the full fetch to complete.

        Each yielded dict only contains PMIDs that pass the censor-year
        bounds; out-of-range PMIDs are filtered per-batch to match the
        all-at-once behaviour of ``fetch_abstracts``.
        """
        pmids = self.validate_pmids(pmids)
        if not pmids:
            logger.error("No valid PMIDs to fetch.")
            return

        lower = self.config.censor_year_lower
        upper = self.config.censor_year_upper

        for batch in self._batch_pmids(pmids):
            batch_result = self._fetch_batch(batch)
            if batch_result:
                batch_dict = dict(zip(batch_result["pmids"],
                                      batch_result["contents"]))
                filtered = {
                    pmid: content
                    for pmid, content in batch_dict.items()
                    if lower <= self.pmid_years.get(pmid, 0) <= upper
                }
                if filtered:
                    yield filtered
            time.sleep(0.34)  # Rate limiting

    def fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """Fetch abstracts for a list of PMIDs (all-at-once convenience wrapper)."""
        abstract_dict: dict[str, str] = {}
        for batch_dict in self.fetch_abstracts_iter(pmids):
            abstract_dict.update(batch_dict)

        if not abstract_dict:
            logger.error("No abstracts fetched successfully.")
            return {}

        logger.info(
            f"Successfully fetched abstracts for {len(abstract_dict)} PMIDs "
            f"(censor years {self.config.censor_year_lower}-"
            f"{self.config.censor_year_upper})."
        )
        return abstract_dict

    def _get_year_for_entry(self, entry: str) -> int:
        """Extract PMID from an entry and look up its publication year."""
        pmid = extract_pmid(entry)
        if pmid:
            return self.pmid_years.get(pmid, 0)
        return 0

    @staticmethod
    def _compute_interleave_ratio(
        top_n_most_cited: int, top_n_most_recent: int,
    ) -> float:
        """Return the cited-to-recent interleaving ratio."""
        if top_n_most_recent > 0:
            return top_n_most_cited / top_n_most_recent
        if top_n_most_cited > 0:
            return float("inf")
        return 1.0

    @staticmethod
    def _execute_interleave(
        cited_entries: list[str],
        recent_entries: list[str],
        ratio: float,
        n: int | None,
    ) -> list[str]:
        """Merge *cited_entries* and *recent_entries* according to *ratio*.

        Entries are de-duplicated by PMID.  At most *n* entries are returned
        (``None`` means no limit).
        """
        result: list[str] = []
        cited_idx = 0
        recent_idx = 0
        used_pmids: set[str] = set()

        def add_entry(entry: str) -> bool:
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

        return result

    def interleave_abstracts(
        self,
        text: str,
        n: int | None = None,
        top_n_most_cited: int = 0,
        top_n_most_recent: int = 0,
    ) -> str:
        """Interleave abstracts based on citation count and recency.

        The input *text* is a delimiter-separated string of abstracts in
        most-cited order.  This method creates a year-sorted copy, computes an
        interleaving ratio from the ``top_n_most_cited`` / ``top_n_most_recent``
        parameters, and merges the two orderings into a single de-duplicated
        list capped at *n* entries.
        """
        if not isinstance(text, str) or text == "[]":
            return ""

        entries = split_abstract_entries(text)
        if not entries:
            return ""

        cited_entries = entries.copy()

        logger.debug("Original order (most cited):")
        for entry in cited_entries[:3]:
            pmid = extract_pmid(entry)
            if pmid:
                logger.debug(f"PMID: {pmid}, Year: {self.pmid_years.get(pmid, 0)}")

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

        ratio = self._compute_interleave_ratio(top_n_most_cited, top_n_most_recent)
        logger.debug(
            f"Interleaving ratio (cited:recent) = "
            f"{ratio if ratio != float('inf') else 'inf'}"
        )

        result = self._execute_interleave(cited_entries, recent_entries, ratio, n)

        if not result:
            return ""

        logger.debug(
            f"Returning {len(result)} abstracts from interleave_abstracts"
        )
        return join_abstract_entries(result)

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

        entries = split_abstract_entries(text)
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
        return join_abstract_entries(optimized_entries)

    def _fetch_pmc_ids(self, pmids: List[str]) -> Dict[str, str]:
        """Map PMIDs to PMCIDs for articles available in PMC.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Dictionary mapping {pmid: pmcid} for articles in PMC
        """
        if not pmids:
            return {}
            
        pmid_to_pmcid = {}
        
        try:
            # Use elink to map PMIDs to PMCIDs
            with Entrez.elink(dbfrom="pubmed", db="pmc", id=pmids) as handle:
                records = Entrez.read(handle)
            
            # Parse the linkset results
            for record in records:
                if "LinkSetDb" in record and record["LinkSetDb"]:
                    # Get the source PMID
                    source_pmid = str(record["IdList"][0]) if "IdList" in record and record["IdList"] else None
                    
                    # Look for PMC links
                    for linksetdb in record["LinkSetDb"]:
                        if linksetdb.get("LinkName") == "pubmed_pmc":
                            # Get the linked PMCID
                            if "Link" in linksetdb and linksetdb["Link"]:
                                pmcid = str(linksetdb["Link"][0]["Id"])
                                if source_pmid:
                                    pmid_to_pmcid[source_pmid] = pmcid
                                    self.logger.debug(f"Mapped PMID {source_pmid} -> PMCID {pmcid}")
            
            self.logger.info(f"Found {len(pmid_to_pmcid)} PMC IDs out of {len(pmids)} PMIDs")
            
        except Exception as e:
            self.logger.error(f"Error mapping PMIDs to PMCIDs: {e}")
            
        return pmid_to_pmcid

    def _fetch_pmc_fulltext(self, pmcid: str, save_xml_path: str = None) -> Dict[str, Any]:
        """Fetch and parse full-text article from PMC.
        
        Args:
            pmcid: PubMed Central ID
            save_xml_path: Optional path to save raw XML for debugging
            
        Returns:
            Dictionary with keys: title, abstract, sections, tables, figures
            Returns None if fetch fails
        """
        try:
            # Fetch PMC article XML
            with Entrez.efetch(db="pmc", id=pmcid, retmode="xml") as handle:
                xml_content = handle.read()
            
            # Optionally save raw XML for debugging
            if save_xml_path:
                try:
                    with open(save_xml_path, "wb") as f:
                        f.write(xml_content)
                    self.logger.info(f"Saved raw XML to: {save_xml_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save XML to {save_xml_path}: {e}")
            
            # Check for publisher restriction
            if b"publisher of this article does not allow downloading of the full text" in xml_content:
                self.logger.warning(f"Publisher restricts full text XML download for PMCID {pmcid}. Only abstract/metadata available.")
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Initialize result structure
            result = {
                "title": "",
                "abstract": "",
                "sections": {},
                "tables": [],
                "figures": []
            }
            
            # Extract title
            title_elem = root.find(".//article-title")
            if title_elem is not None:
                result["title"] = self._extract_text(title_elem)
            
            # Extract abstract
            abstract_elem = root.find(".//abstract")
            if abstract_elem is not None:
                result["abstract"] = self._extract_text(abstract_elem)
            
            # Extract body sections
            body = root.find(".//body")
            if body is not None:
                result["sections"] = self._extract_sections(body)
            
            # Extract tables
            result["tables"] = self._extract_tables(root)
            
            # Extract figures
            result["figures"] = self._extract_figures(root)
            
            # Include PMCID for downstream package downloading
            result["pmcid"] = pmcid
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching PMC full-text for PMCID {pmcid}: {e}")
            return None

    def _extract_text(self, element) -> str:
        """Extract all text content from an XML element, preserving paragraph structure."""
        if element is None:
            return ""
        
        text_parts = []
        
        # Get text from paragraphs if they exist
        paragraphs = element.findall(".//p")
        if paragraphs:
            for p in paragraphs:
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    text_parts.append(p_text)
        else:
            # If no paragraphs, get all text
            text = "".join(element.itertext()).strip()
            if text:
                text_parts.append(text)
        
        return " ".join(text_parts)

    def _extract_sections(self, body_element) -> Dict[str, str]:
        """Extract sections from the article body with inline figure placeholders.
        
        Args:
            body_element: XML element containing the article body
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        
        for sec in body_element.findall(".//sec"):
            # Get section title
            title_elem = sec.find("./title")
            if title_elem is not None:
                section_title = "".join(title_elem.itertext()).strip()
            else:
                section_title = "Untitled Section"
            
            # Normalize section titles
            title_lower = section_title.lower()
            if any(keyword in title_lower for keyword in ["introduction", "background"]):
                key = "Introduction"
            elif any(keyword in title_lower for keyword in ["method", "material", "experimental"]):
                key = "Methods"
            elif any(keyword in title_lower for keyword in ["result", "finding"]):
                key = "Results"
            elif any(keyword in title_lower for keyword in ["discussion", "conclusion"]):
                key = "Discussion"
            else:
                key = section_title
            
            # Extract content including paragraphs and inline figures
            content_parts = []
            
            # Iterate over all children to preserve order
            for child in sec:
                if child.tag == "title":
                    continue
                    
                if child.tag == "p":
                    p_text = "".join(child.itertext()).strip()
                    if p_text:
                        content_parts.append(p_text)
                        
                        # Check for figure references in this paragraph
                        # This handles figures that are in floats-group or elsewhere but referenced here
                        xrefs = child.findall(".//xref[@ref-type='fig']")
                        seen_rids = set()
                        for xref in xrefs:
                            rids = xref.get("rid", "").split()
                            for rid in rids:
                                if rid and rid not in seen_rids:
                                    content_parts.append(f"\n[[FIGURE:{rid}]]\n")
                                    seen_rids.add(rid)
                        
                        # Check for table references in this paragraph
                        table_xrefs = child.findall(".//xref[@ref-type='table']")
                        seen_table_rids = set()
                        for xref in table_xrefs:
                            rids = xref.get("rid", "").split()
                            for rid in rids:
                                if rid and rid not in seen_table_rids:
                                    content_parts.append(f"\n[[TABLE:{rid}]]\n")
                                    seen_table_rids.add(rid)
                        
                elif child.tag == "fig":
                    fig_id = child.get("id")
                    if fig_id:
                        # Insert placeholder
                        content_parts.append(f"\n[[FIGURE:{fig_id}]]\n")
                        
                elif child.tag == "sec":
                    # Handle nested sections recursively if needed, 
                    # but typically standard PMC structure is flat enough for top-level handling
                    # or we just grab text from them.
                    # For simplicity, we can recurse or just grab text. 
                    # Let's simple-recurse by extracting text from this nested sec.
                    # Actually, the outer loop findall(".//sec") might catch nested sections 
                    # if we are not careful about direct children vs descendants.
                    # findall(".//sec") finds ALL descendants. 
                    # This means nested sections are processed as separate keys in 'sections' dict?
                    # The current implementation uses keys like 'Results'. 
                    # If we have nested sections, they might overwrite or append.
                    # 'if key in sections: sections[key] += ...' handles append.
                    # So we don't need to handle child 'sec' here if the outer loop catches it.
                    pass
            
            if content_parts:
                joined_content = " ".join(content_parts)
                # Cleanup extra spaces around newlines
                joined_content = joined_content.replace(" \n[[FIGURE", "\n[[FIGURE").replace("]]\n ", "]]\n")
                
                if key in sections:
                    sections[key] += "\n\n" + joined_content
                else:
                    sections[key] = joined_content
        
        return sections

    def _extract_tables(self, root) -> List[Dict[str, Any]]:
        """Extract tables from the article.
        
        Args:
            root: Root XML element
            
        Returns:
            List of dictionaries with table info (id, caption, data)
        """
        tables = []
        
        for table_wrap in root.findall(".//table-wrap"):
            table_info = {}
            
            # Extract table ID from table-wrap (e.g., "TB1", "TB2")
            table_id = table_wrap.get("id")
            if table_id:
                table_info["id"] = table_id
            else:
                # Fallback to label text
                label = table_wrap.find("./label")
                if label is not None:
                    table_info["id"] = "".join(label.itertext()).strip()
                else:
                    table_info["id"] = f"table_{len(tables) + 1}"
            
            # Store label for display
            label = table_wrap.find("./label")
            if label is not None:
                table_info["label"] = "".join(label.itertext()).strip()
            else:
                table_info["label"] = table_info["id"]
            
            # Extract caption
            caption = table_wrap.find(".//caption")
            if caption is not None:
                table_info["caption"] = self._extract_text(caption)
            else:
                table_info["caption"] = ""
            
            # Extract table data (simplified - just extract text rows)
            table_elem = table_wrap.find(".//table")
            if table_elem is not None:
                rows = []
                for tr in table_elem.findall(".//tr"):
                    row_data = []
                    for cell in tr.findall(".//td") + tr.findall(".//th"):
                        cell_text = "".join(cell.itertext()).strip()
                        row_data.append(cell_text)
                    if row_data:
                        rows.append(row_data)
                
                table_info["data"] = rows
            else:
                table_info["data"] = []
            
            tables.append(table_info)
        
        return tables

    def _extract_figures(self, root) -> List[Dict[str, str]]:
        """Extract figure metadata from the article.
        
        Args:
            root: Root XML element
            
        Returns:
            List of dictionaries with figure info (id, label, caption, graphic_ref)
        """
        figures = []
        
        for fig in root.findall(".//fig"):
            fig_info = {}
            
            # Extract figure ID
            fig_id = fig.get("id", "")
            fig_info["id"] = fig_id
            
            # Extract label
            label = fig.find("./label")
            if label is not None:
                fig_info["label"] = "".join(label.itertext()).strip()
            else:
                fig_info["label"] = f"Figure {len(figures) + 1}"
            
            # Extract caption
            caption = fig.find(".//caption")
            if caption is not None:
                fig_info["caption"] = self._extract_text(caption)
            else:
                fig_info["caption"] = ""
            
            # Extract graphic reference (image filename)
            graphic = fig.find(".//graphic")
            if graphic is not None:
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                fig_info["graphic_ref"] = href
            else:
                fig_info["graphic_ref"] = ""
            
            figures.append(fig_info)
        
        return figures
    
    def _fetch_oa_package_url(self, pmcid: str) -> str:
        """Fetch the FTP/HTTPS URL for the Open Access package of the article.
        
        Args:
            pmcid: PubMed Central ID (e.g., 'PMC3148254')
            
        Returns:
            URL to the tar.gz package or None if not found/error.
        """
        try:
            # PMC ID must have 'PMC' prefix for the OA API
            if not pmcid.startswith("PMC"):
                query_id = f"PMC{pmcid}"
            else:
                query_id = pmcid
                
            api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
            params = {"id": query_id}
            
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code != 200:
                self.logger.warning(f"OA API returned {response.status_code} for {pmcid}")
                return None
                
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find the link with format='tgz'
            # Path: OA -> records -> record -> link check format='tgz'
            for link in root.findall(".//link"):
                if link.get("format") == "tgz":
                    href = link.get("href")
                    # Prefer HTTPS if returned as FTP
                    if href and href.startswith("ftp://"):
                        href = href.replace("ftp://", "https://", 1)
                    return href
            
            self.logger.debug(f"No tgz link found for {pmcid} in OA API response")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching OA package URL for {pmcid}: {e}")
            return None

    def _download_figures_from_package(self, pmcid: str, figures_list: List[Dict], output_dir: str) -> List[Dict]:
        """Download OA package and extract requested figures.
        
        Args:
            pmcid: PMCID string (e.g. PMC3148254)
            figures_list: List of figure dicts with 'graphic_ref'
            output_dir: Directory to save figures to
            
        Returns:
            Updated figures_list with 'local_path' populated
        """
        try:
            package_url = self._fetch_oa_package_url(pmcid)
            if not package_url:
                self.logger.warning(f"No OA package URL found for {pmcid}")
                return figures_list
            
            # Ensure output_dir is a Path object
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)

            self.logger.info(f"Downloading OA package for {pmcid} from {package_url}")
            response = requests.get(package_url, stream=True, timeout=60)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to download tarball: {response.status_code}")
                return figures_list
                
            # Create a set of graphic filenames we want
            # graphic_ref often lacks extension in XML or matches filename in tarball
            # We usually look for matching filenames.
            wanted_graphics = set()
            for fig in figures_list:
                ref = fig.get("graphic_ref")
                if ref:
                    wanted_graphics.add(ref)
            
            if not wanted_graphics:
                return figures_list
                
            extracted_count = 0
            
            # Use tarfile on the streamed content
            # We need to wrap raw stream in a file-like object or download to temp
            # Downloading to temp file is safer for seek operations if needed by tarfile
            with tempfile.NamedTemporaryFile(delete=True) as tmp_tar:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_tar.write(chunk)
                tmp_tar.flush()
                tmp_tar.seek(0)
                
                with tarfile.open(fileobj=tmp_tar, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                            
                        # Check if this file is one of our wanted graphics
                        # The member name includes directory, e.g. "PMC3148254/pone.0023061.g001.jpg"
                        filename = os.path.basename(member.name)
                        
                        # Check match (exact or without extension)
                        # graphic_ref might be "pone.0023061.g001.jpg" or "pone.0023061.g001"
                        # We try to match flexibility
                        match_found = False
                        matched_ref = None
                        
                        if filename in wanted_graphics:
                            match_found = True
                            matched_ref = filename
                        else:
                            # Try matching without extension
                            name_no_ext = os.path.splitext(filename)[0]
                            if name_no_ext in wanted_graphics:
                                match_found = True
                                matched_ref = name_no_ext
                                
                        if match_found:
                            # Extract to output_dir
                            target_path = output_dir / filename
                            with tar.extractfile(member) as source, open(target_path, "wb") as dest:
                                shutil.copyfileobj(source, dest)
                                
                            # Update the figure dict with local path
                            for fig in figures_list:
                                if fig.get("graphic_ref") == matched_ref:
                                    fig["local_path"] = str(target_path)
                            
                            extracted_count += 1
            
            self.logger.info(f"Extracted {extracted_count} figures for {pmcid}")
            
        except Exception as e:
            self.logger.error(f"Error downloading/extracting figures for {pmcid}: {e}")
            
        return figures_list



    def _format_fulltext_complete(self, content: Dict[str, Any]) -> str:
        """Format complete full-text content without truncation.
        
        Used for sending to AI chunker which has 1M+ token context window.
        Includes all sections, tables, and figure metadata.
        
        Args:
            content: Dictionary from _fetch_pmc_fulltext with sections, tables, figures
            
        Returns:
            Complete formatted string with all content
        """
        parts = []
        
        # Title
        if content.get("title"):
            parts.append(f"Title: {content['title']}\n")
        
        # Abstract
        if content.get("abstract"):
            parts.append(f"\nAbstract: {content['abstract']}\n")
        
        # All sections in document order (dict preserves insertion order in Python 3.7+)
        # Sections now contain [[TABLE:id]] and [[FIGURE:id]] placeholders
        sections = content.get("sections", {})
        for section_name, section_content in sections.items():
            parts.append(f"\n{section_name}: {section_content}\n")
        
        # Don't append tables/figures at end - they should be injected inline via placeholders
        # Tables and figures metadata are still available in content dict for injection
        
        return "".join(parts)

    def _format_single_table(self, table: Dict) -> str:
        """Format a single table for inline injection.
        
        Args:
            table: Table dictionary with id, label, caption, and data
            
        Returns:
            Formatted table string with end delimiter
        """
        parts = []
        table_label = table.get('label', table['id'])
        parts.append(f"\n[TABLE {table_label}]: {table['caption']}\n")
        
        # Format table data
        if table.get("data"):
            parts.append("\nData:")
            for row in table["data"]:
                parts.append(" | ".join(row))
            parts.append("")  # Add blank line after table
        
        parts.append("===END TABLE===")
        
        return "\n".join(parts)

    def inject_figures_and_tables(self, raw_data: Dict[str, Any], figures: List[Dict] = None) -> str:
        """Inject figure transcriptions and tables into section text and format.
        
        This method:
        1. Injects figure transcriptions (if provided) into sections at [[FIGURE:id]] placeholders
        2. Injects formatted tables into sections at [[TABLE:id]] placeholders
        3. Cleans up any remaining unreplaced placeholders
        4. Returns the complete formatted text
        
        Args:
            raw_data: Dictionary from _fetch_pmc_fulltext with sections, tables, figures
            figures: Optional list of figures with enhanced_content from ImageAnalyzer
            
        Returns:
            Complete formatted string with injected content
        """
        import re
        
        sections = raw_data.get("sections", {})
        
        # 1. Inject figure transcriptions if provided
        if figures:
            injected_figures = set()
            
            # Collect modifications first (don't iterate and modify simultaneously)
            modifications = {}
            for sec_name, sec_text in sections.items():
                modified_text = sec_text
                for fig in figures:
                    fig_id = fig.get("id")
                    
                    # Skip if this figure was already injected
                    if fig_id in injected_figures:
                        continue
                        
                    transcription = fig.get("enhanced_content", fig.get("caption", ""))
                    placeholder = f"[[FIGURE:{fig_id}]]"
                    
                    if placeholder in modified_text:
                        replacement = f"\n\n[FIGURE ANALYSIS {fig_id}]: {transcription}\n\n===END FIGURE ANALYSIS===\n\n"
                        # Replace only the first occurrence
                        modified_text = modified_text.replace(placeholder, replacement, 1)
                        injected_figures.add(fig_id)
                        self.logger.debug(f"Injected transcription for {fig_id} into {sec_name}")
                
                # Store if modified
                if modified_text != sec_text:
                    modifications[sec_name] = modified_text
            
            # Apply all modifications
            for sec_name, modified_text in modifications.items():
                sections[sec_name] = modified_text
        
        # 2. Inject tables
        tables = raw_data.get("tables", [])
        injected_tables = set()
        
        # Collect modifications first
        modifications = {}
        for sec_name, sec_text in sections.items():
            modified_text = sec_text
            for table in tables:
                table_id = table.get("id")
                
                # Skip if this table was already injected
                if table_id in injected_tables:
                    continue
                
                placeholder = f"[[TABLE:{table_id}]]"
                
                if placeholder in modified_text:
                    # Format the table using the fetcher's method
                    table_content = self._format_single_table(table)
                    replacement = f"\n\n{table_content}\n\n"
                    # Replace only the first occurrence
                    modified_text = modified_text.replace(placeholder, replacement, 1)
                    injected_tables.add(table_id)
                    self.logger.info(f"Injected table {table_id} into {sec_name}")
            
            # Store if modified
            if modified_text != sec_text:
                modifications[sec_name] = modified_text
        
        # Apply all modifications
        for sec_name, modified_text in modifications.items():
            sections[sec_name] = modified_text
        
        # 3. Update raw_data with modified sections
        raw_data["sections"] = sections
        
        # 4. Format the complete text
        enriched_text = self._format_fulltext_complete(raw_data)
        
        # 5. Clean up any remaining figure/table placeholders that weren't replaced
        # (subsequent references after the first injection)
        for fig in raw_data.get("figures", []):
            fig_id = fig.get("id")
            if fig_id:
                enriched_text = re.sub(rf'\n?\[\[FIGURE:{re.escape(fig_id)}\]\]\n?', '', enriched_text)
        
        for table in raw_data.get("tables", []):
            table_id = table.get("id")
            if table_id:
                enriched_text = re.sub(rf'\n?\[\[TABLE:{re.escape(table_id)}\]\]\n?', '', enriched_text)
        
        return enriched_text

    def _format_figures(self, figures: List[Dict]) -> str:
        """Format figure metadata for inclusion in full-text."""
        if not figures:
            return ""
        
        parts = [f"\n\nFigures ({len(figures)}):"]
        
        for fig in figures:
            parts.append(f"\n{fig['label']}: {fig['caption']}")
        
        return "\n".join(parts)


    def fetch_full_text_context(self, pmids: List[str], return_raw: bool = False, save_xml_dir: str = None) -> Dict[str, Any]:
        """Fetch full text (enrichment) for a specific list of PMIDs.
        
        Args:
            pmids: List of PMIDs to fetch full text for.
            return_raw: If True, returns the raw data dictionary from PMC instead of formatted string.
            save_xml_dir: Optional directory path to save raw XML files for debugging (named pmid_<pmid>.xml)
            
        Returns:
            Dictionary mapping PMID to either full text content string or data dictionary.
        """
        if not pmids:
            return {}
            
        self.logger.info(f"Enriching {len(pmids)} articles with full text...")
        
        # Map PMIDs to PMCIDs
        pmid_to_pmcid = self._fetch_pmc_ids(pmids)
        self.logger.info(f"Found {len(pmid_to_pmcid)} PMCIDs for {len(pmids)} PMIDs")
        
        results = {}
        delimiter = "\n\n===END OF FULL TEXT===\n\n"
        
        for i, pmid in enumerate(pmids):
            # Rate limiting based on API key presence
            if i > 0:
                time.sleep(self._rate_limit_delay)

            pmid = str(pmid)
            pmcid = pmid_to_pmcid.get(pmid)
            
            # Default to None (caller might want to know if fetch failed, or just keep abstract)
            # But here we return the *enriched* content replacing the abstract-only one?
            # Or just the full text part?
            # Typically this replaces the content in the pipeline.
            
            if pmcid:
                try:
                    # Prepare XML save path if requested
                    save_xml_path = None
                    if save_xml_dir:
                        import os
                        os.makedirs(save_xml_dir, exist_ok=True)
                        save_xml_path = os.path.join(save_xml_dir, f"pmid_{pmid}_pmcid_{pmcid}.xml")
                    
                    full_text_data = self._fetch_pmc_fulltext(pmcid, save_xml_path=save_xml_path)
                    if full_text_data:
                        if return_raw:
                            results[pmid] = full_text_data
                        else:
                            # Format complete text without truncation for AI chunker
                            # Gemini Flash has 1M+ token context window, so no need to pre-truncate
                            formatted_text = self._format_fulltext_complete(full_text_data)
                            
                            content = f"PMID: {pmid}\n[FULL-TEXT]\n{formatted_text}{delimiter}"
                            results[pmid] = content
                        
                except Exception as e:
                    self.logger.error(f"Error enriching PMID {pmid}: {e}")
                    
        return results

