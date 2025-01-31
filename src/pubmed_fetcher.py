from Bio import Entrez
from src.utils import setup_logger
import time
import re
from typing import List, Dict, Any

logger = setup_logger()

class PubMedFetcher:
    def __init__(self, email: str, api_key: str, max_retries: int = 10, backoff_factor: float = 0.5):
        """Initialize PubMed fetcher with credentials and retry settings."""
        self.email = email
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.pmid_years = {}  # Store PMID -> Year mapping
        self._setup_entrez()
        logger.info("PubMedFetcher initialized")

    def _setup_entrez(self):
        """Configure Entrez with credentials."""
        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def validate_pmids(self, pmids: List[Any]) -> List[str]:
        """Validate PMIDs to ensure they are numeric."""
        valid_pmids = []
        for pmid in pmids:
            pmid_str = str(pmid)
            if pmid_str.isdigit() and len(pmid_str) > 0:
                valid_pmids.append(pmid_str)
            else:
                logger.warning(f"Invalid PMID detected and skipped: {pmid}")
        return valid_pmids

    def _batch_pmids(self, pmids: List[str], batch_size: int = 200) -> List[List[str]]:
        """Split PMIDs into batches."""
        return [pmids[i : i + batch_size] for i in range(0, len(pmids), batch_size)]

    def _extract_publication_year(self, paper: Dict) -> str:
        """Extract publication year from PubMed article data."""
        article = paper["MedlineCitation"]["Article"]
        pub_year = None
        
        # 1. Try PubDate from Article
        pub_date = article.get("ArticleDate", [])
        if pub_date and "Year" in pub_date[0]:
            pub_year = pub_date[0]["Year"]
        
        # 2. Try Journal PubDate
        if not pub_year:
            journal_info = article.get("Journal", {})
            pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
            if "Year" in pub_date:
                pub_year = pub_date["Year"]
            elif "MedlineDate" in pub_date:
                medline_date = pub_date["MedlineDate"]
                year_match = re.search(r'\d{4}', medline_date)
                if year_match:
                    pub_year = year_match.group(0)
        
        # 3. Try MedlineCitation Date
        if not pub_year:
            medline_date = paper["MedlineCitation"].get("DateCompleted", {})
            if "Year" in medline_date:
                pub_year = medline_date["Year"]
        
        # Default to "0000" if no year found
        if not pub_year:
            pmid = str(paper["MedlineCitation"]["PMID"])
            logger.warning(f"No publication year found for PMID {pmid}")
            pub_year = "0000"
            
        return pub_year

    def _fetch_batch(self, batch: List[str]) -> Dict[str, Any]:
        """Fetch a single batch of PMIDs with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                with Entrez.efetch(db="pubmed", id=batch, retmode="xml", rettype="abstract") as efetch:
                    output = Entrez.read(efetch)

                returned_pmids = []
                returned_contents = []
                delimiter = "\n\n===END OF ABSTRACT===\n\n"

                for paper in output.get("PubmedArticle", []):
                    pmid = str(paper["MedlineCitation"]["PMID"])
                    article = paper["MedlineCitation"]["Article"]
                    pub_year = self._extract_publication_year(paper)
                    
                    title = article.get("ArticleTitle", "No title available")
                    abstract_text = " ".join(
                        article.get("Abstract", {}).get("AbstractText", ["No abstract available"])
                    )
                    
                    if len(abstract_text.split()) >= 50:
                        returned_pmids.append(pmid)
                        self.pmid_years[pmid] = int(pub_year)  # Store year in instance
                        content = f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract_text}{delimiter}"
                        returned_contents.append(content)

                return {
                    "pmids": returned_pmids,
                    "contents": returned_contents
                }

            except Exception as e:
                logger.error(f"Attempt {attempt} - Error fetching abstracts for batch: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached for batch. Skipping.")
                    return {}

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
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
                abstract_dict.update(dict(zip(batch_result["pmids"], batch_result["contents"])))
            time.sleep(0.34)  # Rate limiting

        if not abstract_dict:
            logger.error("No abstracts fetched successfully.")
        else:
            logger.info(f"Successfully fetched abstracts for {len(abstract_dict)} PMIDs.")

        return abstract_dict

    def interleave_abstracts(self, text: str, n: int = None, top_n_most_cited: int = 0, top_n_most_recent: int = 0) -> str:
        """
        Interleave abstracts based on citation count and recency with configurable ratios.
        If top_n_most_cited is 0, only returns most recent abstracts.
        If top_n_most_recent is 0, only returns most cited abstracts.
        Otherwise, interleaves based on the ratio of cited:recent.
        """
        if not isinstance(text, str) or text == "[]":
            return ""

        # Split and process entries
        entries = text.split("===END OF ABSTRACT===")
        entries = [e.strip() for e in entries if e.strip()]

        if not entries:
            return ""

        # Keep original order (most cited)
        original_entries = entries.copy()

        logger.debug("Original order (most cited):")
        for entry in original_entries[:3]:  # Show first 3 for brevity
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                year = self.pmid_years.get(pmid, 0)
                logger.debug(f"PMID: {pmid}, Year: {year}")

        # Create year-sorted version
        entries_with_years = []
        for entry in entries:
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                year = self.pmid_years.get(pmid, 0)
                entries_with_years.append((year, entry))

        # Sort by year (newest first)
        year_sorted_entries = [entry for _, entry in sorted(entries_with_years, key=lambda x: x[0], reverse=True)]
        
        logger.debug("\nYear-sorted order (most recent):")
        pmids_logged = []
        for entry in year_sorted_entries[:15]:  # Show first 3 for brevity
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                year = self.pmid_years.get(pmid, 0)
                if pmid not in pmids_logged:
                    logger.debug(f"PMID: {pmid}, Year: {year}")
                pmids_logged.append(pmid)
        # If n is specified, limit both lists before interleaving
        if n is not None:
            original_entries = original_entries[:n]
            year_sorted_entries = year_sorted_entries[:n]
            logger.debug(f"\nLimiting to top {n} entries from each list")
        # Handle special cases
        if top_n_most_cited == 0 and top_n_most_recent > 0:
            # Only return most recent
            result = year_sorted_entries
            logger.debug(f"Returning only {n} most recent entries")
        
        elif top_n_most_recent == 0 and top_n_most_cited > 0:
            # Only return most cited
            result = original_entries
            logger.debug(f"Returning only {n} most cited entries")
        
        else:
            # Calculate interleaving ratio
            ratio = top_n_most_cited / top_n_most_recent if top_n_most_recent > 0 else float('inf')
            logger.debug(f"Interleaving ratio (cited:recent) = {ratio:.2f}")

            # Interleave based on ratio
            result = []
            cited_idx = recent_idx = 0
            used_pmids = set()  # Track which PMIDs we've already added
            
            def get_pmid(entry):
                """Helper function to extract PMID from an entry"""
                pmid_match = re.search(r"PMID: (\d+)", entry)
                return pmid_match.group(1) if pmid_match else None
            
            def add_entry(entry):
                """Helper function to add entry if not already present"""
                pmid = get_pmid(entry)
                if pmid and pmid not in used_pmids:
                    result.append(entry)
                    used_pmids.add(pmid)
                    return True
                return False
        
            while cited_idx < len(original_entries) or recent_idx < len(year_sorted_entries):
                made_progress = False
                
                # Handle ratio > 1 (more cited than recent)
                if ratio > 1 and cited_idx < len(original_entries):
                    expected_cited = round(recent_idx * ratio)
                    # Add enough to catch up to the expected count
                    while cited_idx < min(expected_cited, len(original_entries)):
                        if add_entry(original_entries[cited_idx]):
                            made_progress = True
                        cited_idx += 1
                    # Add one recent entry after the batch of cited entries
                    if recent_idx < len(year_sorted_entries):
                        if add_entry(year_sorted_entries[recent_idx]):
                            made_progress = True
                        recent_idx += 1
                    # If we can't add a recent entry but still have cited entries, add one more cited
                    elif cited_idx < len(original_entries):
                        if add_entry(original_entries[cited_idx]):
                            made_progress = True
                        cited_idx += 1
                
                # Handle ratio < 1 (more recent than cited)
                elif ratio < 1 and recent_idx < len(year_sorted_entries):
                    expected_recent = round(cited_idx / ratio)
                    while recent_idx < min(expected_recent, len(year_sorted_entries)):
                        if add_entry(year_sorted_entries[recent_idx]):
                            made_progress = True
                        recent_idx += 1
                    # Add one cited entry after the batch of recent entries
                    if cited_idx < len(original_entries):
                        if add_entry(original_entries[cited_idx]):
                            made_progress = True
                        cited_idx += 1
                    # If we can't add a cited entry but still have recent entries, add one more recent
                    elif recent_idx < len(year_sorted_entries):
                        if add_entry(year_sorted_entries[recent_idx]):
                            made_progress = True
                        recent_idx += 1

                # Handle ratio == 1 (equal numbers of cited and recent)
                elif ratio == 1:
                    if cited_idx < len(original_entries):
                        if add_entry(original_entries[cited_idx]):
                            made_progress = True
                        cited_idx += 1
                    if recent_idx < len(year_sorted_entries):
                        if add_entry(year_sorted_entries[recent_idx]):
                            made_progress = True
                        recent_idx += 1

                # If we didn't make any progress in this iteration, break to avoid infinite loop
                if not made_progress:
                    break

                # Break if we've processed all entries from both lists
                if cited_idx >= len(original_entries) and recent_idx >= len(year_sorted_entries):
                    break

            logger.debug(f"Cited entries used: {cited_idx}, Recent entries used: {recent_idx}")
            logger.debug(f"Unique PMIDs in result: {len(used_pmids)}")
        logger.debug(f"Final interleaved list contains {len(result)} entries")
            
        return "===END OF ABSTRACT===\n\n".join(result) + "===END OF ABSTRACT===\n\n"

    def optimize_text_length(self, text: str | list, max_tokens: int = 110000, encoding_name: str = "cl100k_base", num_intersections: int = 1) -> str:
        """
        Optimize text length to fit within token limit while preserving complete abstracts.
        
        Args:
            text: Input text containing abstracts (string or list)
            max_tokens: Maximum total tokens allowed
            encoding_name: Name of the tokenizer encoding to use
            num_intersections: Number of intersection sets to distribute tokens across
        """
        # Handle list input
        if isinstance(text, list):
            text = "\n===END OF ABSTRACT===\n".join(text) if text else ""
        
        if not text or max_tokens <= 0:
            return ""
        
        try:
            import tiktoken
            encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.error("tiktoken not installed. Required for token counting.")
            return text
        
        # Adjust max_tokens based on number of intersections
        tokens_per_intersection = max_tokens // num_intersections
        
        entries = text.split("===END OF ABSTRACT===")
        entries = [e.strip() for e in entries if e.strip()]
        
        optimized_entries = []
        current_tokens = 0
        
        for entry in entries:
            entry_tokens = len(encoding.encode(entry))
            if current_tokens + entry_tokens <= tokens_per_intersection:
                optimized_entries.append(entry)
                current_tokens += entry_tokens
            else:
                break
            
        if not optimized_entries:
            return ""
        
        return "===END OF ABSTRACT===\n\n".join(optimized_entries) + "===END OF ABSTRACT===\n\n" 