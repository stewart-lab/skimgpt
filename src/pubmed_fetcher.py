from Bio import Entrez
from src.utils import Config
import time
import re
from typing import List, Dict, Any
import tiktoken
import warnings
import math
from datetime import datetime

# Silence the specific Bio.Entrez.Parser warning about DTD files
warnings.filterwarnings("ignore", message="Failed to save .* at .*")

class PubMedFetcher:
    def __init__(self, config: Config, email: str, api_key: str, max_retries: int = 10, backoff_factor: float = 0.5):
        """Initialize PubMed fetcher with credentials and retry settings."""
        self.config = config  # Store Config instance
        self.logger = config.logger
        self.email = email
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.pmid_years = {}  # Store PMID -> Year mapping
        self.pmid_citations = {}  # Store PMID -> Citation count mapping
        self.pmid_pub_years = {}  # Store PMID -> Publication year mapping
        self._setup_entrez()
        self.logger.info("PubMedFetcher initialized")  # Use config's logger

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
                self.logger.warning(f"Invalid PMID detected and skipped: {pmid}")
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
            self.logger.warning(f"No publication year found for PMID {pmid}")
            pub_year = "0000"
            
        return pub_year

    def fetch_citation_counts(self, pmids: List[str]) -> Dict[str, int]:
        """
        Fetch citation counts for a list of PMIDs using PubMed's elink database.
        """
        pmids = self.validate_pmids(pmids)
        if not pmids:
            self.logger.error("No valid PMIDs to fetch citations for.")
            return {}

        batches = self._batch_pmids(pmids, batch_size=200)
        citation_counts = {}

        for batch in batches:
            batch_result = self._fetch_citation_batch(batch)
            citation_counts.update(batch_result)
            time.sleep(0.34)  # Rate limiting

        # Store results in instance variable
        self.pmid_citations.update(citation_counts)
        
        self.logger.info(f"Successfully fetched citation counts for {len(citation_counts)} PMIDs.")
        return citation_counts

    def _fetch_citation_batch(self, batch: List[str]) -> Dict[str, int]:
        """Fetch citation counts for a single batch of PMIDs."""
        for attempt in range(1, self.max_retries + 1):
            try:
                # Use elink to find papers that cite these PMIDs
                with Entrez.elink(
                    dbfrom="pubmed", 
                    db="pubmed", 
                    id=batch, 
                    linkname="pubmed_pubmed_citedin"
                ) as elink:
                    output = Entrez.read(elink)

                citation_counts = {}
                
                for link_set in output:
                    pmid = str(link_set["IdList"][0]) if link_set.get("IdList") else None
                    if pmid:
                        # Count the citing papers
                        citing_pmids = []
                        for link_set_db in link_set.get("LinkSetDb", []):
                            if link_set_db.get("LinkName") == "pubmed_pubmed_citedin":
                                citing_pmids = link_set_db.get("Link", [])
                                break
                        
                        citation_count = len(citing_pmids)
                        citation_counts[pmid] = citation_count
                        
                        self.logger.debug(f"PMID {pmid}: {citation_count} citations")

                # Ensure all batch PMIDs have entries (even if 0 citations)
                for pmid in batch:
                    if pmid not in citation_counts:
                        citation_counts[pmid] = 0

                return citation_counts

            except Exception as e:
                self.logger.error(f"Attempt {attempt} - Error fetching citations for batch: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("Max retries reached for citation batch. Returning empty dict.")
                    return {pmid: 0 for pmid in batch}  # Return 0 citations for all PMIDs in failed batch

    def calculate_citation_age_scores(self, pmids: List[str], current_year: int = None) -> Dict[str, float]:
        """
        Calculate citation + age normalizer scores using the formula: score = log10(cites+1) + λ/age
        Based on the flowchart specification.
        
        Args:
            pmids: List of PMIDs to calculate scores for
            current_year: Current year for age calculation (defaults to current year)
            
        Returns:
            Dictionary mapping PMID to normalized score
        """
        if current_year is None:
            current_year = datetime.now().year
            
        scores = {}
        citation_lambda = self.config.citation_lambda
        
        for pmid in pmids:
            pmid_str = str(pmid)
            
            # Get citation count (default to 0 if not found)
            citation_count = self.pmid_citations.get(pmid_str, 0)
            
            # Get publication year (default to current year if not found)
            pub_year = self.pmid_years.get(pmid_str, current_year)
            if pub_year == 0:  # Handle cases where year extraction failed
                pub_year = current_year
                
            # Calculate age (minimum age of 1 to avoid division by zero)
            age = max(1, current_year - pub_year)
            
            # Apply the formula: score = log10(cites+1) + λ/age
            citation_score = math.log10(citation_count + 1)
            age_score = citation_lambda / age
            total_score = citation_score + age_score
            
            scores[pmid_str] = total_score
            
            self.logger.debug(f"PMID {pmid_str}: {citation_count} cites, {age} years old, "
                            f"score = log10({citation_count}+1) + {citation_lambda}/{age} = {total_score:.4f}")
        
        return scores

    def normalize_scores_by_pool(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores by pool size: AvgSupport = Σ score / pool size
        As specified in the flowchart.
        
        Args:
            scores: Dictionary of PMID -> score mappings
            
        Returns:
            Dictionary with normalized scores (average support)
        """
        if not scores:
            return {}
            
        # Calculate pool size and sum
        pool_size = len(scores)
        score_sum = sum(scores.values())
        avg_support = score_sum / pool_size
        
        self.logger.info(f"Pool normalization: {pool_size} papers, "
                        f"total score = {score_sum:.4f}, "
                        f"AvgSupport = {avg_support:.4f}")
        
    
        normalized_scores = {pmid: avg_support for pmid in scores.keys()}
        
        return normalized_scores

    def check_pool_balance(self, pool_a_pmids: List[str], pool_b_pmids: List[str], 
                          min_pool_size: int = 10, max_imbalance_percent: float = 90.0) -> Dict[str, Any]:
        """
        Check if two pools are balanced according to the flowchart criteria:
        nA, nB ≥ 10 & |nA-nB| ≤ 90%
        
        Args:
            pool_a_pmids: PMIDs in pool A
            pool_b_pmids: PMIDs in pool B
            min_pool_size: Minimum required pool size (default: 10)
            max_imbalance_percent: Maximum allowed imbalance percentage (default: 90%)
            
        Returns:
            Dictionary with balance analysis results
        """
        nA = len(pool_a_pmids)
        nB = len(pool_b_pmids)
        
        # Calculate imbalance percentage
        if nA + nB == 0:
            imbalance_percent = 0
        else:
            imbalance_percent = abs(nA - nB) / max(nA, nB) * 100
        
        # Check balance criteria
        size_sufficient = nA >= min_pool_size and nB >= min_pool_size
        balance_acceptable = imbalance_percent <= max_imbalance_percent
        is_balanced = size_sufficient and balance_acceptable
        
        # Determine allocation strategy
        if is_balanced:
            allocation_strategy = "equal"  # Equal allocation: nA == nB per run
        else:
            allocation_strategy = "ratio"  # Ratio allocation: α_min = 30%
        
        result = {
            "is_balanced": is_balanced,
            "pool_a_size": nA,
            "pool_b_size": nB,
            "size_sufficient": size_sufficient,
            "imbalance_percent": imbalance_percent,
            "balance_acceptable": balance_acceptable,
            "allocation_strategy": allocation_strategy,
            "criteria": {
                "min_pool_size": min_pool_size,
                "max_imbalance_percent": max_imbalance_percent
            }
        }
        
        self.logger.info(f"Pool Balance Analysis:")
        self.logger.info(f"  Pool A size: {nA}, Pool B size: {nB}")
        self.logger.info(f"  Size sufficient: {size_sufficient} (≥{min_pool_size} each)")
        self.logger.info(f"  Imbalance: {imbalance_percent:.1f}% (≤{max_imbalance_percent}%)")
        self.logger.info(f"  Balanced: {is_balanced}")
        self.logger.info(f"  Strategy: {allocation_strategy} allocation")
        
        return result

    def allocate_abstracts_equal(self, pool_a_pmids: List[str], pool_b_pmids: List[str], 
                               per_pool_count: int) -> Dict[str, List[str]]:
        """
        Equal allocation: nA == nB per run
        Used when pools are balanced.
        
        Args:
            pool_a_pmids: PMIDs in pool A
            pool_b_pmids: PMIDs in pool B  
            per_pool_count: Number of abstracts to select from each pool
            
        Returns:
            Dictionary with selected PMIDs from each pool
        """
        # Get individual citation + age scores for ranking
        pool_a_scores = self.calculate_citation_age_scores(pool_a_pmids)
        pool_b_scores = self.calculate_citation_age_scores(pool_b_pmids)
        
        # Sort by individual scores (highest first)
        pool_a_sorted = sorted(pool_a_pmids, 
                              key=lambda pmid: pool_a_scores.get(str(pmid), 0), 
                              reverse=True)
        pool_b_sorted = sorted(pool_b_pmids, 
                              key=lambda pmid: pool_b_scores.get(str(pmid), 0), 
                              reverse=True)
        
        # Select top N from each pool
        selected_a = pool_a_sorted[:per_pool_count]
        selected_b = pool_b_sorted[:per_pool_count]
        
        self.logger.info(f"Equal Allocation:")
        self.logger.info(f"  Selected from Pool A: {len(selected_a)}/{len(pool_a_pmids)}")
        self.logger.info(f"  Selected from Pool B: {len(selected_b)}/{len(pool_b_pmids)}")
        
        return {
            "pool_a_selected": selected_a,
            "pool_b_selected": selected_b,
            "allocation_type": "equal"
        }

    def allocate_abstracts_ratio(self, pool_a_pmids: List[str], pool_b_pmids: List[str], 
                               total_count: int, alpha_min: float = 0.30) -> Dict[str, List[str]]:
        """
        Ratio allocation: α_min = 30%
        Used when pools are skewed.
        
        Args:
            pool_a_pmids: PMIDs in pool A
            pool_b_pmids: PMIDs in pool B
            total_count: Total number of abstracts to select
            alpha_min: Minimum allocation ratio for smaller pool (default: 30%)
            
        Returns:
            Dictionary with selected PMIDs from each pool
        """
        nA = len(pool_a_pmids)
        nB = len(pool_b_pmids)
        
        # Determine which pool is smaller
        if nA <= nB:
            smaller_pool_pmids = pool_a_pmids
            larger_pool_pmids = pool_b_pmids
            smaller_is_a = True
        else:
            smaller_pool_pmids = pool_b_pmids
            larger_pool_pmids = pool_a_pmids
            smaller_is_a = False
        
        # Calculate allocation counts
        smaller_count = max(1, int(total_count * alpha_min))
        larger_count = total_count - smaller_count
        
        # Get individual citation + age scores for ranking
        smaller_scores = self.calculate_citation_age_scores(smaller_pool_pmids)
        larger_scores = self.calculate_citation_age_scores(larger_pool_pmids)
        
        # Sort by individual scores (highest first)
        smaller_sorted = sorted(smaller_pool_pmids, 
                               key=lambda pmid: smaller_scores.get(str(pmid), 0), 
                               reverse=True)
        larger_sorted = sorted(larger_pool_pmids, 
                              key=lambda pmid: larger_scores.get(str(pmid), 0), 
                              reverse=True)
        
        # Select based on allocation
        selected_smaller = smaller_sorted[:smaller_count]
        selected_larger = larger_sorted[:larger_count]
        
        # Map back to A and B
        if smaller_is_a:
            selected_a = selected_smaller
            selected_b = selected_larger
        else:
            selected_a = selected_larger
            selected_b = selected_smaller
        
        actual_alpha = len(selected_smaller) / (len(selected_smaller) + len(selected_larger))
        
        self.logger.info(f"Ratio Allocation (α_min = {alpha_min}):")
        self.logger.info(f"  Smaller pool: {len(selected_smaller)}/{len(smaller_pool_pmids)} ({actual_alpha:.1%})")
        self.logger.info(f"  Larger pool: {len(selected_larger)}/{len(larger_pool_pmids)} ({1-actual_alpha:.1%})")
        self.logger.info(f"  Selected from Pool A: {len(selected_a)}/{nA}")
        self.logger.info(f"  Selected from Pool B: {len(selected_b)}/{nB}")
        
        return {
            "pool_a_selected": selected_a,
            "pool_b_selected": selected_b,
            "allocation_type": "ratio",
            "alpha_actual": actual_alpha,
            "smaller_is_a": smaller_is_a
        }

    def apply_citation_age_normalization(self, pmids: List[str], pool_name: str = "default") -> Dict[str, float]:
        """
        Apply the complete citation + age normalization workflow as specified in the flowchart:
        1. Calculate citation + age scores: score = log10(cites+1) + λ/age
        2. Normalize by pool size: AvgSupport = Σ score / pool size
        
        Args:
            pmids: List of PMIDs to process
            pool_name: Name of the pool for logging purposes
            
        Returns:
            Dictionary of normalized scores
        """
        if not pmids:
            self.logger.warning(f"No PMIDs provided for {pool_name} normalization")
            return {}
            
        # Ensure we have citation counts and years
        pmids_str = [str(pmid) for pmid in pmids]
        missing_citations = [pmid for pmid in pmids_str if pmid not in self.pmid_citations]
        if missing_citations:
            self.logger.info(f"Fetching missing citation counts for {len(missing_citations)} PMIDs in {pool_name}")
            self.fetch_citation_counts(missing_citations)
        
        # Calculate citation + age scores
        citation_age_scores = self.calculate_citation_age_scores(pmids_str)
        
        # Normalize by pool size
        normalized_scores = self.normalize_scores_by_pool(citation_age_scores)
        
        # Log summary for this pool
        if citation_age_scores:
            min_score = min(citation_age_scores.values())
            max_score = max(citation_age_scores.values())
            avg_score = list(normalized_scores.values())[0]  # All values are the same after normalization
            
            self.logger.info(f"{pool_name} pool normalization complete:")
            self.logger.info(f"  - Pool size: {len(pmids_str)} abstracts")
            self.logger.info(f"  - Score range: {min_score:.4f} to {max_score:.4f}")
            self.logger.info(f"  - Normalized AvgSupport: {avg_score:.4f}")
        
        return normalized_scores

    def allocate_abstracts_balanced_workflow(self, pool_a_pmids: List[str], pool_b_pmids: List[str], 
                                           total_abstracts: int = 20) -> Dict[str, Any]:
        """
        Complete balanced allocation workflow following the flowchart:
        1. Check if pools are balanced (nA, nB ≥ 10 & |nA-nB| ≤ 90%)
        2. If balanced: Equal allocation (nA == nB per run)
        3. If skewed: Ratio allocation (α_min = 30%)
        
        Args:
            pool_a_pmids: PMIDs in pool A
            pool_b_pmids: PMIDs in pool B
            total_abstracts: Total number of abstracts to allocate
            
        Returns:
            Dictionary with complete allocation results
        """
        self.logger.info("=== BALANCED ALLOCATION WORKFLOW ===")
        
        # Step 1: Check pool balance
        balance_analysis = self.check_pool_balance(pool_a_pmids, pool_b_pmids)
        
        # Step 2: Apply appropriate allocation strategy
        if balance_analysis["is_balanced"]:
            # Balanced: Equal allocation
            per_pool_count = total_abstracts // 2
            allocation_result = self.allocate_abstracts_equal(
                pool_a_pmids, pool_b_pmids, per_pool_count
            )
        else:
            # Skewed: Ratio allocation  
            allocation_result = self.allocate_abstracts_ratio(
                pool_a_pmids, pool_b_pmids, total_abstracts, alpha_min=0.30
            )
        
        # Step 3: Calculate pool-level normalized scores for comparison
        pool_a_normalized = self.apply_citation_age_normalization(
            allocation_result["pool_a_selected"], "Pool_A"
        )
        pool_b_normalized = self.apply_citation_age_normalization(
            allocation_result["pool_b_selected"], "Pool_B"
        )
        
        # Step 4: Combine results
        workflow_result = {
            "balance_analysis": balance_analysis,
            "allocation_result": allocation_result,
            "pool_a_normalized_scores": pool_a_normalized,
            "pool_b_normalized_scores": pool_b_normalized,
            "summary": {
                "is_balanced": balance_analysis["is_balanced"],
                "strategy_used": allocation_result["allocation_type"],
                "pool_a_count": len(allocation_result["pool_a_selected"]),
                "pool_b_count": len(allocation_result["pool_b_selected"]),
                "pool_a_avg_support": list(pool_a_normalized.values())[0] if pool_a_normalized else 0,
                "pool_b_avg_support": list(pool_b_normalized.values())[0] if pool_b_normalized else 0
            }
        }
        
        # Step 5: Log summary
        summary = workflow_result["summary"]
        self.logger.info(f"=== WORKFLOW SUMMARY ===")
        self.logger.info(f"Strategy: {summary['strategy_used']} allocation")
        self.logger.info(f"Pool A: {summary['pool_a_count']} abstracts, AvgSupport = {summary['pool_a_avg_support']:.4f}")
        self.logger.info(f"Pool B: {summary['pool_b_count']} abstracts, AvgSupport = {summary['pool_b_avg_support']:.4f}")
        
        return workflow_result

    def demonstrate_citation_normalization(self, pmids: List[str]) -> None:
        """
        Demonstrate the citation + age normalization workflow with detailed logging.
        This method shows how the flowchart formulas are applied step by step.
        """
        if not pmids:
            self.logger.error("No PMIDs provided for demonstration")
            return
            
        self.logger.info("=== Citation + Age Normalization Demonstration ===")
        self.logger.info(f"Processing {len(pmids)} PMIDs...")
        
        # Step 1: Fetch abstracts and citations
        self.logger.info("Step 1: Fetching abstracts and citations...")
        abstracts = self.fetch_abstracts(pmids, fetch_citations=True)
        
        # Step 2: Apply citation + age normalization
        self.logger.info("Step 2: Applying citation + age normalization...")
        normalized_scores = self.apply_citation_age_normalization(pmids, "demonstration")
        
        # Step 3: Show detailed results
        self.logger.info("Step 3: Detailed results by PMID:")
        current_year = datetime.now().year
        
        for pmid in pmids[:5]:  # Show first 5 for brevity
            pmid_str = str(pmid)
            if pmid_str in self.pmid_years and pmid_str in self.pmid_citations:
                pub_year = self.pmid_years[pmid_str]
                citations = self.pmid_citations[pmid_str]
                age = max(1, current_year - pub_year)
                
                # Calculate individual score components
                citation_score = math.log10(citations + 1)
                age_score = self.config.citation_lambda / age
                total_score = citation_score + age_score
                normalized_score = normalized_scores.get(pmid_str, 0)
                
                self.logger.info(f"  PMID {pmid_str}:")
                self.logger.info(f"    Published: {pub_year} (age: {age} years)")
                self.logger.info(f"    Citations: {citations}")
                self.logger.info(f"    Citation score: log10({citations}+1) = {citation_score:.4f}")
                self.logger.info(f"    Age score: {self.config.citation_lambda}/{age} = {age_score:.4f}")
                self.logger.info(f"    Total score: {total_score:.4f}")
                self.logger.info(f"    Normalized (AvgSupport): {normalized_score:.4f}")
                
        self.logger.info("=== Demonstration Complete ===")

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
                    
                    if len(abstract_text.split()) >= self.config.min_word_count:
                        returned_pmids.append(pmid)
                        self.pmid_years[pmid] = int(pub_year)  # Store year in instance
                        content = f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract_text}{delimiter}"
                        returned_contents.append(content)

                return {
                    "pmids": returned_pmids,
                    "contents": returned_contents
                }

            except Exception as e:
                self.logger.error(f"Attempt {attempt} - Error fetching abstracts for batch: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("Max retries reached for batch. Skipping.")
                    return {}

    def fetch_abstracts(self, pmids: List[str], fetch_citations: bool = True) -> Dict[str, str]:
        """
        Fetch abstracts for a list of PMIDs.
        
        Args:
            pmids: List of PMIDs to fetch abstracts for
            fetch_citations: Whether to also fetch citation counts (default: True)
        """
        pmids = self.validate_pmids(pmids)
        if not pmids:
            self.logger.error("No valid PMIDs to fetch.")
            return {}

        batches = self._batch_pmids(pmids)
        abstract_dict = {}

        for batch in batches:
            batch_result = self._fetch_batch(batch)
            if batch_result:
                abstract_dict.update(dict(zip(batch_result["pmids"], batch_result["contents"])))
            time.sleep(0.34)  # Rate limiting

        if not abstract_dict:
            self.logger.error("No abstracts fetched successfully.")
            return {}
<<<<<<< Updated upstream

        else: 
            self.logger.info(f"Successfully fetched abstracts for {len(abstract_dict)} PMIDs.")

        # Filter abstracts by publication year bounds
        lower = self.config.censor_year_lower
        upper = self.config.censor_year_upper
        filtered_dict = {}
        for pmid, content in abstract_dict.items():
            year = self.pmid_years.get(pmid, 0)
            if lower <= year <= upper:
                filtered_dict[pmid] = content
        self.logger.info(f"Filtered {len(filtered_dict)}/{len(abstract_dict)} abstracts by publication year bounds ({lower}-{upper})")

        return filtered_dict
=======
        
        self.logger.info(f"Successfully fetched abstracts for {len(abstract_dict)} PMIDs.")
        
        # Fetch citation counts if requested
        if fetch_citations:
            self.logger.info("Fetching citation counts for abstracts...")
            fetched_pmids = list(abstract_dict.keys())
            self.fetch_citation_counts(fetched_pmids)
            
        return abstract_dict
>>>>>>> Stashed changes

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

    
        entries_with_citations = []
        for entry in entries:
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                citation_count = self.pmid_citations.get(pmid, 0)
                entries_with_citations.append((citation_count, entry))

        # Sort by citation count (highest first)
        citation_sorted_entries = [entry for _, entry in sorted(entries_with_citations, key=lambda x: x[0], reverse=True)]

        self.logger.debug("Citation-sorted order (most cited first):")
        for entry in citation_sorted_entries[:3]:  # Show first 3 for brevity
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                year = self.pmid_years.get(pmid, 0)
                citations = self.pmid_citations.get(pmid, 0)
                self.logger.debug(f"PMID: {pmid}, Year: {year}, Citations: {citations}")

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
        
        self.logger.debug("\nYear-sorted order (most recent):")
        pmids_logged = []
        for entry in year_sorted_entries[:15]:  # Show first 3 for brevity
            pmid_match = re.search(r"PMID: (\d+)", entry)
            if pmid_match:
                pmid = pmid_match.group(1)
                year = self.pmid_years.get(pmid, 0)
                if pmid not in pmids_logged:
                    self.logger.debug(f"PMID: {pmid}, Year: {year}")
                pmids_logged.append(pmid)

        # Calculate interleaving ratio
        ratio = top_n_most_cited / top_n_most_recent if top_n_most_recent > 0 else float(top_n_most_cited)
        self.logger.debug(f"Interleaving ratio (cited:recent) = {ratio:.2f}")

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
            # Check if we've reached the limit before adding
            if n is not None and len(result) >= n:
                return False
            
            pmid = get_pmid(entry)
            if pmid and pmid not in used_pmids:
                result.append(entry)
                used_pmids.add(pmid)
                return True
            return False

        while (cited_idx < len(citation_sorted_entries) or recent_idx < len(year_sorted_entries)) and (n is None or len(result) < n):
            made_progress = False
            
            # Handle ratio > 1 (more cited than recent)
            if ratio > 1 and cited_idx < len(citation_sorted_entries):
                expected_cited = round(recent_idx * ratio)
                # Add enough to catch up to the expected count
                while cited_idx < min(expected_cited, len(citation_sorted_entries)):
                    if add_entry(citation_sorted_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
                # Add one recent entry after the batch of cited entries
                if recent_idx < len(year_sorted_entries):
                    if add_entry(year_sorted_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1
                # If we can't add a recent entry but still have cited entries, add one more cited
                elif cited_idx < len(citation_sorted_entries):
                    if add_entry(citation_sorted_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
            
            # Handle ratio < 1 (more recent than cited)
            elif ratio < 1 and recent_idx < len(year_sorted_entries):
                expected_recent = round(cited_idx / ratio) if ratio > 0 else len(year_sorted_entries) # or maybe inf
                while recent_idx < min(expected_recent, len(year_sorted_entries)):
                    if add_entry(year_sorted_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1
                # Add one cited entry after the batch of recent entries
                if cited_idx < len(citation_sorted_entries):
                    if add_entry(citation_sorted_entries[cited_idx]):
                        made_progress = True
                    cited_idx += 1
                # If we can't add a cited entry but still have recent entries, add one more recent
                elif recent_idx < len(year_sorted_entries):
                    if add_entry(year_sorted_entries[recent_idx]):
                        made_progress = True
                    recent_idx += 1

            # Handle ratio == 1 (equal numbers of cited and recent)
            elif ratio == 1:
                if cited_idx < len(citation_sorted_entries):
                    if add_entry(citation_sorted_entries[cited_idx]):
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
            if cited_idx >= len(citation_sorted_entries) and recent_idx >= len(year_sorted_entries):
                break
            self.logger.debug(f"cited_idx: {cited_idx}, recent_idx: {recent_idx}, n: {n}")
            
            self.logger.debug(f"Cited entries used: {cited_idx}, Recent entries used: {recent_idx}")
            self.logger.debug(f"Unique PMIDs in result: {len(used_pmids)}")
            self.logger.debug(f"Final interleaved list contains {len(result)} entries")
            
        if not result:
            return ""
        
        final_text = "===END OF ABSTRACT===\n\n".join(result) + "===END OF ABSTRACT===\n\n"
        
        # Add debug logging to count actual abstracts
        abstract_count = len(result)
        self.logger.debug(f"Returning {abstract_count} abstracts from interleave_abstracts")
        
        return final_text

    def optimize_text_length(self, text: str | list, max_tokens: int = 110000, encoding_name: str = "cl100k_base", num_intersections: int = 1) -> str:

      # Handle list input
        if isinstance(text, list):
            text = "\n===END OF ABSTRACT===\n".join(text) if text else ""
        
        if not text or max_tokens <= 0:
            return ""
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            self.logger.error("tiktoken not installed. Required for token counting.")
            return text
        
        # Adjust max_tokens based on number of intersections
        tokens_per_intersection = max_tokens // num_intersections
        
        entries = text.split("===END OF ABSTRACT===")
        entries = [e.strip() for e in entries if e.strip()]
        #self.logger.debug(f"Entries: {entries}")
        optimized_entries = []
        current_tokens = 0
        
        for entry in entries:
            entry_tokens = len(encoding.encode(entry))
            self.logger.debug(f"Entry tokens: {entry_tokens}")
            self.logger.debug(f"Current tokens: {current_tokens}")
            self.logger.debug(f"entry: {entry}")
            if current_tokens + entry_tokens <= tokens_per_intersection:
                optimized_entries.append(entry)
                current_tokens += entry_tokens
                self.logger.debug(f"adding entry: {entry}")
            else:
                self.logger.debug(f"breaking at entry: {entry}")
                break
            
        if not optimized_entries:
            return ""
        
        final_text = "===END OF ABSTRACT===\n\n".join(optimized_entries) + "===END OF ABSTRACT===\n\n"
        
        # Add debug logging
        abstract_count = len(optimized_entries)
        self.logger.debug(f"Returning {abstract_count} abstracts from optimize_text_length")
        
        return final_text 