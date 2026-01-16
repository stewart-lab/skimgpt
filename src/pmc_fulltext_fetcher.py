"""PMC Full-Text Fetcher - handles fetching and parsing full-text articles from PubMed Central."""

from Bio import Entrez
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import tiktoken


class PMCFullTextFetcher:
    """Fetches and parses full-text articles from PubMed Central (PMC)."""
    
    def __init__(self, logger):
        """Initialize PMC full-text fetcher."""
        self.logger = logger
        self.logger.info("PMCFullTextFetcher initialized")
    
    def fetch_pmc_ids(self, pmids: List[str]) -> Dict[str, str]:
        """Map PMIDs to PMCIDs for articles available in PMC."""
        if not pmids:
            return {}
            
        pmid_to_pmcid = {}
        
        try:
            with Entrez.elink(dbfrom="pubmed", db="pmc", id=pmids) as handle:
                records = Entrez.read(handle)
            
            for record in records:
                if "LinkSetDb" in record and record["LinkSetDb"]:
                    source_pmid = str(record["IdList"][0]) if "IdList" in record and record["IdList"] else None
                    
                    for linksetdb in record["LinkSetDb"]:
                        if linksetdb.get("LinkName") == "pubmed_pmc":
                            if "Link" in linksetdb and linksetdb["Link"]:
                                pmcid = str(linksetdb["Link"][0]["Id"])
                                if source_pmid:
                                    pmid_to_pmcid[source_pmid] = pmcid
                                    self.logger.debug(f"Mapped PMID {source_pmid} -> PMCID {pmcid}")
            
            self.logger.info(f"Found {len(pmid_to_pmcid)} PMC IDs out of {len(pmids)} PMIDs")
            
        except Exception as e:
            self.logger.error(f"Error mapping PMIDs to PMCIDs: {e}")
            
        return pmid_to_pmcid
