from Bio import Entrez
from src.utils import Config, extract_pmid
import time
import re
from typing import List, Dict, Any
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import tarfile
import tempfile
import shutil
import io

# Configure tiktoken cache directory before import to avoid permission issues
# in shared computing environments
tiktoken_cache_dir = Path.home() / ".cache" / "tiktoken"
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

import tiktoken
import warnings

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
        self._setup_entrez()
        self._rate_limit_delay = 0.1 if api_key else 0.34         # Rate limit: 10 req/s with API key, 3 req/s without
        self.logger.info(f"PubMedFetcher initialized (rate limit: {1/self._rate_limit_delay:.1f} req/s)")

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

    def _fetch_batch(self, batch: List[str]) -> Dict[str, Any]:
        """Fetch a single batch of parameters (abstracts only)."""
        for attempt in range(1, self.max_retries + 1):
            try:
                # Fetch basic metadata from PubMed
                with Entrez.efetch(db="pubmed", id=batch, retmode="xml", rettype="abstract") as efetch:
                    output = Entrez.read(efetch)

                returned_pmids = []
                returned_contents = []
                delimiter = "\n\n===END OF ABSTRACT===\n\n"
                skipped_min_wc_pmids = []

                for paper in output.get("PubmedArticle", []):
                    pmid = str(paper["MedlineCitation"]["PMID"])
                    article = paper["MedlineCitation"]["Article"]
                    pub_year = self._extract_publication_year(paper)
                    
                    title = article.get("ArticleTitle", "No title available")
                    abstract_list = article.get("Abstract", {}).get("AbstractText", ["No abstract available"])
                    if isinstance(abstract_list, list):
                        abstract_text = " ".join(abstract_list)
                    else:
                        abstract_text = str(abstract_list)

                    if len(abstract_text.split()) >= self.config.min_word_count:
                        returned_pmids.append(pmid)
                        self.pmid_years[pmid] = int(pub_year)
                        content = f"PMID: {pmid}\n[ABSTRACT]\nTitle: {title}\nAbstract: {abstract_text}{delimiter}"
                        returned_contents.append(content)
                    else:
                        skipped_min_wc_pmids.append(pmid)

                # Summary log for MIN_WORD_COUNT exclusions
                if skipped_min_wc_pmids:
                    self.logger.debug(
                        f"Excluded {len(skipped_min_wc_pmids)} PMIDs due to MIN_WORD_COUNT="
                        f"{self.config.min_word_count}. Example: {skipped_min_wc_pmids[:5]}"
                    )

                return {
                    "pmids": returned_pmids,
                    "contents": returned_contents
                }

            except Exception as e:
                self.logger.error(f"Attempt {attempt} - Error fetching batch: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Failed to fetch batch after {self.max_retries} attempts.")
                    return {}

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

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """Fetch abstracts for a list of PMIDs."""
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
            time.sleep(self._rate_limit_delay)  # Rate limiting

        if not abstract_dict:
            self.logger.error("No abstracts fetched successfully.")
            return {}

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

        self.logger.debug("Original order (most cited):")
        for entry in original_entries[:3]:  # Show first 3 for brevity
            pmid = extract_pmid(entry)
            if pmid:
                year = self.pmid_years.get(pmid, 0)
                self.logger.debug(f"PMID: {pmid}, Year: {year}")

        # Create year-sorted version
        entries_with_years = []
        for entry in entries:
            pmid = extract_pmid(entry)
            if pmid:
                year = self.pmid_years.get(pmid, 0)
                entries_with_years.append((year, entry))

        # Sort by year (newest first)
        year_sorted_entries = [entry for _, entry in sorted(entries_with_years, key=lambda x: x[0], reverse=True)]
        
        self.logger.debug("\nYear-sorted order (most recent):")
        pmids_logged = []
        for entry in year_sorted_entries[:15]:  # Show first 3 for brevity
            pmid = extract_pmid(entry)
            if pmid:
                year = self.pmid_years.get(pmid, 0)
                if pmid not in pmids_logged:
                    self.logger.debug(f"PMID: {pmid}, Year: {year}")
                pmids_logged.append(pmid)

        # Calculate interleaving ratio with proper zero handling
        if top_n_most_recent > 0:
            ratio = top_n_most_cited / top_n_most_recent
        elif top_n_most_cited > 0:
            ratio = float('inf')  # All cited, no recent
        else:
            ratio = 1.0  # Default to equal if both are 0
        self.logger.debug(f"Interleaving ratio (cited:recent) = {ratio if ratio != float('inf') else 'inf'}")

        # Interleave based on ratio
        result = []
        cited_idx = recent_idx = 0
        used_pmids = set()  # Track which PMIDs we've already added
        
        def get_pmid(entry):
            """Helper function to extract PMID from an entry"""
            return extract_pmid(entry) or None
        
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

        while (cited_idx < len(original_entries) or recent_idx < len(year_sorted_entries)) and (n is None or len(result) < n):
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
                # Calculate expected_recent properly, handling edge cases
                if ratio > 0:
                    expected_recent = round(cited_idx / ratio)
                else:
                    # If ratio is 0, use all available recent entries
                    expected_recent = len(year_sorted_entries)
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
        
        # Log summary after loop completes
        self.logger.debug(f"Interleaving complete: cited_idx={cited_idx}, recent_idx={recent_idx}, unique_pmids={len(used_pmids)}")
            
        if not result:
            return ""
        
        final_text = "===END OF ABSTRACT===\n\n".join(result) + "===END OF ABSTRACT===\n\n"
        
        # Add debug logging to count actual abstracts
        abstract_count = len(result)
        self.logger.debug(f"Returning {abstract_count} abstracts from interleave_abstracts")
        
        return final_text

    def optimize_text_length(self, text: str | list, max_tokens: int = 110000000, encoding_name: str = "cl100k_base", num_intersections: int = 1) -> str:

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
        optimized_entries = []
        current_tokens = 0
        
        for entry in entries:
            entry_tokens = len(encoding.encode(entry))
            if current_tokens + entry_tokens <= tokens_per_intersection:
                optimized_entries.append(entry)
                current_tokens += entry_tokens
            else:
                break
        
        self.logger.debug(f"Optimized to {len(optimized_entries)}/{len(entries)} entries, {current_tokens} tokens")
            
        if not optimized_entries:
            return ""
        
        final_text = "===END OF ABSTRACT===\n\n".join(optimized_entries) + "===END OF ABSTRACT===\n\n"
        
        # Add debug logging
        abstract_count = len(optimized_entries)
        self.logger.debug(f"Returning {abstract_count} abstracts from optimize_text_length")
        
        return final_text 