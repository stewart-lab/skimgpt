import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from pubmed_fetcher import PubMedFetcher

class TestPubMedFetcherNoTruncation(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.logger = MagicMock()
        self.fetcher = PubMedFetcher(self.mock_config, "test@example.com", "api_key")

    def test_extract_tables_no_truncation(self):
        # Create a table with 30 rows
        root = ET.Element("article")
        table_wrap = ET.SubElement(root, "table-wrap")
        table = ET.SubElement(table_wrap, "table")
        
        for i in range(30):
            tr = ET.SubElement(table, "tr")
            td = ET.SubElement(tr, "td")
            td.text = f"Row {i+1}"

        # Extract tables
        tables = self.fetcher._extract_tables(root)
        
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["data"]), 30)
        self.assertFalse(tables[0].get("truncated", False))
        print(f"Verified: Extracted {len(tables[0]['data'])} rows (expected 30)")

    @patch("src.pubmed_fetcher.time.sleep")
    @patch("src.pubmed_fetcher.PubMedFetcher._fetch_pmc_ids")
    @patch("src.pubmed_fetcher.PubMedFetcher._fetch_pmc_fulltext")
    def test_rate_limiting_sleep(self, mock_fetch_fulltext, mock_fetch_ids, mock_sleep):
        # Setup mocks
        mock_fetch_ids.return_value = {"1": "PMC1", "2": "PMC2", "3": "PMC3"}
        mock_fetch_fulltext.return_value = {"title": "Test", "pmcid": "PMC"}
        
        pmids = ["1", "2", "3"]
        self.fetcher.fetch_full_text_context(pmids)
        
        # Should sleep for 2nd and 3rd request (indices 1 and 2)
        # Total calls to sleep should be len(pmids) - 1 = 2
        self.assertEqual(mock_sleep.call_count, 2)
        print(f"Verified: time.sleep called {mock_sleep.call_count} times for {len(pmids)} PMIDs")

if __name__ == "__main__":
    unittest.main()
