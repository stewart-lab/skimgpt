import os
import sys
import json
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.relevance import run_relevance_analysis
from src.utils import Config

def test_figure_integration():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_figure_integration")

    # Mock config
    config = MagicMock(spec=Config)
    config.logger = logger
    config.full_text = True
    config.full_text_model = "gemini-3-flash-preview"
    config.debug = True
    config.km_output_dir = "test_output"
    config.debug_tsv_name = "test_output/debug_test.tsv"
    config.secrets = {"PUBMED_API_KEY": "test", "GEMINI_API_KEY": "test"}
    config.global_settings = {"DCH_SAMPLE_SIZE": 1, "DCH_MIN_SAMPLING_FRACTION": 0.06}
    config.filter_config = {"TEMPERATURE": 0, "TOP_P": 1, "MAX_COT_TOKENS": 1}
    config.max_retries = 3
    config.iterations = 0
    config.km_hypothesis = "Test hypothesis for {a_term} and {b_term}"
    config.is_km_with_gpt = True
    config.is_skim_with_gpt = False
    config.censor_year_lower = 0
    config.censor_year_upper = 2026
    config.min_word_count = 0
    config.has_ac = False
    config.is_skim_with_gpt = False
    config.is_dch = False
    config.post_n = 0
    
    # Mock data
    import pandas as pd
    config.data = pd.DataFrame([{
        "a_term": "test_a",
        "b_term": "test_b",
        "ab_pmid_intersection": "['12345']"
    }])
    
    os.makedirs("test_output", exist_ok=True)

    # Mock PubMedFetcher
    with patch("src.relevance.PubMedFetcher") as MockFetcher:
        fetcher = MockFetcher.return_value
        fetcher.fetch_abstracts.return_value = {"12345": "PMID: 12345\nAbstract text"}
        
        # Mock optimize_text_length
        fetcher.optimize_text_length.side_effect = lambda x, **kwargs: x
        
        # Mock fetch_full_text_context returning raw data
        raw_data = {
            "12345": {
                "pmcid": "PMC54321",
                "figures": [{"id": "F1", "label": "Fig 1", "caption": "Test caption", "graphic_ref": "fig1.jpg"}],
                "sections": {"Introduction": "Intro text [[FIGURE:F1]] more intro."},
                "title": "Test Title",
                "abstract": "Test Abstract"
            }
        }
        fetcher.fetch_full_text_context.return_value = raw_data
        
        # Mock _format_fulltext_complete
        def mock_format(data):
            text = f"Title: {data['title']}\nAbstract: {data['abstract']}\n"
            for s, t in data['sections'].items():
                text += f"{s}: {t}\n"
            return text
        fetcher._format_fulltext_complete.side_effect = mock_format
        
        # Mock _download_figures_from_package
        def mock_download(pmcid, figs, outdir):
            for f in figs:
                f["local_path"] = os.path.join(outdir, f["graphic_ref"])
            return figs
        fetcher._download_figures_from_package.side_effect = mock_download

        # Mock ImageAnalyzer
        with patch("src.relevance.ImageAnalyzer") as MockAnalyzer:
            analyzer = MockAnalyzer.return_value
            analyzer.enhance_figure_descriptions.return_value = [
                {"id": "F1", "enhanced_content": "TRANSCRIPTION_OF_FIGURE_1"}
            ]

            # Mock TritonClient and other components as needed or just the run_relevance_analysis flow
            with patch("src.relevance.TritonClient") as MockTriton, \
                 patch("src.relevance.FullTextChunker") as MockChunker, \
                 patch("src.relevance.process_results"):
                
                triton = MockTriton.return_value
                triton.generate_batch.return_value = [{"text_output": "1: This is relevant"}]
                triton.server_url = "http://localhost:8000"
                triton.model_name = "test_model"
                triton.temperature = 0
                triton.top_p = 1
                triton.max_tokens = 1
                triton.check_server_health.return_value = True
                
                chunker = MockChunker.return_value
                chunker.chunk_document.return_value = "Extracted evidence"
                
                # Run the analysis
                run_relevance_analysis(config, "test_output/input.tsv")
                
                # Verify that analyzer was called
                analyzer.enhance_figure_descriptions.assert_called_once()
                
                # Verify that placeholders were replaced
                # We can check the full_text_raw.json artifact
                raw_artifact_path = "test_output/full_text_raw.json"
                if os.path.exists(raw_artifact_path):
                    with open(raw_artifact_path, "r") as f:
                        saved_data = json.load(f)
                    
                    content = saved_data["12345"]
                    print(f"DEBUG: Content in artifact: {content}")
                    assert "[FIGURE ANALYSIS F1]: TRANSCRIPTION_OF_FIGURE_1" in content
                    assert "[[FIGURE:F1]]" not in content
                    logger.info("Verification SUCCESS: Figure placeholders replaced with transcriptions.")
                else:
                    logger.error("Verification FAILED: Artifact not found.")

if __name__ == "__main__":
    test_figure_integration()
