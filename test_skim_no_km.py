import pytest
import skim_no_km as skim_no_km
import openai
import requests
import pandas as pd
from unittest.mock import MagicMock
from skim_no_km import analyze_paper
from unittest.mock import patch
from skim_no_km import find_papers
from multiprocessing import Pool
from skim_no_km import process_row


def test_find_papers_valid_terms():
    c_term = "cancer"
    b_term = "treatment"
    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": []}
        papers = find_papers(c_term, b_term)
        assert papers == [], "Papers should be an empty list for valid terms"


def test_find_papers_invalid_status_code():
    c_term = "invalid_term"
    b_term = "another_invalid_term"
    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.status_code = 404
        papers = find_papers(c_term, b_term)
        assert papers is None, "Papers should be None for invalid status code"


def test_find_papers_with_mocked_requests():
    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": []}
        c_term = "cancer"
        b_term = "treatment"
        papers = find_papers(c_term, b_term)
        assert papers == [], "Papers should be an empty list when mocked"


def test_analyze_paper():
    abstract = "This paper discusses a new treatment for cancer."
    result = analyze_paper(abstract)
    assert "new" in result, "Invalid categorization"


def test_process_row_valid_row():
    row = {"C_term": "cancer", "B_term": "treatment"}
    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [{"abstract": "abstract text"}]
        }
        with patch.object(
            openai.ChatCompletion,
            "create",
            return_value={"choices": [{"message": {"content": "new"}}]},
        ):
            results = process_row(row)
            assert results == ["new"], "Results should contain 'new' for valid row"


def test_analyze_paper_with_mocked_openai():
    abstract = "This paper discusses a new treatment for cancer."
    mock_response = {"choices": [{"message": {"content": "new"}}]}
    with patch.object(openai.ChatCompletion, "create", return_value=mock_response):
        result = analyze_paper(abstract)
        assert result == "new", "Result should be 'new' when mocked"


def test_process_row_invalid_row():
    row = {"C_term": "invalid_term", "B_term": "another_invalid_term"}
    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.status_code = 404
        results = process_row(row)
        assert results == [], "Results should be an empty list for invalid row"


def test_main_execution(tmp_path):
    # Create a temporary CSV file
    file_path = tmp_path / "test_file.csv"
    file_path.write_text("C_term\tB_term\ncancer\ttreatment\n")

    # Create a DataFrame from the temporary file
    df = pd.read_csv(file_path, sep="\t")

    # Mock find_papers to return a sample paper
    with patch("skim_no_km.find_papers", return_value=[{"abstract": "abstract text"}]):
        # Mock analyze_paper to return "new"
        with patch("skim_no_km.analyze_paper", return_value="new"):
            # Run the main execution
            with Pool() as pool:
                analyses = pool.map(process_row, [row for _, row in df.iterrows()])
                combined_analysis = "\n".join(
                    [item for sublist in analyses for item in sublist]
                )

                assert combined_analysis == "new", "Analysis should not be empty"
