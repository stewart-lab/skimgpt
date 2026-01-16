# Visualizing Enriched Full Text

This guide explains how to use the `tests/test_enriched_fulltext.py` script to fetch and enrich full text (including figure analysis) from PubMed/PMC, and how to visualize the results using the `json_viewer.html` tool.

## Overview

The workflow consists of two main steps:
1. **Generate Data**: Run the Python script to fetch the article, analyze figures, and save the enriched text to a JSON file.
2. **Visualize Data**: Use the web-based viewer to render the JSON output in a readable, formatted way with figure reinjection.

## Step 1: Generate Enriched Text Data

The `tests/test_enriched_fulltext.py` script is used to fetch data for a specific PMID.

### Prerequisites
- Ensure your `secrets.json` is configured with `PUBMED_API_KEY` and `GEMINI_API_KEY`.
- Ensure you are in the project root directory.

### Usage
Run the script with the `PMID` and the `--output` argument to save the results.

```bash
# Syntax
python tests/test_enriched_fulltext.py <PMID> --output <OUTPUT_FILE.json> [--skip-figures] [--debug]

# Example (Fetch PMID 34567890 and save to output.json)
python tests/test_enriched_fulltext.py 34567890 --output output.json
```

**Arguments:**
- `PMID`: The PubMed ID of the article you want to process.
- `--output`: Path to the file where the JSON result will be saved. **Required for the viewer.**
- `--skip-figures`: (Optional) Skips the figure analysis step.
- `--debug`: (Optional) Enables verbose logging.

## Step 2: View the Results

The `json_viewer.html` is a standalone HTML file that renders the JSON output.

### Opening the Viewer

You can open the viewer in your web browser in two ways:

**Option A: Using a Local Server (Recommended)**
If you are running a Python HTTP server (e.g., for development):
1. Ensure the server is running. It is best to specify the project directory explicitly:
   ```bash
   python3 -m http.server 8000 --directory /path/to/skimgpt
   ```
2. Open your browser to: `http://localhost:8000/json_viewer.html`

**Option B: Opening the File Directly**
1. Locate `json_viewer.html` in your file explorer.
2. Double-click to open it in your default web browser.

### Loading and Navigating

1. **Load the JSON**: Click the **"Load JSON File"** button in the sidebar and select the JSON file you generated in Step 1 **you must download the JSON file from the server** (e.g., `output.json`).
2. **Select Content**:
   - The sidebar will populate with keys from the JSON file. 
   - **Click on `enriched_text`** in the sidebar to view the full enriched article.
3. **View the Article**:
   - The main content area will display the formatted article.
   - **Figures**: Figure analyses are reinjected into the text and will appear as distinct "Figure Analysis" cards containing the transcribed description and insights.
   - **Structure**: Section headers (Introduction, Methods, etc.) are detected and formatted as Markdown headers.
