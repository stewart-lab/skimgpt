# PDF Parser Test Results

## Result: PDF parser is NOT needed for full text retrieval

### Test Summary

| Check | Count | Result |
|-------|-------|--------|
| Random PMC Open Access articles sampled | 200 | -- |
| OA API reports as PDF/tgz-only (no XML listed) | 200/200 (100%) | Misleading |
| Entrez XML fetch returns full body text | 20/20 (100%) | Full text works |
| Entrez XML fetch restricted by publisher | 0/20 | -- |
| Entrez XML fetch with no body | 0/20 | -- |

### Why a PDF parser is not needed

The PMC Open Access API (oa.fcgi) only lists tgz and pdf as available formats. It does not advertise XML. However, the Entrez efetch API (db=pmc, retmode=xml) successfully returns complete XML with full body text for every article tested, including all sections (Introduction, Methods, Results, Discussion).

The existing pipeline in pubmed_fetcher.py uses Entrez.efetch for XML retrieval, which works correctly regardless of what the OA API format field says.

### Previous reports were misleading

Earlier tests (e.g. pmcid_conclusive_report.txt testing 20,000 PMCIDs) only checked the OA API format field and concluded 100% of articles were PDF-only. That conclusion was wrong. It confused the OA API not listing XML as a download format with XML not being available. The Entrez API serves XML independently of the OA download packages.

### What the OA packages are actually for

The OA tgz packages contain PDFs, high-res figure images, and supplementary files. These are useful for figure extraction (which the project already does via _download_figures_from_package), but are not needed for article text.
