#!/usr/bin/env python3
"""
Quick test: check whether PMC Open Access articles provide XML or only PDF/tgz.

Samples random PMCIDs, hits the OA API, and checks what format is available.
Also tries Entrez XML fetch to see if body text is actually returned.
"""

import random
import time
import json
import xml.etree.ElementTree as ET
from collections import Counter

import requests
from Bio import Entrez

Entrez.email = "test@example.com"

OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
SAMPLE_SIZE = 200
BATCH_SIZE = 50
XML_SPOT_CHECK = 20


def get_random_pmcids(n: int) -> list:
    """Fetch random PMCIDs by searching PMC for recent articles."""
    pmcids = set()
    retmax = min(n * 5, 10000)
    handle = Entrez.esearch(db="pmc", term="open access[filter]", retmax=retmax, sort="relevance")
    result = Entrez.read(handle)
    handle.close()
    ids = result.get("IdList", [])
    random.shuffle(ids)
    for pid in ids[:n]:
        pmcids.add(f"PMC{pid}")
    return list(pmcids)[:n]


def check_oa_api(pmcids: list) -> dict:
    """Check OA API for each PMCID; return format info."""
    results = {"pdf_only": 0, "xml_available": 0, "both": 0, "neither": 0, "total": 0, "details": []}
    for i in range(0, len(pmcids), BATCH_SIZE):
        batch = pmcids[i:i + BATCH_SIZE]
        for pmcid in batch:
            try:
                resp = requests.get(OA_API, params={"id": pmcid}, timeout=15)
                has_pdf = "format=\"tgz\"" in resp.text or "format=\"pdf\"" in resp.text
                has_xml = "format=\"xml\"" in resp.text
                results["total"] += 1
                if has_pdf and has_xml:
                    results["both"] += 1
                    results["details"].append((pmcid, "both"))
                elif has_pdf:
                    results["pdf_only"] += 1
                    results["details"].append((pmcid, "pdf_only"))
                elif has_xml:
                    results["xml_available"] += 1
                    results["details"].append((pmcid, "xml_only"))
                else:
                    results["neither"] += 1
                    results["details"].append((pmcid, "neither"))
            except Exception as e:
                results["details"].append((pmcid, f"error: {e}"))
            time.sleep(0.35)
        print(f"  OA API checked {min(i + BATCH_SIZE, len(pmcids))}/{len(pmcids)}")
    return results


def spot_check_xml(pmcids: list, n: int) -> dict:
    """Try fetching XML via Entrez for a subset and check if body text exists."""
    sample = random.sample(pmcids, min(n, len(pmcids)))
    results = {"has_body": 0, "no_body": 0, "error": 0, "restricted": 0, "total": len(sample), "details": []}
    for pmcid in sample:
        raw_id = pmcid.replace("PMC", "")
        try:
            handle = Entrez.efetch(db="pmc", id=raw_id, retmode="xml")
            xml_bytes = handle.read()
            handle.close()

            if b"publisher of this article does not allow downloading" in xml_bytes:
                results["restricted"] += 1
                results["details"].append((pmcid, "restricted"))
                continue

            root = ET.fromstring(xml_bytes)
            body = root.find(".//body")
            if body is not None and len(list(body)) > 0:
                results["has_body"] += 1
                results["details"].append((pmcid, "has_body"))
            else:
                results["no_body"] += 1
                results["details"].append((pmcid, "no_body"))
        except Exception as e:
            results["error"] += 1
            results["details"].append((pmcid, f"error: {e}"))
        time.sleep(0.4)
        print(f"  XML spot-check {len(results['details'])}/{results['total']}")
    return results


def main():
    print(f"=== PDF Availability Test (n={SAMPLE_SIZE}) ===\n")

    print("1. Fetching random PMCIDs...")
    pmcids = get_random_pmcids(SAMPLE_SIZE)
    print(f"   Got {len(pmcids)} PMCIDs\n")

    print("2. Checking OA API for format availability...")
    oa = check_oa_api(pmcids)
    print(f"\n   OA API Results ({oa['total']} checked):")
    print(f"     PDF/tgz only:  {oa['pdf_only']} ({100*oa['pdf_only']/max(oa['total'],1):.1f}%)")
    print(f"     XML available:  {oa['xml_available']} ({100*oa['xml_available']/max(oa['total'],1):.1f}%)")
    print(f"     Both:           {oa['both']} ({100*oa['both']/max(oa['total'],1):.1f}%)")
    print(f"     Neither:        {oa['neither']} ({100*oa['neither']/max(oa['total'],1):.1f}%)\n")

    pdf_only_pmcids = [d[0] for d in oa["details"] if d[1] == "pdf_only"]
    xml_pmcids = [d[0] for d in oa["details"] if d[1] in ("xml_only", "both")]

    print(f"3. Spot-checking Entrez XML fetch for {XML_SPOT_CHECK} PDF-only articles...")
    if pdf_only_pmcids:
        xml_check = spot_check_xml(pdf_only_pmcids, XML_SPOT_CHECK)
        print(f"\n   Entrez XML Spot-Check ({xml_check['total']} articles):")
        print(f"     Has body text:  {xml_check['has_body']}")
        print(f"     No body text:   {xml_check['no_body']}")
        print(f"     Restricted:     {xml_check['restricted']}")
        print(f"     Errors:         {xml_check['error']}")
    else:
        xml_check = None
        print("   No PDF-only articles to spot-check.")

    print("\n" + "=" * 60)
    needs_parser = oa["pdf_only"] > 0
    pct = 100 * oa["pdf_only"] / max(oa["total"], 1)
    if needs_parser:
        print(f"RESULT: PDF parser IS needed.")
        print(f"  {oa['pdf_only']}/{oa['total']} ({pct:.1f}%) of OA articles are PDF/tgz-only.")
        if xml_check and xml_check["has_body"] > 0:
            print(f"  NOTE: {xml_check['has_body']}/{xml_check['total']} PDF-only articles DID return XML body via Entrez.")
            print(f"  The OA API may undercount XML availability.")
    else:
        print(f"RESULT: PDF parser is NOT needed for this sample.")
        print(f"  All {oa['total']} OA articles had XML available.")
    print("=" * 60)

    report = {
        "sample_size": SAMPLE_SIZE,
        "oa_api": {k: v for k, v in oa.items() if k != "details"},
        "xml_spot_check": {k: v for k, v in (xml_check or {}).items() if k != "details"},
        "needs_pdf_parser": needs_parser,
        "pdf_only_pct": round(pct, 1),
    }
    with open("tests/pdf_test_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRaw results saved to tests/pdf_test_results.json")


if __name__ == "__main__":
    main()
