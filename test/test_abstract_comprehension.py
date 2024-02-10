import openai
import src.abstract_comprehension as ab
import json
import os
import pandas as pd
import random

TEST_JSON_FILE = "test.json"
TEST_CONFIG_FILE = "test_config.json"


def test_openai_connection():
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": "Test connection to OpenAI."},
            ],
        )
        print("Successfully connected to OpenAI!")
    except Exception as e:
        print(f"Failed to connect to OpenAI. Error: {e}")


def test_process_json():
    # Load test data and config from JSON files
    with open(TEST_JSON_FILE, "r") as f:
        json_data = json.load(f)

    with open(TEST_CONFIG_FILE, "r") as f:
        config = json.load(f)

    # Call the function with test data and config
    result = ab.process_json(json_data, config)

    # Perform assertions to check if the function works as expected
    assert result is not None, "Result should not be None"

    # Check if the keys in the result match the expected diseases and treatments
    assert set(result.keys()) == {"Crohn's disease"}, "Unexpected diseases in result"

    assert (
        result["Crohn's disease"]["FERRIC MALTOL"]["Total Score"] == 6
    ), "Incorrect Total Score for FERRIC MALTOL"
    assert (
        result["Crohn's disease"]["FERRIC MALTOL"]["Average Score"] == 1.5
    ), "Incorrect Average Score for FERRIC MALTOL"
    assert (
        result["Crohn's disease"]["FERRIC MALTOL"]["Count"] == 4
    ), "Incorrect Count for FERRIC MALTOL"


def test_gpt4_leakage(use_breast_cancer_drugs=True):
    # Reading the uploaded file containing the list of drugs
    file_path = "/isiseqruns/jfreeman_tmp_home/skimGPT/input_lists/FDA_approved_ProductsActiveIngredientOnly_DupsRemovedCleanedUp.txt"

    with open(file_path, "r") as file:
        drugs = file.read().splitlines()

    drugs = [drug for drug in drugs if drug.strip()]

    # Use breast cancer drugs if the flag is True
    if use_breast_cancer_drugs:
        drugs = breast_cancer_drugs

    df_data = []
    for _ in range(10):
        random_drug = random.choice(drugs)
        random_pubmed_ids = [random.randint(10000000, 30000000) for _ in range(10)]
        df_data.append(
            {
                "a_count": random.randint(1000, 50000),
                "a_term": "Breast Cancer",
                "ab_count": random.randint(1, 30),
                "ab_pmid_intersection": str(random_pubmed_ids),
                "ab_pred_score": random.uniform(0, 0.02),
                "ab_pvalue": random.uniform(0, 1),
                "ab_sort_ratio": random.uniform(0, 0.01),
                "b_count": random.randint(1000, 10000),
                "b_term": random_drug,
                "total_count": random.randint(30000000, 40000000),
            }
        )

    return pd.DataFrame(df_data)


breast_cancer_drugs = [
    "Abemaciclib",
    "Abraxane (Paclitaxel Albumin-stabilized Nanoparticle Formulation)",
    "Ado-Trastuzumab Emtansine",
    "Afinitor (Everolimus)",
    "Afinitor Disperz (Everolimus)",
    "Alpelisib",
    "Anastrozole",
    "Aredia (Pamidronate Disodium)",
    "Arimidex (Anastrozole)",
    "Aromasin (Exemestane)",
    "Capecitabine",
    "Cyclophosphamide",
    "Docetaxel",
    "Doxorubicin Hydrochloride",
    "Elacestrant Dihydrochloride",
    "Ellence (Epirubicin Hydrochloride)",
    "Enhertu (Fam-Trastuzumab Deruxtecan-nxki)",
    "Epirubicin Hydrochloride",
    "Eribulin Mesylate",
    "Everolimus",
    "Exemestane",
    "5-FU (Fluorouracil Injection)",
    "Fam-Trastuzumab Deruxtecan-nxki",
    "Fareston (Toremifene)",
    "Faslodex (Fulvestrant)",
    "Femara (Letrozole)",
    "Fluorouracil Injection",
    "Fulvestrant",
    "Gemcitabine Hydrochloride",
    "Gemzar (Gemcitabine Hydrochloride)",
    "Goserelin Acetate",
    "Halaven (Eribulin Mesylate)",
    "Herceptin Hylecta (Trastuzumab and Hyaluronidase-oysk)",
    "Herceptin (Trastuzumab)",
    "Ibrance (Palbociclib)",
    "Infugem (Gemcitabine Hydrochloride)",
    "Ixabepilone",
    "Ixempra (Ixabepilone)",
    "Kadcyla (Ado-Trastuzumab Emtansine)",
    "Keytruda (Pembrolizumab)",
    "Kisqali (Ribociclib)",
    "Lapatinib Ditosylate",
    "Letrozole",
    "Lynparza (Olaparib)",
    "Margenza (Margetuximab-cmkb)",
    "Margetuximab-cmkb",
    "Megestrol Acetate",
    "Methotrexate Sodium",
    "Neratinib Maleate",
    "Nerlynx (Neratinib Maleate)",
    "Olaparib",
    "Orserdu (Elacestrant Dihydrochloride)",
    "Paclitaxel",
    "Paclitaxel Albumin-stabilized Nanoparticle Formulation",
    "Palbociclib",
    "Pamidronate Disodium",
    "Pembrolizumab",
    "Perjeta (Pertuzumab)",
    "Pertuzumab",
    "Pertuzumab, Trastuzumab, and Hyaluronidase-zzxf",
    "Phesgo (Pertuzumab, Trastuzumab, and Hyaluronidase-zzxf)",
    "Piqray (Alpelisib)",
    "Ribociclib",
    "Sacituzumab Govitecan-hziy",
    "Soltamox (Tamoxifen Citrate)",
    "Talazoparib Tosylate",
    "Talzenna (Talazoparib Tosylate)",
    "Tamoxifen Citrate",
    "Taxotere (Docetaxel)",
    "Tecentriq (Atezolizumab)",
    "Tepadina (Thiotepa)",
    "Thiotepa",
    "Toremifene",
    "Trastuzumab",
    "Trastuzumab and Hyaluronidase-oysk",
    "Trexall (Methotrexate Sodium)",
    "Trodelvy (Sacituzumab Govitecan-hziy)",
    "Tucatinib",
    "Tukysa (Tucatinib)",
    "Tykerb (Lapatinib Ditosylate)",
    "Verzenio (Abemaciclib)",
    "Vinblastine Sulfate",
    "Xeloda (Capecitabine)",
    "Zol",
]
