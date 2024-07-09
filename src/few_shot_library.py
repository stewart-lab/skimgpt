def diabetes_example_1():
    """
    Example 1: Hypothesis: Metformin may be a useful treatment for treating the underlying causes of Diabetes
    """
    reasoning = (
        "Hypothesis: Metformin may be a useful treatment for treating the underlying causes of Diabetes - Score: 3 - Reasoning: PMID 11832527 states that metformin reduces the incidence of type 2 diabetes. Elevated plasma glucose levels, being overweight, and having a sedentary lifestyle are reversible risk factors for diabetes. Metformin reduced the incidence of diabetes in high-risk prediabetic patients by 31%. Score: 3.\n"
        "PMID 9742977 states that metformin slows the progression of diabetes by controlling blood glucose levels and reduces the incidence diabetes-related risks such as death in diabetic patients. Score: 3.\n"
        "PMID 18784090 states that benefits from metformin treatment on diabetic patients were long-lasting. Score: 3.\n"
        "PMID 11602624 investigates the mechanism by which metformin aids glucose control, concluding that metformin activates AMPK, which leads to reduced ACC activity, increased fatty acid oxidation, suppressed expression of lipogenic enzymes, and suppressed expression of SREBP-1. Score: 3.\n\n"
    )
    return reasoning


def diabetes_example_2():
    """
    Example 2: Hypothesis: Ezetimibe may be a useful treatment for treating the underlying causes of Diabetes
    """
    reasoning = (
        "Hypothesis: Ezetimibe may be a useful treatment for treating the underlying causes of Diabetes - Score: -1 - Reasoning: PMID 23956253 discusses familial hypercholesterolaemia in relation to coronary heart disease (CHD). Low-density lipoprotein (LDL) levels are discussed; however, there is no obvious connection between diabetes, LDL, and ezetimibe presented. Score: 0.\n"
        "PMID 23444397 implies that ezetimibe helps manage LDL levels. Diabetes is mentioned in passing. Score: 0.\n"
        "PMID 28886926 implies that ezetimibe is useful for lowering LDL levels to lower atherosclerotic cardiovascular disease (ASCVD) risk. Diabetes is mentioned in passing. Score: 0.\n"
        "PMID 27701660 found that LDL-lowering genetic variants were positively associated with a higher risk of type 2 diabetes. This implies that lowering LDL levels could possibly increase diabetes risk. “potential adverse effects of LDL-C-lowering therapy” is mentioned at the end. Score: -1.\n"
        "PMID 19095139 examines the effects of lowering LDL-C with statins and ezetimibe in diabetic patients with regards to atherosclerosis. This article is not relevant to the hypothesis. Score: 0.\n"
        "PMID 21095263 states that lowering LDL levels reduces the incidence of atherosclerotic events. This article is not relevant to the hypothesis. Score: 0.\n\n"
    )
    return reasoning


def diabetes_example_3():
    """
    Example 3: Hypothesis: bumetanide may be a useful treatment for treating the underlying causes of Diabetes
    """
    reasoning = (
        "Hypothesis: bumetanide may be a useful treatment for treating the underlying causes of Diabetes - Score: 1 - Reasoning: PMID 32410463 indicates that empagliflozin synergized with bumetanide to help flush sodium and glucose via a diuretic effect. This is mostly discussed in relation to heart failure, but lowering blood glucose helps prevent diabetes. Score: 1.\n"
        "PMID 29440005 discusses treatment with dapagliflozin to lower blood glucose levels in diabetic patients via inhibition of the sodium-glucose-linked transporter 2. Bumetanide is hypothesized to synergize with dapagliflozin to help flush sodium and glucose from the blood. Score: 1.\n"
        "PMID 9185518 describes the BSC2 Na-K-Cl cotransporter as “bumetanide-sensitive”. The experiments described in this article suggest that BSC2 is regulated by inflammatory cytokines and fluid mechanical stimuli, implying a role for BSC2 in “vascular homeostasis and inflammation.” This is related to diabetes because blood glucose levels affect osmotic conditions.\n"
        "PMID 12904328 states that diabetes is associated with water and sodium losses, and the authors hypothesize that the kidney compensates for this by upregulating sodium and water transporter proteins. Experiments show that the bumetanide-sensitive Na-K-2Cl cotransporter (NKCC2) correlate with blood glucose levels. It's unclear from this text how bumetanide would affect blood glucose levels. Score: 0\n"
    )
    return reasoning


def diabetes_example_4():
    """
    Example 4: Hypothesis: sodium may be a useful treatment for treating the underlying causes of Diabetes
    """
    reasoning = "Hypothesis: sodium may be a useful treatment for treating the underlying causes of Diabetes - Score: 0 - Reasoning: The texts provided do not discuss sodium as a treatment, but rather in the context of the sodium-glucose cotransporter 2 (SGLT2). Several articles describe inhibiting SGLT2 in an effort to improve cardiovascular health in diabetic patients. Score: 0.\n\n"
    return reasoning


def breast_cancer_example_1():
    """
    Example 1: Hypothesis: Sirolimus may be a useful treatment for treating the underlying causes of Breast Cancer
    """
    reasoning = (
        "Hypothesis: Sirolimus may be a useful treatment for treating the underlying causes of breast cancer - "
        "Score: 2 - Reasoning: PMID 19225562 suggests that sirolimus may be a good drug candidate for breast cancer patients who are African-American, "
        "though the evidence is fairly weak. Score: 1.\n"
        "PMID 16404634 finds that sirolimus (an immunosuppressant) is an inhibitor of BCRP. The abstract states that some immunosuppressants have significant "
        "drug-drug interactions with chemotherapeutic drugs that are BCRP substrates, increasing bioavailability of the chemotherapeutic drug. Presumably, an "
        "increase of bioavailability of these drugs would potentially improve therapeutic outcomes but may increase toxicity. Score: 2.\n"
        "PMID 32335491 describes sirolimus as an mTOR inhibitor. The abstract describes a retrospective study in which patients with HR-positive breast cancer "
        "were treated with sirolimus plus endocrine therapy. Sirolimus was found to be approximately as effective as everolimus in a small sample size (36 patients), "
        "though patients with PI3K/Akt/mTOR pathway alterations seem to have responded better than those without those alterations. Score: 2.\n"
        "PMID 31205412 describes a pathway analysis of several drugs, one of which is sirolimus, in the MCF7 breast cancer cell line. The study seems relatively weak "
        "and not very relevant to the hypothesis. Score: 0.\n"
        "PMID 37904250 describes sirolimus (rapamycin) as an mTOR inhibitor. In this study, the authors treated breast tissue with sirolimus in order to study its "
        "effects on mammary stem cell activity and growth. Sirolimus was found to decrease mammary stem cell activity; specifically, it decreased their “self-renewal "
        "capacity”. Biomarkers associated with early breast cancer progression were reportedly decreased. Score: 2."
    )
    return reasoning


def breast_cancer_example_2():
    """
    Example 2: Hypothesis: Idelalisib may be a useful treatment for treating the underlying causes of Breast Cancer
    """
    reasoning = (
        "Hypothesis: Idelalisib may be a useful treatment for treating the underlying causes of Breast Cancer - Score: 1 - "
        "Reasoning: PMID 26242379 describes idelalisib as a phosphatidylinositol 3-kinase-δ inhibitor approved for some subtypes of leukaemia and lymphoma. "
        "The only mention of breast cancer in this abstract is that idelalisib does not inhibit breast cancer resistance protein (BCRP). Score: 0.\n"
        "PMID 25760671 is similar to the previous abstract in that idelalisib is reported as a PI3Kδ inhibitor and breast cancer was only mentioned in passing "
        "with regards to BCRP being quantified. Score: 0.\n"
        "PMID 33774181 discusses PI3K inhibitors with regards to advances in the treatment of various cancers. Alpelisib is mentioned as a PI 3-kinase-α and "
        "breast cancer treatment. Idelalisib is mentioned in the treatment of different cancers. Score: 0.\n"
        "PMID 37422644 describes an attempt to create a predictive model of breast cancer from gene signatures, dividing breast cancer patients into high- and "
        "low-risk groups. The study mentions idelalisib as a potential treatment for low-risk breast cancer, though not much evidence is described in the abstract. "
        "Score: 1.\n"
        "PMID 34547385 describes idelalisib as an FDA-approved treatment for ER+/HER- advanced metastatic breast cancer because of its role as a PI3K inhibitor, "
        "though this does not seem to be reported elsewhere. Score: 2."
    )
    return reasoning


def raynauds_disease_example_1():
    """
    Hypothesis: Leflunomide may be a useful treatment for treating the underlying causes of Raynaud's disease
    """
    reasoning = (
        "Hypothesis: Leflunomide may be a useful treatment for treating the underlying causes of Raynaud's disease - Score: -1 - "
        "Reasoning: PMID 12634229 mentions Raynaud's phenomenon as related to systemic sclerosis. None of the treatments mentioned (including leflunomide) "
        "were effective in treating Raynaud's. Score: -1.\n"
        "PMID 16769661 describes a study with two patients with Raynaud's phenomenon caused by anti-Jo-1 antibodies that were successfully treated with leflunomide. "
        "Because of the low number of patients and the highly specific cause of Raynaud's, this study is somewhat weak. Score: 1.\n"
        "PMID 28336160 discusses Raynaud's as positively associated with leflunomide treatment, indicating Raynaud's as a possible adverse reaction to leflunomide. Score: -1.\n"
        "PMID 30640987 describes a study in which leflunomide was combined with prednisone and a Chinese herbal remedy. The study does not seem well-conducted; "
        "however, they report lower Raynaud's symptoms for both treatment groups. Score: 1.\n"
        "PMID 27987518 describes a study in which two patients present with Raynaud's phenomenon as part of a larger study on rare ocular diseases. Leflunomide is described "
        "as an immunosuppressant. The abstract does not discuss leflunomide's effect on Raynaud's. Score: 0."
    )
    return reasoning


def raynauds_disease_example_2():
    """
    Hypothesis: DEXAMETHASONE may be a useful treatment for treating the underlying causes of Raynaud's disease
    """
    reasoning = (
        "Hypothesis: DEXAMETHASONE may be a useful treatment for treating the underlying causes of Raynaud's disease - Score: 1 - "
        "Reasoning: PMID 7839076 describes no change in Raynaud's phenomenon with dexamethasone treatment in 35 systemic sclerosis patients. Score: 1.\n"
        "PMID 21207214 reports on a patient with systemic sclerosis and multiple myeloma, with Raynaud's as a symptom. She was treated with cyclophosphamide, "
        "thalidomide, and dexamethasone. This treatment was effective in treating the sclerosis and myeloma. Given that only one patient was studied here, "
        "and dexamethasone was used in combination with 2 other drugs, the evidence is fairly weak. Score: 1.\n"
        "PMID 8014051 reports on a patient with Raynaud's on exposure to the cold. Dexamethasone treatment was effective. Given that only one patient was studied here, "
        "the evidence is fairly weak. Score: 1.\n"
        "PMID 31305439 reports on a patient with a rare case of type I monoclonal cryoglobulinemia, with Raynaud's as one of the symptoms. Cyclophosphamide, "
        "thalidomide, and dexamethasone was used to treat the patient successfully. Score: 1.\n"
        "PMID 35585632 reports on a patient with Guillain-Barre syndrome, with Raynaud's presumably a symptom. Treatment with lenalidomide and dexamethasone improved "
        "the Raynaud's phenomenon. Score: 1."
    )
    return reasoning


def heart_failure_example_1():
    """
    Hypothesis: LIXISENATIDE may be a useful treatment for treating the underlying causes of Heart Failure
    """
    reasoning = (
        "Hypothesis: LIXISENATIDE may be a useful treatment for treating the underlying causes of Heart Failure - Score: -1 - "
        "Reasoning: PMID 26630143 describes a study in which lixisenatide was trialled to see if it improved cardiovascular outcomes in patients with type 2 diabetes who recently had a recent acute coronary event. "
        "Lixisenatide was found to not be superior compared to placebo in preventing hospitalization, heart failure, or death. Score: -1.\n"
        "PMID 31422062 describes a meta-analysis of GLP-1 receptor agonists on cardiovascular outcomes. There are several GLP-1 receptor agonists considered; one of which is lixisenatide. "
        "GLP-1 receptor agonist treatment was found to have beneficial effects on cardiovascular, mortality, and kidney outcomes in patients with type 2 diabetes. However, the outcomes of lixisenatide are not specifically discussed. Score: 1.\n"
        "PMID 29221659 describes GLP-1 receptor agonists as glucose-lowering drugs. Similar to the previous abstract, several GLP-1 receptor agonists are considered, and overall they are shown to reduce cardiovascular events in patients with type 2 diabetes. "
        "However, it's unclear if lixisenatide specifically is beneficial in this application. Score: 1.\n"
        "PMID 30648112 only mentions lixisenatide as part of a study name; nothing about the hypothesis can be inferred from this abstract. Score: 0.\n"
        "PMID 30609236 states that patients with type 2 diabetes have an excess risk of heart failure, even after controlling hyperglycaemia. "
        "Lixisenatide is described as having no effect on heart failure risk. Score: -1."
    )
    return reasoning


def heart_failure_example_2():
    """
    Hypothesis: OMEGA-3-ACID ETHYL ESTERS may be a useful treatment for treating the underlying causes of Heart Failure
    """
    reasoning = (
        "Hypothesis: OMEGA-3-ACID ETHYL ESTERS may be a useful treatment for treating the underlying causes of Heart Failure - Score: 1 - "
        "Reasoning: PMID 19619690 describes the initiation of a study to determine if omega-3 ethyl esters are beneficial for atrial fibrillation, which increases the risk of heart failure. "
        "Because the abstract only describes the initiation of a study, there are no conclusions yet. However, the background presented in the abstract mentions that there is some evidence to suggest that omega-3 polyunsaturated fatty acids may have antiarrhythmic benefits. Score: 1.\n"
        "PMID 21701167 finds that omega-3-acid ethyl esters cause reduced platelet activity in patients with coronary artery disease. This is moderately strong evidence of a mechanism by which omega-3-acid ethyl esters may improve problems that cause heart disease, similar to aspirin and statins. Score: 2.\n"
        "PMID 21986642 states that omega-3 polyunsaturated fatty acids (n-3 PUFAs) improve cardiovascular health. Omega-3-acid ethyl esters are a derivative of n-3 PUFAs, and are described here as reducing triglyceride levels. "
        "n-3 PUFA consumption was associated with a lower risk of hospitalization in heart failure patients. Score: 2.\n"
        "PMID 35361442 found that high intensity interval training (HIIT) improved cardiovascular health in adults at high risk of heart failure, but omega-3-acid ethyl ester supplementation did not affect cardiovascular health. Score: -1.\n"
        "PMID 35772177 found that omega-3-acid ethyl ester supplementation did not affect cardiovascular event incidence among patients receiving statins. However, this does not necessarily mean that omega-3-acid ethyl esters are ineffective; they may just not have an additive effect on statins. Score: 0."
    )
    return reasoning


def fictitious_disease_example_1():
    """
    Hypothesis: Fludamorin can be treated with Verazol through its interaction with GeneX. GeneX could be an important intermediary in the treatment of Fludamorin with Verazol.
    """
    reasoning = (
        "Hypothesis: Fludamorin can be treated with Verazol through its interaction with GeneX - Score: -2 - "
        "Reasoning: PMID 12345678 discusses the involvement of GeneX in various genetic disorders but does not link it to Fludamorin or any similar diseases, nor does it mention Verazol as a treatment.\n"
        "PMID 23456789 reviews the pharmacodynamics of Verazol and indicates that it targets GeneY primarily, with no significant interaction with GeneX.\n"
        "PMID 34567890 details a clinical trial of Verazol for a condition unrelated to Fludamorin, in which GeneX expression was monitored and found not to be affected by Verazol treatment.\n"
        "PMID 45678901 studies the pathology of Fludamorin and specifically rules out any involvement of GeneX, suggesting other genetic factors are at play.\n"
        "PMID 56789012 explores the adverse effects of Verazol, including exacerbation of symptoms in patients with conditions similar to Fludamorin, with no evidence of beneficial interaction through GeneX or otherwise."
    )
    return reasoning

def fictitious_disease_example_2():
    """
    Hypothesis: Chronitis can be treated with Zaltron through its interaction with GeneY. GeneY could be an important intermediary in the treatment of Chronitis with Zaltron.
    """
    reasoning = (
        "Hypothesis: Chronitis can be treated with Zaltron through its interaction with GeneY - Score: -1 - "
        "Reasoning: PMID 98765432 indicates that GeneY activation is closely associated with the proliferation of Chronitis symptoms. It highlights that increased activity of GeneY exacerbates the condition.\n"
        "PMID 87654321 describes a study with patients treated with Zaltron for an unrelated condition. The study unintentionally discovered that Zaltron significantly increases the activity of GeneY.\n"
        "PMID 76543210 reviews the molecular pathways affected by Zaltron and confirms its primary action is the activation of GeneY, though it was initially developed for other therapeutic purposes.\n"
        "PMID 65432109 provides a comprehensive overview of Chronitis pathology, indicating that the activation of GeneY not only worsens the disease but may also trigger new symptoms in genetically predisposed individuals.\n"
        "PMID 54321098 discusses several cases where Zaltron was used off-label in treating similar conditions to Chronitis, leading to an increase in disease symptoms attributed to enhanced GeneY activity."
    )
    return reasoning


def fictitious_disease_example_3():
    """
    Hypothesis: Neuradox can be treated with Cytorex through its interaction with GeneZ. GeneZ could be an important intermediary in the treatment of Neuradox with Cytorex.
    """
    reasoning = (
        "Hypothesis: Neuradox can be treated with Cytorex through its interaction with GeneZ - Score: 0 - "
        "Reasoning: PMID 32109876 discusses several potential targets for treating Neuradox, but GeneZ is not specifically mentioned, nor is its connection to Cytorex explored.\n"
        "PMID 21098765 details the biochemical properties of Cytorex and mentions its interaction with multiple genes, including GeneZ, but does not provide conclusive evidence on the therapeutic outcomes for Neuradox.\n"
        "PMID 10987654 reviews a small cohort study where Cytorex was administered to patients with Neuradox. The results were inconclusive with regards to the efficacy of the treatment, and no clear connection with GeneZ activity was established.\n"
        "PMID 09876543 explores the genetic factors involved in Neuradox without linking any therapeutic benefits or detriments to the activity of GeneZ specifically.\n"
        "PMID 98765432 provides a theoretical analysis of how Cytorex might interact with GeneZ, suggesting potential mechanisms but acknowledging that empirical evidence is still lacking to support any definitive conclusions."
    )
    return reasoning


def fictitious_disease_example_4():
    """
    Hypothesis: Vasculan can be treated with Modulax through its interaction with GeneK. GeneK could be an important intermediary in the treatment of Vasculan with Modulax.
    """
    reasoning = (
        "Hypothesis: Vasculan can be treated with Modulax through its interaction with GeneK - Score: 1 - "
        "Reasoning: PMID 15975348 suggests that Modulax may indirectly inhibit the activity of GeneK, which has been identified as a promoter of Vasculan in several studies.\n"
        "PMID 35795124 discusses a small-scale preliminary study in which Modulax was used to treat Vasculan with some success, suggesting that it may interact beneficially with GeneK to reduce disease symptoms.\n"
        "PMID 75315986 describes the pharmacological profile of Modulax, noting that while it targets several pathways, its effect on GeneK is particularly promising for conditions like Vasculan.\n"
        "PMID 95135726 reports on laboratory experiments that show Modulax can modulate the expression of GeneK in cell cultures derived from Vasculan patients, providing indirect evidence of its potential effectiveness.\n"
        "PMID 86149253 reviews the therapeutic landscape for Vasculan and identifies Modulax as a candidate for further research due to its interaction with GeneK, though more definitive clinical trials are needed."
    )
    return reasoning

def fictitious_disease_example_5():
    """
    Hypothesis: Cardioxin can be treated with Luminor through its interaction with GeneQ. GeneQ could be an important intermediary in the treatment of Cardioxin with Luminor.
    """
    reasoning = (
        "Hypothesis: Cardioxin can be treated with Luminor through its interaction with GeneQ - Score: 2 - "
        "Reasoning: PMID 12345678 describes a large, well-conducted clinical trial where Luminor treatment led to significant improvement in patients with Cardioxin, specifically through its modulation of GeneQ.\n"
        "PMID 23456789 provides strong biochemical evidence showing that Luminor directly interacts with GeneQ, resulting in the inhibition of pathways that exacerbate Cardioxin symptoms.\n"
        "PMID 34567890 reports findings from a multi-center study confirming that GeneQ plays a critical role in the pathophysiology of Cardioxin, and that its modulation by Luminor can effectively manage and treat the disease.\n"
        "PMID 45678901 discusses the molecular mechanisms by which Luminor interacts with GeneQ, offering detailed insights into how this interaction contributes significantly to the therapeutic effects observed in Cardioxin patients.\n"
        "PMID 56789012 presents longitudinal data from patients treated with Luminor, showing consistent and significant long-term improvement in Cardioxin symptoms, further supported by changes in GeneQ activity."
    )
    return reasoning
