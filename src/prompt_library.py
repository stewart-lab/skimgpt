from . import few_shot_library as fsl
from . import scoring_guidelines as sg


def km_prompt_template(b_term, a_term, consolidated_abstracts):
    return "HELLO WORLD"


def skim_prompt_template(b_term, a_term, consolidated_abstracts, c_term):
    return "HELLO WORLD"


def drug_disease_few_shot_prompt(b_term, a_term, consolidated_abstracts):
    return (
        "Assessment Task:\n"
        "Conduct a thorough analysis of the provided biomedical texts to evaluate the level of support for the stated hypothesis. "
        "Assign a score based on the evidence's strength and relevance. This score should encapsulate the degree to which the research data "
        "and findings in the texts corroborate or refute the hypothesis. Ensure that your score is supported by specific references to the texts. "
        "Consider the quality of the research, the relevance of the findings to the hypothesis, and the presence of any limitations or conflicting "
        "evidence in your evaluation.\n\n"
        "Hypothesis:\n"
        f"{b_term} may effectively alleviate or target key pathogenic mechanisms of {a_term}, potentially offering therapeutic benefits or slowing disease progression.\n\n"
        "Instructions:\n"
        "Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or "
        "opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple "
        "abstracts, as no single abstract will be conclusive. Provide a justification for your score in simple terms understandable to an "
        "undergraduate biochemist.\n\n"
        "Few-Shot Learning Examples:\n"
        "These are examples showing how to apply the scoring guidelines to specific hypotheses based on the provided abstracts. Review these examples to understand how to analyze the texts and justify the scoring.\n"
        
        f"Example 1: {fsl.breast_cancer_example_1()}\n"
        f"Example 2: {fsl.breast_cancer_example_2()}\n"
        f"Example 3: {fsl.raynauds_disease_example_1()}\n"
        f"Example 4: {fsl.raynauds_disease_example_2()}\n"
        f"Example 5: {fsl.heart_failure_example_1()}\n"
        f"Example 6: {fsl.heart_failure_example_2()}\n"
        "Format your response as:\n"
        "Score: [Number] - Reasoning: [Reasoning]\n\n"
        "Scoring Guidelines:\n"
        f"{sg.gpt_customized_scoring_system(b_term, a_term)}\n"
        "Biomedical Abstracts for Analysis:\n"
        f"{consolidated_abstracts}"
    )


def diabetes_few_shot_prompt(b_term, a_term, consolidated_abstracts):
    return (
        "Assessment Task:\n"
        "Evaluate the support for the following hypothesis in the given biomedical texts, giving a score from -1 to 3.\n\n"
        "Hypothesis:\n"
        f"{b_term} may be a useful treatment for treating the underlying causes of {a_term}\n\n"
        "Instructions:\n"
        "Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or "
        "opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple "
        "abstracts, as no single abstract will be conclusive. Provide a justification for your score in simple terms understandable to an "
        "undergraduate biochemist.\n\n"
        "Few-Shot Learning Examples:\n"
        "These are examples showing how to apply the scoring guidelines to specific hypotheses based on the provided abstracts. Review these examples to understand how to analyze the texts and justify the scoring.\n"
        "Example 1: Hypothesis: Metformin may be a useful treatment for treating the underlying causes of Diabetes - Score: 3 - Reasoning: PMID 11832527 states that metformin reduces the incidence of type 2 diabetes. Elevated plasma glucose levels, being overweight, and having a sedentary lifestyle are reversible risk factors for diabetes. Metformin reduced the incidence of diabetes in high-risk prediabetic patients by 31%. Score: 3.\n"
        "PMID 9742977 states that metformin slows the progression of diabetes by controlling blood glucose levels and reduces the incidence diabetes-related risks such as death in diabetic patients. Score: 3.\n"
        "PMID 18784090 states that benefits from metformin treatment on diabetic patients were long-lasting. Score: 3.\n"
        "PMID 11602624 investigates the mechanism by which metformin aids glucose control, concluding that metformin activates AMPK, which leads to reduced ACC activity, increased fatty acid oxidation, suppressed expression of lipogenic enzymes, and suppressed expression of SREBP-1. Score: 3.\n\n"
        "Example 2: Hypothesis: Ezetimibe may be a useful treatment for treating the underlying causes of Diabetes - Score: -1 - Reasoning: PMID 23956253 discusses familial hypercholesterolaemia in relation to coronary heart disease (CHD). Low-density lipoprotein (LDL) levels are discussed; however, there is no obvious connection between diabetes, LDL, and ezetimibe presented. Score: 0.\n"
        "PMID 23444397 implies that ezetimibe helps manage LDL levels. Diabetes is mentioned in passing. Score: 0.\n"
        "PMID 28886926 implies that ezetimibe is useful for lowering LDL levels to lower atherosclerotic cardiovascular disease (ASCVD) risk. Diabetes is mentioned in passing. Score: 0.\n"
        "PMID 27701660 found that LDL-lowering genetic variants were positively associated with a higher risk of type 2 diabetes. This implies that lowering LDL levels could possibly increase diabetes risk. “potential adverse effects of LDL-C-lowering therapy” is mentioned at the end. Score: -1.\n"
        "PMID 19095139 examines the effects of lowering LDL-C with statins and ezetimibe in diabetic patients with regards to atherosclerosis. This article is not relevant to the hypothesis. Score: 0.\n"
        "PMID 21095263 states that lowering LDL levels reduces the incidence of atherosclerotic events. This article is not relevant to the hypothesis. Score: 0.\n\n"
        "Example 3: Hypothesis: bumetanide may be a useful treatment for treating the underlying causes of Diabetes - Score: 1 - Reasoning: PMID 32410463 indicates that empagliflozin synergized with bumetanide to help flush sodium and glucose via a diuretic effect. This is mostly discussed in relation to heart failure, but lowering blood glucose helps prevent diabetes. Score: 1.\n"
        "PMID 29440005 discusses treatment with dapagliflozin to lower blood glucose levels in diabetic patients via inhibition of the sodium-glucose-linked transporter 2. Bumetanide is hypothesized to synergize with dapagliflozin to help flush sodium and glucose from the blood. Score: 1.\n"
        "PMID 9185518 describes the BSC2 Na-K-Cl cotransporter as “bumetanide-sensitive”. The experiments described in this article suggest that BSC2 is regulated by inflammatory cytokines and fluid mechanical stimuli, implying a role for BSC2 in “vascular homeostasis and inflammation.” This is related to diabetes because blood glucose levels affect osmotic conditions.\n"
        "PMID 12904328 states that diabetes is associated with water and sodium losses, and the authors hypothesize that the kidney compensates for this by upregulating sodium and water transporter proteins. Experiments show that the bumetanide-sensitive Na-K-2Cl cotransporter (NKCC2) correlate with blood glucose levels. It’s unclear from this text how bumetanide would affect blood glucose levels. Score: 0\n"
        "Example 4: Hypothesis: sodium may be a useful treatment for treating the underlying causes of Diabetes - Score: 0 - Reasoning: The texts provided do not discuss sodium as a treatment, but rather in the context of the sodium-glucose cotransporter 2 (SGLT2). Several articles describe inhibiting SGLT2 in an effort to improve cardiovascular health in diabetic patients. Score: 0.\n\n"
        "Format your response as:\n"
        "Score: [Number] - Reasoning: [Reasoning]\n\n"
        "Scoring Guidelines:\n"
        "   - -1 Point: There is sufficient evidence in the provided texts to say that the hypothesis is probably not true. The evidence described in the texts directly contradict the hypothesis.\n"
        "   - 0 Points: There is no evidence to support the hypothesis in the provided texts. The experiments described in the texts may or may not be well conducted, but they do not have any bearing on the hypothesis.\n"
        "   - 1 Point: Evidence that supports the hypothesis is weak in the provided texts. The experiments seem poorly conducted, or there is mixed evidence that the hypothesis is true.\n"
        "   - 2 Points: Evidence that supports the hypothesis is moderate in the provided texts. The experiments are well conducted, and there is a fair amount of evidence in the texts that the hypothesis may be true.\n"
        "   - 3 Points: Evidence that supports the hypothesis is strong in the provided texts. There is a large amount of direct evidence in the texts that the hypothesis is true.\n\n"
        "Biomedical Abstracts for Analysis:\n"
        f"{consolidated_abstracts}"
    )


def diabetes_drug_prompt(b_term, a_term, consolidated_abstracts):
    return (
        f"Assessment Task:\n"
        f"Evaluate the support for the following hypothesis in the given biomedical texts, giving a score from -1 to 3.\n\n"
        f"Hypothesis:\n"
        f"{b_term} may be a useful treatment for treating the underlying causes of {a_term}\n\n"
        f"Instructions:\n"
        f"Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or "
        f"opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple "
        f"abstracts, as no single abstract will be conclusive. Provide a justification for your score in simple terms understandable to an "
        f"undergraduate biochemist.\n\n"
        f"Format your response as:\n"
        f"Score: [Number] - Reasoning: [Reasoning]\n\n"
        f"Scoring Guidelines:\n"
        f"   - -1 Point: There is sufficient evidence in the provided texts to say that the hypothesis is probably not true. The evidence described in the texts directly contradict the hypothesis.\n"
        f"   - 0 Points: There is no evidence to support the hypothesis in the provided texts. The experiments described in the texts may or may not be well conducted, but they do not have any bearing on the hypothesis.\n"
        f"   - 1 Point: Evidence that supports the hypothesis is weak in the provided texts. The experiments seem poorly conducted, or there is mixed evidence that the hypothesis is true.\n"
        f"   - 2 Points: Evidence that supports the hypothesis is moderate in the provided texts. The experiments are well conducted, and there is a fair amount of evidence in the texts that the hypothesis may be true.\n"
        f"   - 3 Points: Evidence that supports the hypothesis is strong in the provided texts. There is a large amount of direct evidence in the texts that the hypothesis is true.\n\n"
        f"Biomedical Abstracts for Analysis:\n{consolidated_abstracts}"
    )


def skim_prompt(b_term, a_term, consolidated_abstracts, c_term):
    return (
        f"Assessment Task: Critically evaluate the potential indirect relationship between '{a_term}' and '{c_term}' within the provided biomedical texts, assigning a score of -1, 0, or 1. \n"
        f"Hypothesis: 'There exists an indirect link between '{a_term}' and '{c_term}' mediated through '{b_term} found in biomedical interactions.' \n"
        f"Instructions: Utilize the information within the provided abstracts to infer the potential indirect relationship. Employ analytical skills to deduce connections, ensuring your reasoning is accessible to an undergraduate biochemist. Format your response as 'Score: [Number] - Reasoning: [Your Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"   - -1 Point: The evidence strongly negates any indirect relationship between '{a_term}' and '{c_term}'. Clear, direct information in the texts contradicts the existence of such a link, with no plausible alternative interpretations.\n"
        f"   - 0 Points: The evidence neither strongly supports nor negates the indirect relationship. This includes cases where the evidence is neutral, inconclusive, or not directly relevant to the hypothesized link. Evidence that does not explicitly negate or support the relationship falls here, even if substantial extrapolation or inference is needed.\n"
        f"   - 1 Point: The evidence suggests a plausible indirect relationship between '{a_term}' and '{c_term}'. This includes cases where the evidence, though perhaps indirect or requiring some inference, supports the existence of a connection via '{b_term}'.\n"
        f"Biomedical Abstracts for Analysis:\n{consolidated_abstracts}"
    )


def hypothesis_confirmation(b_term, a_term, consolidated_abstracts):
    return (
        f"Assessment Task: Critically evaluate the support for the following hypothesis within the provided biomedical texts, assigning a score of -1 or 0. \n"
        f'Hypothesis: "{b_term} causes {a_term}." \n'
        f"Instructions: Rely solely on the information within the provided abstract to judge the hypothesis. Use analytical skills to interpret the data, and articulate your reasoning in terms understandable to an undergraduate biochemist. Format: 'Score: [Number] - Reasoning: [Your Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"   - -1 Point: The evidence decisively refutes the hypothesis. There is unambiguous and compelling information in the texts that directly contradicts the hypothesis, leaving no room for alternative interpretation.\n"
        f"   - 0 Points: The evidence does not decisively refute the hypothesis. This includes cases where the evidence is supportive, neutral, inconclusive, or not directly relevant. Any evidence that does not explicitly negate the hypothesis falls into this category, even if it requires significant extrapolation or inference to draw a link.\n"
        f"Biomedical Abstract for Analysis:\n{consolidated_abstracts}"
    )


def hypothesis_confirmation_rms(b_term, a_term, consolidated_abstracts):
    # print("Aterm:", a_term)
    return (
        f"Assessment Task: Critically evaluate the support for the following hypothesis within the provided biomedical texts, assigning a score of -1, 0, or 1. \n"
        f'Hypothesis: "{b_term} and {a_term} will have a drug-drug interaction." \n'
        f"Instructions: Rely solely on the information within the provided abstract to judge the hypothesis. Use analytical skills to interpret the data, and articulate your reasoning in terms understandable to an undergraduate biochemist. Format: 'Score: [Number] - Reasoning: [Your Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"   - -1 Point: The evidence decisively refutes the hypothesis. There is unambiguous and compelling information in the texts that directly contradicts the hypothesis, leaving no room for alternative interpretation.\n"
        f"   - 0 Points: The evidence does not decisively refute the hypothesis. This includes cases where the evidence is neutral, inconclusive, or not directly relevant. Any evidence that does not explicitly negate the hypothesis falls into this category, even if it requires significant extrapolation or inference to draw a link.\n"
        f"   - 1 Point: The evidence is supportive of the hypothesis. This includes cases where the evidence is supportive.\n"
        f"Biomedical Abstracts for Analysis:\n{consolidated_abstracts}"
    )


def hypothesis_confirmation_ddi(b_term, a_term, consolidated_abstracts):
    return (
        f"Assessment Task: Critically evaluate the support for the following hypothesis within the provided biomedical texts, assigning a score of -1, 0, or 1. \n"
        f'Hypothesis: "{b_term} and {a_term} will have a drug-drug interaction." \n'
        f"Instructions: Rely solely on the information within the provided abstract to judge the hypothesis. Use analytical skills to interpret the data, and articulate your reasoning in terms understandable to an undergraduate biochemist. Format: 'Score: [Number] - Reasoning: [Your Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"   - -2 Points: The evidence decisively refutes the hypothesis. There is unambiguous and compelling information in the texts that directly contradicts the hypothesis, leaving no room for alternative interpretation.\n"
        f"   - 1 Points: The evidence does not decisively refute the hypothesis, but there is indirect evidence that the hypothesis should be refuted. This includes cases where the evidence is indirect, but seems to refute the hypothesis.\n"
        f"   - 0 Points: The evidence does not decisively or indirectly refute the hypothesis. This includes cases where the evidence is neutral, inconclusive, or not relevant. Any evidence that does not negate the hypothesis falls into this category, even if it requires significant extrapolation or inference to draw a link.\n"
        f"   - 1 Point: The evidence is weakly supportive of the hypothesis. This includes cases where the evidence is supportive but indirect such as:  drug A binds X and drugB also binds X.\n"
        "   -  2 Points: The evidence is strongly supportive of the hypothesis. This includes cases where the evidence is directly supportive.\n"
        f"Biomedical Abstract for Analysis:\n{consolidated_abstracts}"
    )


def alzheimer_gene_prompt_2(b_term, a_term, consolidated_abstracts):
    return (
        f"Assessment Task: Evaluate the support for the following hypothesis in the given biomedical texts, assigning a score from -1 to 5. \n"
        f"Hypothesis: \"In Alzheimer's disease patients, the levels of protein product(s) of gene {b_term} in blood plasma differ significantly from those in individuals without Alzheimer's disease.\" \n"
        f"Instructions: Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple abstracts, as no single abstract will be conclusive. Provide a justification for your score in simple terms understandable to an undergraduate biochemist. Format your response as: 'Score: [Number] - Reasoning: [Your Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"   - -1 Point: Evidence is moderately or strongly against the hypothesis. There is consistent evidence in the provided texts that the hypothesis is not true; little or no extrapolation is needed to reach this conclusion.\n"
        f"   - 0 Points: There is no evidence to support the hypothesis. Either the abstracts are not relevant to the hypothesis, or the conducted studies reached no conclusions to support or refute the hypothesis.\n"
        f"   - 1 Point: Evidence is weak, with unclear or poorly conducted studies that support the hypothesis. The hypothesis might be a reasonable guess at best. A high amount of extrapolation or inference is required based on available information.\n"
        f"   - 2 Points: Evidence is weak-to-moderate, and could be worth a second glance. There are a few studies that are well conducted that support the hypothesis. A relatively high amount of extrapolation or inference is required to support the hypothesis.\n"
        f"   - 3 Points: Evidence is moderate, with clear, high-quality methodology and results that support the hypothesis. The results may be somewhat inconsistent, but generally favor the hypothesis.\n"
        f"   -  4 Points: Evidence is fairly strong, with well-conducted studies and fairly definitive results that support the hypothesis. The results are mostly consistent between abstracts.\n"
        f"   - 5 Points: Evidence is strong, with well-conducted studies and clear, definitive results that support the hypothesis. The results are consistent between abstracts. Little extrapolation or inference is required to determine that the hypothesis is very likely true.\n"
        f"Biomedical Abstracts for Analysis:\n{consolidated_abstracts}"
    )


def alzheimer_gene_prompt_1(b_term, a_term, consolidated_abstracts):
    return (
        f"Assessment Task: Evaluate the validity of the following hypothesis, giving a score from 0 to 10 based on the provided biomedical abstracts. \n"
        f"Hypothesis: \"In Alzheimer's disease patients, the levels of protein product(s) of gene {b_term} in blood plasma differ significantly from those in individuals without Alzheimer's disease.\" \n"
        f"Instructions: Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple abstracts, as no single abstract will be conclusive. \n"
        f"Provide a justification for your score in simple terms understandable to an undergraduate biochemist. Format your response as: 'Score: [Number] - Classification: [Reasoning]'.\n"
        f"Scoring Guidelines:\n"
        f"1. Relevance (0-3 Points):\n"
        f"   - 0 Points: Abstracts provide no relevant information regarding the gene/protein in Alzheimer's disease.\n"
        f"   - 1 Point: Abstracts provide minimal relevant information, tangential to the hypothesis.\n"
        f"   - 2 Points: Abstracts provide relevant information that somewhat supports or refutes the hypothesis.\n"
        f"   - 3 Points: Abstracts provide highly relevant information directly supporting or refuting the hypothesis.\n"
        f"2. Consistency Across Abstracts (0-3 Points):\n"
        f"   - 0 Points: Findings are inconsistent across abstracts, leading to conflicting interpretations.\n"
        f"   - 1 Point: Some consistency is observed, but significant contradictions exist.\n"
        f"   - 2 Points: Most abstracts show consistency in findings, with minor contradictions.\n"
        f"   - 3 Points: Strong consistency in findings across all abstracts.\n"
        f"3. Strength of Evidence (0-2 Points):\n"
        f"   - 0 Points: Evidence is weak, with unclear or poorly conducted studies.\n"
        f"   - 1 Point: Evidence is moderate, with some clear methodology and results.\n"
        f"   - 2 Points: Evidence is strong, with well-conducted studies and clear, definitive results.\n"
        f"4. Analytical Extrapolation (0-2 Points):\n"
        f"   - 0 Points: No analytical extrapolation evident, with unsubstantiated conclusions.\n"
        f"   - 1 Point: Basic analytical extrapolation evident but lacks depth.\n"
        f"   - 2 Points: Strong analytical skills shown, with insightful extrapolation from data.\n"
        f"Total Score: Sum of the above points.\n"
        f"Classification of Score:\n"
        f"   - 0-4 Points: Low Validity\n"
        f"   - 5-7 Points: Moderate Validity\n"
        f"   - 8-10 Points: High Validity\n"
        f"Example Justification:\n"
        f"   'Score: 8 - Classification: High Validity. The abstracts consistently show a significant difference in protein levels of gene {b_term} in Alzheimer’s patients. While some abstracts offer stronger evidence than others, the hypothesis is well-supported overall, reinforced by analytical extrapolation.'\n"
        f"The biomedical abstracts follow:\n{consolidated_abstracts}"
    )


def drug_process_relationship_classification_prompt(b_term, a_term, abstract):
    return (
        f"Read the following abstract carefully. Using the abstract, classify the relationship between {b_term} and {a_term} into one of the "
        f"following categories :"
        f"{b_term} is useful for treating {a_term}, "
        f"{b_term} is potentially useful for treating {a_term}, "
        f"{b_term} is ineffective for treating {a_term}, "
        f"{b_term} is potentially harmful for treating {a_term}, "
        f"{b_term} is harmful for treating {a_term}, "
        f"{b_term} is useful for treating only a specific symptom of {a_term}, "
        f"{b_term} is potentially useful for treating only a specific symptom of {a_term}, "
        f"{b_term} is potentially harmful for treating only a specific symptom of {a_term}, "
        f"{b_term} is harmful for treating only a specific symptom of {a_term}, "
        f"{b_term} is useful for diagnosing {a_term}, "
        f"{b_term} is potentially useful for diagnosing {a_term}, "
        f"{b_term} is ineffective for diagnosing {a_term}, "
        f"The relationship between {b_term} and {a_term} is unknown. "
        f"Provide at least two sentences explaining your classification. Your answer should be in the following format: 'Classification': "
        f"'Rationale': {abstract}"
    )


def drug_process_relationship_scoring(term):
    return [
        (f"{term} is useful for treating", 3),
        (f"{term} is potentially useful for treating", 2),
        (f"{term} is ineffective for treating", -1),
        (f"{term} is potentially harmful for treating", -2),
        (f"{term} is harmful for treating", -3),
        (f"{term} is useful for treating only a specific symptom of", 2),
        (f"{term} is potentially useful for treating only a specific symptom of", 1),
        (f"{term} is potentially harmful for treating only a specific symptom of", -1),
        (f"{term} is harmful for treating only a specific symptom of", -2),
        (f"{term} is useful for diagnosing", 1),
        (f"{term} is potentially useful for diagnosing", 0.5),
        (f"{term} is ineffective for diagnosing", -0.5),
        (f"The relationship between {term} and", 0),
    ]


def drug_synergy_prompt(b_term, a_term, consolidated_abstracts):
    return (
        f"I want you to determine the reasonability of this hypothesis based on the biomedical texts below and provide a score between 0 and 10. "
        f'The hypothesis is "Inhibiting BRD4 and {b_term} will synergistically inhibit {a_term}". '
        f"I want you to only use the information in these abstracts, but you should extrapolate based on the available information. "
        f"Support for or against the hypothesis will not be spelled out explicitly in these abstracts. "
        f"You will need to look across abstracts to determine the score for the hypothesis; no single abstract contains this information. "
        f"Support your score with a sentence or two of explanation, using language that an undergraduate biochemist could understand."
        f"Your answer should be in the following format: 'Score: [Number] - Classification': 'Rationale'\n"
        f"Scoring System:\n"
        f"10 - Highly Reasonable: Strongly supported by multiple abstracts with a clear synergistic relationship.\n"
        f"7-9 - Reasonable: Supported by some abstracts with evidence of a potential synergistic relationship.\n"
        f"4-6 - Uncertain: Mixed evidence from the abstracts; unclear synergistic relationship.\n"
        f"1-3 - Slightly Not Reasonable: Limited support with little to no evidence of a synergistic relationship.\n"
        f"0 - Not Reasonable: No support and evidence against a synergistic relationship.\n"
        f"The biomedical abstracts follow:\n{consolidated_abstracts}"
    )


def pathway_augmentation_prompt(b_term, a_term, consolidated_abstracts):
    return (
        f"Using only the information from the biomedical abstracts provided, determine if the gene '{b_term}' "
        f"is in the pathway '{a_term}'. "
        f"Provide a binary classification (Yes or No) and at least two sentences explaining the rationale behind your classification. "
        f"The biomedical abstracts follow:\n{consolidated_abstracts}"
    )


def exercise_3_few_shot_prompt(b_term, a_term, consolidated_abstracts):
    return (
        "Assessment Task:\n"
        "Conduct a thorough analysis of the provided biomedical texts to evaluate the level of support for the stated hypothesis. "
        "Assign a score based on the evidence's strength and relevance. This score should encapsulate the degree to which the research data "
        "and findings in the texts corroborate or refute the hypothesis. Ensure that your score is supported by specific references to the texts. "
        "Consider the quality of the research, the relevance of the findings to the hypothesis, and the presence of any limitations or conflicting "
        "evidence in your evaluation.\n\n"
        "Hypothesis:\n"
        f"Being diagnosed with {a_term} will positively impact {b_term}\n\n"
        "Instructions:\n"
        "Use only the information from the provided abstracts to assess the hypothesis. While the abstracts may not explicitly state support or "
        "opposition to the hypothesis, use your analytical skills to extrapolate the necessary information. Synthesize findings from multiple "
        "abstracts, as no single abstract will be conclusive. Provide a justification for your score in simple terms understandable to an "
        "undergraduate biochemist.\n\n"
        "Format your response as:\n"
        "Score: [Number] - Reasoning: [Reasoning]\n\n"
        "Scoring Guidelines:\n"
        f"{sg.gpt_customized_scoring_system(b_term, a_term)}\n"
        "Biomedical Abstracts for Analysis:\n"
        f"{consolidated_abstracts}"
    )
