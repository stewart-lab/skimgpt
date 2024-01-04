def prompt_template(b_term, a_term, consolidated_abstracts):
    return "HELLO WORLD"


def build_your_own_prompt(b_term, a_term, consolidated_abstracts):
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
