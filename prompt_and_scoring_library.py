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
        f"is involved in the pathway '{a_term}'. "
        f"Provide a binary classification (Yes or No) and at least two sentences explaining the rationale behind your classification. "
        f"The biomedical abstracts follow:\n{consolidated_abstracts}"
    )
