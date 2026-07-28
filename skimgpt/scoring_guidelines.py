def ab_scoring_guidelines(a_term, b_term):
    return (
        "• **-2:** The hypothesis is **refuted** by consistent evidence indicating that the interaction between "
        f"{a_term} and {b_term} **contradicts** the proposed outcome.\n"
        "• **-1:** The hypothesis is **likely refuted** based on the evidence. There is moderate indication that "
        f"the interaction between {a_term} and {b_term} **contradicts** the proposed outcome, but some uncertainty "
        "or contradictory evidence exists.\n"
        "• **0:** The hypothesis is **neither supported nor refuted** by the provided texts. The evidence regarding "
        f"the interaction between {a_term} and {b_term} is inconclusive, mixed, lacks sufficient detail, or there "
        "is a lack of evidence.\n"
        "• **+1:** The hypothesis is **likely supported** by the provided texts. The evidence suggests that the "
        f"interaction between {a_term} and {b_term} may **align with** the proposed outcome, but some uncertainty "
        "or contradictory evidence exists.\n"
        "• **+2:** The hypothesis is **supported** by consistent evidence indicating that the interaction between "
        f"{a_term} and {b_term} **aligns with** the proposed outcome, with no significant contradictory evidence."
    )


def ac_scoring_guidelines(a_term, c_term):
    return (
        "• **-2:** The hypothesis is **refuted** by consistent evidence indicating that the interaction between "
        f"{a_term} and {c_term} **contradicts** the proposed outcome.\n"
        "• **-1:** The hypothesis is **likely refuted** based on the evidence. There is moderate indication that "
        f"the interaction between {a_term} and {c_term} **contradicts** the proposed outcome, but some uncertainty "
        "or contradictory evidence exists.\n"
        "• **0:** The hypothesis is **neither supported nor refuted** by the provided texts. The evidence regarding "
        f"the interaction between {a_term} and {c_term} is inconclusive, mixed, lacks sufficient detail, or there "
        "is a lack of evidence.\n"
        "• **+1:** The hypothesis is **likely supported** by the provided texts. The evidence suggests that the "
        f"interaction between {a_term} and {c_term} may **align with** the proposed outcome, but some uncertainty "
        "or contradictory evidence exists.\n"
        "• **+2:** The hypothesis is **supported** by consistent evidence indicating that the interaction between "
        f"{a_term} and {c_term} **aligns with** the proposed outcome, with no significant contradictory evidence."
    )


def abc_scoring_guidelines(a_term, b_term, c_term):
    return (
        "• **-2:** The hypothesis is **refuted** by consistent evidence indicating that the interactions between "
        f"{a_term}-{b_term} and/or {b_term}-{c_term} **contradict** the proposed outcome.\n"
        "• **-1:** The hypothesis is **likely refuted** based on the evidence. There is moderate indication that the "
        f"interactions between {a_term}-{b_term} and/or {b_term}-{c_term} **contradict** the proposed outcome, but "
        "some uncertainty or contradictory evidence exists.\n"
        "• **0:** The hypothesis is **neither supported nor refuted** by the provided texts. The evidence regarding "
        f"the interactions between {a_term}-{b_term} and {b_term}-{c_term} is inconclusive, mixed, lacks sufficient "
        "detail, or there is a lack of evidence.\n"
        "• **+1:** The hypothesis is **likely supported** by the provided texts. The evidence suggests that the "
        f"interactions between {a_term}-{b_term} and {b_term}-{c_term} may **align with** the proposed outcome, but "
        "some uncertainty or contradictory evidence exists.\n"
        "• **+2:** The hypothesis is **supported** by consistent evidence indicating that the interactions between "
        f"{a_term}-{b_term} and {b_term}-{c_term} **align with** the proposed outcome, with no significant "
        "contradictory evidence."
    )


def cont_ab_scoring_guidelines(a_term, b_term):
    """
    Provides continuous scoring guidelines for evaluating the support of a hypothesis
    regarding the interaction between two biomedical terms.

    The scoring scale ranges from -100 to +100, where:
    - Negative scores indicate refutation of the hypothesis.
    - Positive scores indicate support for the hypothesis.
    - A score of 0 indicates neutrality or insufficient evidence.

    Parameters:
    - a_term (str): The first biomedical term in the interaction.
    - b_term (str): The second biomedical term in the interaction.

    Returns:
    - str: A formatted string containing the scoring guidelines.
    """
    return f"""
**Continuous Scoring Guidelines (-100 to +100):**

• **-100:** The hypothesis is **strongly refuted** by overwhelming and consistent evidence indicating that the interaction between {a_term} and {b_term} **directly contradicts** the proposed outcome.
  
• **-75:** The hypothesis is **significantly refuted** with substantial evidence suggesting that the interaction between {a_term} and {b_term} **contradicts** the proposed outcome.
  
• **-50:** The hypothesis is **moderately refuted** by evidence indicating a contradiction in the interaction between {a_term} and {b_term}.
  
• **-25:** The hypothesis is **slightly refuted** based on limited evidence suggesting that the interaction between {a_term} and {b_term} **contradicts** the proposed outcome.
  
• **0:** The hypothesis is **neutral**, meaning there is **no clear support or refutation** regarding the interaction between {a_term} and {b_term} based on the provided abstracts.
  
• **+25:** The hypothesis is **slightly supported** by limited evidence suggesting that the interaction between {a_term} and {b_term} **aligns with** the proposed outcome.
  
• **+50:** The hypothesis is **moderately supported** by evidence indicating that the interaction between {a_term} and {b_term} **aligns with** the proposed outcome.
  
• **+75:** The hypothesis is **significantly supported** by substantial evidence suggesting that the interaction between {a_term} and {b_term} **aligns with** the proposed outcome.
  
• **+100:** The hypothesis is **strongly supported** by overwhelming and consistent evidence indicating that the interaction between {a_term} and {b_term} **directly aligns with** the proposed outcome.

**Guidelines for Intermediate Scores:**

- **Proportional Scoring:** Assign scores between the defined anchor points based on the relative strength and consistency of the evidence. For example, a score of +60 would indicate support stronger than moderate but not as strong as significant.

- **Avoid Clustering:** Do not disproportionately favor specific integer values unless the evidence explicitly warrants it. Ensure that scores are distributed across the range to reflect the nuanced degree of support or refutation.

- **Evidence-Based Justification:** Base the score on the cumulative evidence from the abstracts. Consider the number of abstracts supporting or refuting the hypothesis, the quality of the evidence, and the presence of any contradictory information.

- **Neutrality and Insufficiency:** A score close to **0** should be assigned when the evidence is inconclusive, mixed, or insufficient to determine the support or refutation of the hypothesis.

- **Consistent Reasoning:** Ensure that the reasoning provided for the score directly correlates with the assigned value, clearly demonstrating how the evidence leads to that specific score.

**Best Practices to Maintain Objectivity:**

- **Clear Definitions:** Utilize the anchor points to anchor your scoring decisions, ensuring each score has a clear and objective basis.

- **Comprehensive Review:** Thoroughly analyze all provided abstracts to capture the full scope of evidence before assigning a score.

- **Unit Consistency:** Keep the scoring consistent across different hypotheses by adhering strictly to these guidelines, avoiding subjective or arbitrary score assignments.

By following these continuous scoring guidelines, you can provide a nuanced and objective assessment of the hypothesis' support level based on the interaction between {a_term} and {b_term}.
"""

def cont_ab_direct_comp_scoring_guidelines(hypothesis_1, hypothesis_2):
    """
    Provides continuous scoring guidelines for evaluating the support of one hypothesis
    versus another hypothesis.
    The scoring scale ranges from -0 to +100, where:
    - The score is the percentage of support for hypothesis 1 over hypothesis 2.
    - 100 indicates 100% support for hypothesis 1.
    - 0 indicates 100% support for hypothesis 2.
    - A score of 50 indicates that the two hypotheses are equally supported.
    - A score of 75 indicates that hypothesis 1 is supported three times more than hypothesis 2.
    - A score of 25 indicates that hypothesis 2 is supported three times more than hypothesis 1.
    - The scoring is based on the evidence provided in the abstracts only.
    Parameters:
    - a_term (str): The biomedical term involved in hypothesis 1.
    - b_term (str): The biomedical term involved in hypothesis 2.
    Returns:
    - str: A formatted string containing the scoring guidelines.
    """
    return f"""
**Continuous Scoring Guidelines (0 to +100):**
• **0:** Hypothesis 1 is **strongly refuted** by overwhelming and consistent evidence indicating that {hypothesis_1} **directly contradicts** the proposed outcome, while Hypothesis 2 is **strongly supported** by overwhelming and consistent evidence indicating that {hypothesis_2} **directly aligns with** the proposed outcome.
  
• **25:** Hypothesis 2 ({hypothesis_2}) is three times more likely than Hypothesis 1 ({hypothesis_1}).
  
• **50:** Hypothesis 1 ({hypothesis_1}) is equally likely compared to Hypothesis 2 ({hypothesis_2}).
  
• **75:** Hypothesis 1 ({hypothesis_1}) is three times more likely than Hypothesis 2 ({hypothesis_2}).
• **100:** Hypothesis 1 is **strongly supported** by overwhelming and consistent evidence indicating that {hypothesis_1} **directly aligns with** the proposed outcome, while Hypothesis 2 is **strongly refuted** by overwhelming and consistent evidence indicating that {hypothesis_2} **directly contradicts** the proposed outcome.
  
**Guidelines for Intermediate Scores:**
- **Proportional Scoring:** Assign scores between the defined anchor points based on the relative strength and consistency of the evidence. For example, a score of +66 would indicate approximately twice as much support for Hypothesis 1 compared to Hypothesis 2. In the case where there is no evidence for one of the Hypotheses, but there is supporting evidence for the other hypothesis, then this should generate a score in favor of the other hypothesis, the score value being dependent on the actual level of support for the other hypothesis. For instance, if there is no evidence at all for Hypothesis 2, and there are 10 abstracts that exhibit reasonably strong support for Hypothesis 1 (but do not provide absolutely unequivocal support for Hypothesis 1), then this should generate a score strongly in favor of Hypothesis 1, but not necessarily the max score of 100. 
- **Avoid Clustering:** Do not disproportionately favor specific integer values unless the evidence explicitly warrants it. Ensure that scores are distributed across the range to reflect the nuanced degree of support or refutation.
- **Evidence-Based Justification:** Base the score on the cumulative evidence from the abstracts. Consider the number of abstracts supporting or refuting each hypothesis, the quality of the evidence, and the presence of any contradictory information.
- **Neutrality and Insufficiency:** A score close to **50** should be assigned when the evidence is inconclusive, mixed, or insufficient to determine the support of both Hypotheses. Note that in the case where there is no evidence for Hypothesis 1 and no evidence for Hypothesis 2, this should generate a score of 50.
- **Consistent Reasoning:** Ensure that the reasoning provided for the score directly correlates with the assigned value, clearly demonstrating how the evidence leads to that specific score.
**Best Practices to Maintain Objectivity:**
- **Clear Definitions:** Utilize the anchor points to anchor your scoring decisions, ensuring each score has a clear and objective basis.
- **Comprehensive Review:** Thoroughly analyze all provided abstracts to capture the full scope of evidence before assigning a score.
- **Unit Consistency:** Keep the scoring consistent across different hypotheses by adhering strictly to these guidelines, avoiding subjective or arbitrary score assignments.
By following these continuous scoring guidelines, you can provide a nuanced and objective assessment of the support level for Hypothesis 1 compared to Hypothesis 2 based on the hypotheses {hypothesis_1} and {hypothesis_2}.
"""





