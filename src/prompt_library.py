import src.scoring_guidelines as sg


def km_with_gpt(b_term, a_term, hypothesis_template, consolidated_abstracts):
    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Assessment Task:
Evaluate the degree of support for the hypothesis, which posits a significant interaction between {a_term} and {b_term}. 
The texts provided above come from PubMed and each abstract will include only {a_term} and {b_term}. 
The texts need to be your only source of information for arriving at your classification result. If texts arent available, use what you know about {a_term} and {b_term} to make an educated guess.

Hypothesis:
{hypothesis_template}

Instructions:
1. Review each abstract to understand how {a_term} and {b_term} might be interconnected based on the available information.
2. Analyze the presence and implications of the term pairing ({a_term} + {b_term}) in the context of the hypothesis.
3. Synthesize the findings from multiple texts. Consider how the pieces fit together to support or refute the hypothesis: {hypothesis_template}. Remember, no single text may be conclusive.
4. Provide a justification for your scoring decision based on the analysis. Explain your reasoning step-by-step in terms understandable to an undergraduate biochemist. Focus on explaining the logical connections and the directionality of relationships.
5. Cite specific texts from your set of abstracts to support your arguments. Clearly reference these citations in your reasoning.

Format your response as:
Score: [Number] - Reasoning: [Reasoning]

Scoring Guidelines:
{sg.ab_scoring_guidelines(a_term, b_term)}"""


def skim_with_gpt_ac(a_term, hypothesis_template, consolidated_abstracts, c_term):
    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Assessment Task:
Evaluate the degree of support for the hypothesis: {hypothesis_template}, which posits an interaction between {a_term} and {c_term}. Use only the provided abstracts from PubMed that mention {a_term} and {c_term} to inform your analysis.

Your evaluation should:

• Integrate evidence from all abstracts to discern how {a_term} and {c_term} might be interconnected.
• Employ logical reasoning, drawing logical inferences where appropriate, based on the interaction between {a_term} and {c_term}.
• Assess the nature and directionality of the interaction, determining whether they are beneficial or detrimental to the proposed outcome in the hypothesis.
• Be vigilant for any evidence that contradicts or challenges the hypothesis, explicitly addressing any contradictions in your reasoning.
• Avoid inferring effects not supported by the texts, ensuring that all conclusions are grounded in the provided information.
• Ensure that your scoring reflects an unbiased assessment based solely on the provided evidence, considering logical inferences from evidence.

**Examples:**

• **Example 1: Strong Positive Outcome**
  - **Hypothesis:** {c_term} treats {a_term}.
  - **Evidence:**
     Multiple abstracts show that {c_term} **inhibits processes that worsen {a_term}**.
     Inhibition of these processes significantly **improves {a_term}** in various contexts.
  - **Logical Conclusion:**
     Since {c_term} inhibits harmful processes, and inhibition of these processes improves {a_term}, there is strong, consistent evidence supporting the hypothesis.
  - **Scoring:**
     The interactions are consistent and beneficial.
     **Assigned Score:** **+2**

• **Example 2: Likely Positive Outcome**
  - **Hypothesis:** {c_term} treats {a_term}.
  - **Evidence:**
     Some abstracts suggest that {c_term} **may activate beneficial pathways affecting {a_term}**.
     Activation of these pathways might improve {a_term}, but evidence is limited or not robust.
  - **Logical Conclusion:**
     There is evidence indicating potential beneficial interaction, but some uncertainty exists due to limited data or minor contradictions.
  - **Scoring:**
     The hypothesis is likely supported by the evidence.
     **Assigned Score:** **+1**

• **Example 3: Neutral Outcome**
  - **Hypothesis:** {c_term} treats {a_term}.
  - **Evidence:**
     Abstracts provide insufficient or inconclusive information about the interaction.
     Evidence might be mixed or does not directly relate to the hypothesis.
  - **Logical Conclusion:**
     There is not enough evidence to support or refute the hypothesis.
  - **Scoring:**
     The evidence is inconclusive.
     **Assigned Score:** **0**

• **Example 4: Likely Negative Outcome**
  - **Hypothesis:** {c_term} treats {a_term}.
  - **Evidence:**
     Some abstracts indicate that {c_term} **activates processes that worsen {a_term}**.
     Activation of these processes may worsen {a_term}, but evidence is limited or not definitive.
  - **Logical Conclusion:**
     There is substantial indication that the interaction may be detrimental to the proposed outcome, but some uncertainty or exceptions exist.
  - **Scoring:**
     The hypothesis is likely refuted based on the evidence.
     **Assigned Score:** **-1**

• **Example 5: Strong Negative Outcome**
  - **Hypothesis:** {c_term} treats {a_term}.
  - **Evidence:**
     Multiple abstracts consistently show that {c_term} **inhibits beneficial processes for {a_term}**.
     Inhibition of these processes clearly worsens {a_term} across various studies.
     No evidence suggests any interaction aligns with supporting the hypothesis.
  - **Logical Conclusion:**
     Since {c_term} inhibits beneficial processes, and inhibition of these processes worsens {a_term}, the hypothesis is clearly refuted by strong, consistent evidence.
  - **Scoring:**
     The interaction does not align with supporting the hypothesis.
     **Assigned Score:** **-2**

Your goal is to determine the degree of support for the hypothesis:

{hypothesis_template}

**Instructions:**

1. **Review each abstract** to understand how {a_term} and {c_term} might be interconnected based on the available information.
2. **Analyze the presence and implications** of the interaction between {a_term} and {c_term} in the context of the hypothesis.
3. **Carefully assess whether the interaction is beneficial or detrimental** to the proposed outcome in the hypothesis.
4. **Synthesize the findings from multiple texts**, considering how the pieces fit together to support or refute the hypothesis: {hypothesis_template}.
5. **Provide a justification for your scoring decision** based on the analysis. **Explain your reasoning step-by-step** in terms understandable to an undergraduate biochemist. Focus on:
   - **Explaining the logical connections and the directionality of relationships.**
   - **Determining whether the interaction supports or contradicts the hypothesis**, using evidence and logical inferences.
   - **Assessing if the interaction is beneficial or detrimental** to the proposed outcome.
6. **Be vigilant for any evidence that contradicts or challenges the hypothesis**. Address any contradictions explicitly in your reasoning.
7. **Verify any assumptions about the roles and effects** of {a_term} and {c_term} as presented in the abstracts. **Avoid inferring effects not supported by the texts**.
8. In your final assessment, **explicitly cite the scoring guideline** that corresponds to your conclusion. **Explain why the evidence meets the criteria for that specific score**.

**Note:** Pay close attention to the nature of the interaction between entities. An interaction that activates a harmful process may be detrimental, while inhibition of a beneficial process may also be detrimental. **Consider whether such interaction contradicts the hypothesis**.

**Definitions:**
• **Evidence:** Multiple sources agree and provide clear indications supporting a particular interaction and its effects.
• **Contradictory Evidence:** Evidence that directly opposes the hypothesis, showing that the proposed mechanism of action does not produce the expected outcome.

**Checklist Before Finalizing Your Response:**
• Have you **addressed whether the interaction supports or contradicts the hypothesis**?
• Have you **assessed if the interaction is beneficial or detrimental**?
• Have you **explicitly cited the scoring guideline** that matches your conclusion?
• Have you **explained why the evidence meets the criteria** for the assigned score?

**Format your response as:**

Score: \[Number\] Point(s) - Reasoning: \[Reasoning\]

**Scoring Guidelines:**
{sg.ac_scoring_guidelines(a_term, c_term)}"""


def skim_with_gpt(b_term, a_term, hypothesis_template, consolidated_abstracts, c_term):
    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Assessment Task:
Evaluate the degree of support for the hypothesis, which posits an interaction between {a_term} and {c_term} through their own interactions with {b_term}. Use only the provided abstracts from PubMed, each containing only two of the three terms at a time ({a_term} + {b_term}, or {b_term} + {c_term}), to inform your analysis.

Your evaluation should:

- **Integrate evidence from all abstracts** to discern how {a_term}, {b_term}, and {c_term} might be interconnected.

- **Assess the directionality and nature of each interaction** (e.g., activation, inhibition, reactivation) to determine whether it **aligns with** or **contradicts** the hypothesis.

- **Identify and analyze any opposing mechanisms** where one interaction may negate the effects of another.

- **Employ logical reasoning**, drawing logical inferences where appropriate, based on {a_term}-{b_term} and {b_term}-{c_term} interactions.

- **Assess the nature and directionality of the interactions**, determining whether they are beneficial or detrimental to the proposed outcome in the hypothesis.

- **Be vigilant for any evidence that contradicts or challenges the hypothesis**, explicitly addressing any contradictions in your reasoning.

- **Avoid inferring effects not supported by the texts**, ensuring that all conclusions are grounded in the provided information.

- **Ensure that your scoring reflects an unbiased assessment based solely on the provided evidence**, considering logical inferences from indirect evidence.

**Examples:**

- **Example 1: Strong Positive Outcome**

  - **Hypothesis:** {c_term} treats {a_term} through {b_term}.

  - **Evidence:**

    - Multiple abstracts show that {c_term} **inhibits** {b_term}.

    - Inhibition of {b_term} significantly **improves** {a_term} in various contexts.

  - **Logical Conclusion:**

    - Since {c_term} inhibits {b_term}, and inhibition of {b_term} improves {a_term}, there is strong, consistent indirect evidence supporting the hypothesis.

  - **Scoring:**

    - The interactions are consistent and beneficial.

    - **Assigned Score:** **+2**

- **Example 2: Likely Positive Outcome**

  - **Hypothesis:** {c_term} treats {a_term} through {b_term}.

  - **Evidence:**

    - Some abstracts suggest that {c_term} **may activate** {b_term}.

    - Activation of {b_term} **might improve** {a_term}, but evidence is limited or not robust.

  - **Logical Conclusion:**

    - There is evidence indicating potential beneficial interactions, but some uncertainty exists due to limited data or minor contradictions.

  - **Scoring:**

    - The hypothesis is likely supported by the evidence.

    - **Assigned Score:** **+1**

- **Example 3: Neutral Outcome**

  - **Hypothesis:** {c_term} treats {a_term} through {b_term}.

  - **Evidence:**

    - Abstracts provide insufficient or inconclusive information about the interactions.

    - Evidence might be mixed or does not directly relate to the hypothesis.

  - - **Logical Conclusion:**

    - There is not enough evidence to support or refute the hypothesis.

  - **Scoring:**

    - The evidence is inconclusive.

    - **Assigned Score:** **0**

- **Example 4: Likely Negative Outcome**

  - **Hypothesis:** {c_term} treats {a_term} through {b_term}.

  - **Evidence:**

    - Some abstracts indicate that {c_term} **activates** {b_term}.

    - Activation of {b_term} **may worsen** {a_term}, but evidence is limited or not definitive.

  - **Logical Conclusion:**

    - There is moderate indication that the interactions may be detrimental to the proposed outcome, but some uncertainty or exceptions exist.

  - **Scoring:**

    - The hypothesis is likely refuted based on the evidence.

    - **Assigned Score:** **-1**

- **Example 5: Strong Negative Outcome**

  - **Hypothesis:** {c_term} treats {a_term} through {b_term}.

  - **Evidence:**

    - Multiple abstracts consistently show that {c_term} **inhibits** {b_term}.

    - Inhibition of {b_term} **clearly worsens** {a_term} across various studies.

    - No evidence suggests any interaction aligns with supporting the hypothesis.

  - **Logical Conclusion:**

    - Since {c_term} inhibits {b_term}, and inhibition of {b_term} worsens {a_term}, the hypothesis is clearly refuted by strong, consistent evidence.

  - **Scoring:**

    - The interactions do not align with supporting the hypothesis.

    - **Assigned Score:** **-2**

Your goal is to determine the degree of support for the hypothesis:

{hypothesis_template}

**Instructions:**

1. **Review each abstract** to understand how {a_term}, {b_term}, and {c_term} might be interconnected based on the available information.

2. **Identify and assess the nature of each interaction** between the terms (e.g., activation, inhibition, reactivation).

3. **Determine whether each interaction aligns with or contradicts the hypothesis** based on its nature:
   
   - **Alignment:** Interactions that **support** the hypothesis (e.g., inhibition of {b_term} by {c_term} leading to improvement in {a_term}).
   
   - **Contradiction:** Interactions that **oppose** the hypothesis (e.g., activation or reactivation of {b_term} by {c_term} leading to worsening of {a_term}).

4. **Synthesize the findings from multiple abstracts**, considering how the interactions fit together to support or refute the hypothesis: {hypothesis_template}.

5. **Provide a justification for your scoring decision** based on the analysis. **Explain your reasoning step-by-step** in terms understandable to an undergraduate biochemist. Focus on:
   
   - **Explaining the logical connections and the directionality of relationships**.
   
   - **Determining whether the interactions support or contradict the hypothesis**, using indirect evidence and logical inferences.
   
   - **Assessing if the interactions are beneficial or detrimental** to the proposed outcome.

6. **Be vigilant for any evidence that contradicts or challenges the hypothesis**. Address any contradictions explicitly in your reasoning.

7. **Verify any assumptions about the roles and effects** of {a_term}, {b_term}, and {c_term} as presented in the abstracts. **Avoid inferring effects not supported by the texts**.

8. In your final assessment, **explicitly cite the scoring guideline** that corresponds to your conclusion. **Explain why the evidence meets the criteria for that specific score**.

**Note:** Pay close attention to the nature of the interactions between entities. An interaction that **activates** a harmful process may be detrimental, while inhibition of a beneficial process may also be detrimental. **Consider whether such interactions contradict the hypothesis**.

**Definitions:**

- **Evidence:** Multiple sources agree and provide clear indications supporting a particular interaction and its effects.

- **Contradictory Evidence:** Evidence that directly opposes the hypothesis, showing that the proposed mechanism of action does not produce the expected outcome.

- **Directionality of Interaction:** The specific effect an interaction has (e.g., activation, inhibition, reactivation) and whether it supports or opposes the hypothesis.

**Checklist Before Finalizing Your Response:**

- Have you **addressed whether the interactions support or contradict the hypothesis**?

- Have you **assessed if the interactions are beneficial or detrimental**?

- Have you **explicitly cited the scoring guideline** that matches your conclusion?

- Have you **explained why the evidence meets the criteria** for the assigned score?

**Format your response as:**

Score: [Number] Point(s) - Reasoning: [Reasoning]

**Scoring Guidelines:**
{sg.abc_scoring_guidelines(a_term, b_term, c_term)}"""


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


