import skimgpt.scoring_guidelines as sg
from skimgpt.utils import extract_pmids as _extract_pmids

def extract_pmids(consolidated_abstracts):
    pmids = _extract_pmids(consolidated_abstracts)
    pmid_list = ", ".join(pmids) if pmids else "None found"
    return pmid_list

def km_with_gpt(b_term, a_term, hypothesis_template, consolidated_abstracts):
    pmid_list = extract_pmids(consolidated_abstracts)

    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Available PMIDs for Citation: {pmid_list}

Assessment Task:
Evaluate the degree of support for the hypothesis, which posits a significant interaction between {a_term} and {b_term}. 
The texts provided above come from PubMed and each abstract will include only {a_term} and {b_term}. 
The texts need to be your only source of information for arriving at your classification result. 

IMPORTANT: You must only cite PMIDs that are explicitly provided in the abstracts above. Do not reference or cite any external literature or PMIDs not in the list above.

Hypothesis:
{hypothesis_template}

Instructions:
1. Review each abstract to understand how {a_term} and {b_term} might be interconnected based on the available information.
2. Analyze the presence and implications of the term pairing ({a_term} + {b_term}) in the context of the hypothesis.
3. Synthesize the findings from multiple texts. Consider how the pieces fit together to support or refute the hypothesis: {hypothesis_template}. Remember, no single text may be conclusive.
4. Provide a justification for your scoring decision based on the analysis. Explain your reasoning step-by-step in terms understandable to an undergraduate biochemist. Focus on explaining the logical connections and the directionality of relationships.
5. Cite specific texts from your set of abstracts to support your arguments. Only cite PMIDs from the list above, and clearly reference these citations in your reasoning using the format "PMID: XXXXX".

Scoring Guidelines:
{sg.ab_scoring_guidelines(a_term, b_term)}

Output policy:
- Return ONLY a single JSON object matching the schema below. The API's response_format guarantees raw JSON output; do NOT wrap it in Markdown code fences.
- JSON schema (for reference; do not print this schema):
{km_with_gpt_json_schema()}
"""

def km_with_gpt_direct_comp(hypothesis_1, hypothesis_2, consolidated_abstracts):
    pmid_list = extract_pmids(consolidated_abstracts)

    return f"""

Biomedical Abstracts (verbatim):
{consolidated_abstracts}

Available PMIDs for citation: {pmid_list}

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Task:
Compare Hypothesis 1 vs Hypothesis 2 using ONLY the abstracts above. Assign each abstract
a SINGLE label from {{supports_H1, supports_H2, both, neither, inconclusive}}. Then produce
the 4-key tally, assign a continuous 0–100 score, and choose a decision.

Label definitions — read these carefully, they are the most common source of error:

  • supports_H1 — the abstract presents evidence that directly aligns with Hypothesis 1
    AS STATED, including its causal direction. Reporting that the relationship is widely
    discussed, believed, hypothesized, or feared is NOT support — only evidence that the
    claim itself holds.

  • supports_H2 — same standard as supports_H1, but for Hypothesis 2.

  • both — independent evidence in the abstract aligns with each hypothesis. This is rare
    and should only be used when the abstract makes affirmative claims that match each
    hypothesis's direction. Do not use 'both' for abstracts that merely mention both topics.

  • neither — the abstract addresses the topic but the evidence does NOT support either
    hypothesis as stated. Use this when:
      – the abstract reports the OPPOSITE causal direction to a hypothesis (a treatment
        study showing "X reduces Y" does not support "X causes Y" — label it neither);
      – the abstract reports beliefs, opinions, surveys, attitudes, perceptions, fears,
        or media coverage about a claim rather than evidence of the claim ("N% of parents
        believe X causes Y" is NOT evidence that X causes Y);
      – the abstract co-mentions the terms without endorsing either causal claim.

  • inconclusive — the abstract does not bear on either hypothesis, or its evidence is
    genuinely ambiguous / mixed.

Instructions:
1. For each abstract, decide on a single label by asking, in order:
   (a) Is there direct evidence aligning with H1 in the stated direction? With H2?
   (b) If yes to both independently, label = both.
   (c) If yes to exactly one, label = supports_H1 or supports_H2.
   (d) If the abstract addresses the topic but the evidence does NOT support either
       hypothesis (opposite direction, belief/attitude reporting, topical co-mention
       without endorsement) — label = neither.
   (e) Otherwise — label = inconclusive.
2. Topical match alone is NOT support. An abstract that mentions both terms in a
   hypothesis but does not affirmatively assert the causal claim is at best inconclusive,
   and is neither when it addresses the topic without endorsing the claim.
3. Assess directionality. A treatment study showing "X reduces / inhibits / suppresses Y"
   does NOT support "X causes Y" — they are opposite causal directions; label that
   abstract neither (or supports_H2 only if H2 is the opposite causal claim).
4. Most abstracts come from a single B-term's intersection list and will not bear on the
   other hypothesis. That alone does not make them supports_H1 or supports_H2 — apply the
   same evidence standard to the hypothesis they do address.
5. Cite only PMIDs from the list above when justifying your labels in evidence and
   score_rationale. Do not reference outside literature.

Worked examples:

  • Hypotheses: H1 = "{{a_term}} is caused by inhibition of {{b_term1}}",
               H2 = "{{a_term}} is caused by inhibition of {{b_term2}}".
    Abstract (paraphrased): "Compound X inhibits {{b_term1}} and reduces {{a_term}} growth
    in xenograft models, supporting {{b_term1}} inhibition as a therapeutic strategy."
    Correct label: neither.
    Why: the study shows that inhibiting {{b_term1}} REDUCES {{a_term}} — the opposite of
    H1's "inhibition of {{b_term1}} CAUSES {{a_term}}". It does not address {{b_term2}}.

  • Same hypotheses.
    Abstract (paraphrased): "{{b_term1}} and {{b_term2}} are both expressed in {{a_term}}
    cell lines; the abstract reports expression levels but no functional or causal claim."
    Correct label: inconclusive.
    Why: topical co-mention without any causal evidence.

  • Hypothesis 1: "Autism is caused by Genetic predisposition";
    Hypothesis 2: "Autism is caused by Vaccines".
    Abstract (paraphrased): "16.5% of caregivers in a parent survey endorsed vaccines as
    perceived causes of autism; we analyse demographic correlates of this belief."
    Correct label: neither.
    Why: the abstract reports BELIEFS about a causal claim, not evidence for it. Belief-
    prevalence surveys, attitude studies, and media-coverage analyses are not evidence of
    causation — they are evidence about what people think.

  • Same hypotheses.
    Abstract (paraphrased): "Whole-exome sequencing identifies de novo loss-of-function
    variants in CHD8 enriched in autism probands relative to unaffected siblings."
    Correct label: supports_H1.
    Why: direct genetic evidence aligned with H1's causal direction.

Continuous Scoring Guidelines (0–100):
{sg.cont_ab_direct_comp_scoring_guidelines(hypothesis_1, hypothesis_2)}

Output policy:
- Return ONLY a single JSON object matching the schema below. The API's response_format guarantees raw JSON output; do NOT wrap it in Markdown code fences.
- JSON schema (for reference; do not print this schema):
{km_with_gpt_direct_comp_json_schema()}
"""

def skim_with_gpt_ac(a_term, hypothesis_template, consolidated_abstracts, c_term):
    pmid_list = extract_pmids(consolidated_abstracts)

    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Available PMIDs for Citation: {pmid_list}

Assessment Task:
Evaluate the degree of support for the hypothesis: {hypothesis_template}, which posits an interaction between {a_term} and {c_term}. Use only the provided abstracts from PubMed that mention {a_term} and {c_term} to inform your analysis.

IMPORTANT: You must only cite PMIDs that are explicitly provided in the abstracts above. Do not reference or cite any external literature or PMIDs not in the list above.

Your evaluation should:

• Integrate evidence from all abstracts to discern how {a_term} and {c_term} might be interconnected.
• Employ logical reasoning, drawing logical inferences where appropriate, based on the interaction between {a_term} and {c_term}.
• Assess the nature and directionality of the interaction, determining whether they are beneficial or detrimental to the proposed outcome in the hypothesis.
• Be vigilant for any evidence that contradicts or challenges the hypothesis, explicitly addressing any contradictions in your reasoning.
• Avoid inferring effects not supported by the texts, ensuring that all conclusions are grounded in the provided information.
• Ensure that your scoring reflects an unbiased assessment based solely on the provided evidence, considering logical inferences from evidence.
• When citing evidence, use the PMID format (e.g., "PMID: 12345678") and only cite PMIDs from the list above.

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

**Scoring Guidelines:**
{sg.ac_scoring_guidelines(a_term, c_term)}

Output policy:
- Return ONLY a single JSON object matching the schema below. The API's response_format guarantees raw JSON output; do NOT wrap it in Markdown code fences.
- JSON schema (for reference; do not print this schema):
{skim_with_gpt_json_schema()}
"""


def skim_with_gpt(b_term, a_term, hypothesis_template, consolidated_abstracts, c_term):
    pmid_list = extract_pmids(consolidated_abstracts)

    return f"""Biomedical Abstracts for Analysis:
{consolidated_abstracts}

Available PMIDs for Citation: {pmid_list}

Assessment Task:
Evaluate the degree of support for the hypothesis, which posits an interaction between {a_term} and {c_term} through their own interactions with {b_term}. Use only the provided abstracts from PubMed, each containing only two of the three terms at a time ({a_term} + {b_term}, or {b_term} + {c_term}), to inform your analysis.

IMPORTANT: You must only cite PMIDs that are explicitly provided in the abstracts above. Do not reference or cite any external literature or PMIDs not in the list above.

Your evaluation should:

- **Integrate evidence from all abstracts** to discern how {a_term}, {b_term}, and {c_term} might be interconnected.

- **Assess the directionality and nature of each interaction** (e.g., activation, inhibition, reactivation) to determine whether it **aligns with** or **contradicts** the hypothesis.

- **Identify and analyze any opposing mechanisms** where one interaction may negate the effects of another.

- **Employ logical reasoning**, drawing logical inferences where appropriate, based on {a_term}-{b_term} and {b_term}-{c_term} interactions.

- **Assess the nature and directionality of the interactions**, determining whether they are beneficial or detrimental to the proposed outcome in the hypothesis.

- **Be vigilant for any evidence that contradicts or challenges the hypothesis**, explicitly addressing any contradictions in your reasoning.

- **Avoid inferring effects not supported by the texts**, ensuring that all conclusions are grounded in the provided information.

- **Ensure that your scoring reflects an unbiased assessment based solely on the provided evidence**, considering logical inferences from indirect evidence.

- **When citing evidence, use the PMID format (e.g., "PMID: 12345678") and only cite PMIDs from the list above.**

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

**Scoring Guidelines:**
{sg.abc_scoring_guidelines(a_term, b_term, c_term)}

Output policy:
- Return ONLY a single JSON object matching the schema below. The API's response_format guarantees raw JSON output; do NOT wrap it in Markdown code fences.
- JSON schema (for reference; do not print this schema):
{skim_with_gpt_json_schema()}
"""


def km_with_gpt_direct_comp_json_schema():
    # Property order matters: under OpenAI Structured Outputs strict mode the model
    # emits fields in declaration order, which we use as a forcing function for
    # chain-of-thought — `evidence` is committed BEFORE the collapsed `label`, so
    # the bucket name has to follow from a cited quote rather than the other way
    # around. Strict mode also requires every listed property to be in `required`.
    return {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "per_abstract": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pmid": {"type": "string", "pattern": "^[0-9]+$"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                        "minItems": 1
                    },
                    "label": {
                        "type": "string",
                        "enum": ["supports_H1","supports_H2","both","neither","inconclusive"]
                    },
                },
                "required": ["pmid","evidence","label"]
            }
        },
        "score_rationale": {
            "type": "array",
            "items": {"type": "string", "maxLength": 1000, "description": "Evidence-based rationale with PMIDs, e.g., 'Two RCTs report X (PMID: 123, 456)' that uses the scoring guidelines to justify the score."},
            "minItems": 1,
            "maxItems": 6
        },
        "tallies": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "support_H1": {"type": "integer", "minimum": 0},
                "support_H2": {"type": "integer", "minimum": 0},
                "both": {"type": "integer", "minimum": 0},
                "neither_or_inconclusive": {"type": "integer", "minimum": 0}
            },
            "required": ["support_H1","support_H2","both","neither_or_inconclusive"]
        },
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "decision": {"type": "string", "enum": ["H1","H2","tie","insufficient_evidence"]}
    },
    "required": ["per_abstract","score_rationale","tallies","score","decision"]
    }


def km_with_gpt_direct_comp_system_instructions():
    return """\
You compare two competing biomedical hypotheses using ONLY the provided PubMed abstracts.

Rules:
- Use ONLY the provided abstracts. Do not use outside knowledge or any PMIDs not provided.
- Every claim must map to at least one provided PMID from the input set.
- Output MUST be a single JSON object matching the required schema exactly. response_format guarantees raw JSON; do NOT wrap the output in Markdown code fences.
- Follow the provided continuous scoring guidelines verbatim (0..100 scale). Do not derive or use any explicit scoring formula.

Per-abstract reasoning order — emit fields in the order the schema declares them; each field commits you before the next one:
1. `pmid` — identifies the abstract.
2. `evidence` — REQUIRED, at least one short verbatim or near-verbatim quote (≤300 chars each) from THIS abstract that bears on either hypothesis. Cite the quote BEFORE you pick a label, and pick the label that follows from the quote — not the other way around.
3. `label` — one bucket from {supports_H1, supports_H2, both, neither, inconclusive}, picked on the basis of the evidence you just cited.

Label definitions:
- supports_H1: the cited evidence directly aligns with Hypothesis 1 AS STATED, including its causal direction.
- supports_H2: the cited evidence directly aligns with Hypothesis 2 AS STATED, including its causal direction.
- both: independent evidence in the abstract aligns with both H1 and H2 — rare; use only for genuine dual-support, not for topical co-mention.
- neither: the abstract addresses the topic but the evidence does NOT support either hypothesis. This is the correct label for:
    (a) evidence in the opposite causal direction (a treatment study showing "X reduces Y" does not support "X causes Y" — label it neither);
    (b) reports of beliefs, opinions, surveys, attitudes, perceptions, fears, or media coverage about a claim rather than evidence of the claim ("N% of parents believe X causes Y" is NOT evidence that X causes Y — label it neither);
    (c) topical co-mention without affirmative endorsement of either claim.
- inconclusive: the abstract does not bear on either hypothesis, or its evidence is genuinely ambiguous / mixed.
- IMPORTANT: topical match alone is NOT supports. Apply the same evidence standard whether the abstract came from H1's intersection list or H2's. Do not collapse refute-shaped evidence (opposite causal direction) into 'supports' for the other hypothesis unless that other hypothesis is itself the opposite causal claim.

Self-check before you finalise each entry: re-read your `evidence` quote. If you cannot defend the chosen `label` from that quote alone, change the label (most often to `neither` or `inconclusive`).

Tallies — four counts, equal to the number of per_abstract entries with each label
(neither_or_inconclusive merges 'neither' and 'inconclusive'):
- support_H1, support_H2, both, neither_or_inconclusive.
- The four tallies MUST sum to the total number of abstracts (len(per_abstract)).
- support_H1 = count of entries with label == "supports_H1"; support_H2 = count of "supports_H2"; both = count of "both"; neither_or_inconclusive = count of "neither" + count of "inconclusive".

The final 'decision' is one of: H1, H2, tie, insufficient_evidence; choose based on the scoring guidelines and the provided evidence set only.
"""


def km_with_gpt_json_schema():
    return {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "per_abstract": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pmid": {"type": "string", "pattern": "^[0-9]+$"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                        "minItems": 1
                    },
                    "label": {
                        "type": "string",
                        "enum": ["supports","refutes","inconclusive"]
                    },
                },
                "required": ["pmid","evidence","label"]
            }
        },
        "score_rationale": {
            "type": "array",
            "items": {"type": "string", "maxLength": 1000, "description": "Evidence-based rationale with PMIDs that uses the scoring guidelines to justify the score."},
            "minItems": 1,
            "maxItems": 6
        },
        "tallies": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "support": {"type": "integer", "minimum": 0},
                "refute": {"type": "integer", "minimum": 0},
                "inconclusive": {"type": "integer", "minimum": 0}
            },
            "required": ["support","refute","inconclusive"]
        },
        "score": {"type": "number", "minimum": -2, "maximum": 2},
        "decision": {"type": "string", "enum": ["supports","refutes","insufficient_evidence"]}
    },
    "required": ["per_abstract","score_rationale","tallies","score","decision"]
    }


def km_with_gpt_system_instructions():
    return """\
You evaluate a single biomedical hypothesis using ONLY the provided PubMed abstracts.
Rules:
- Use ONLY the provided abstracts. Do not use outside knowledge or any PMIDs not provided.
- Every claim must map to at least one provided PMID from the input set.
- Output MUST be a single JSON object matching the required schema exactly. response_format guarantees raw JSON; do NOT wrap the output in Markdown code fences.
- Follow the provided discrete scoring guidelines verbatim (-2..+2). Do not derive or use any explicit scoring formula.
- Tally counts as requested: number supporting, number refuting, and number that are inconclusive.
- Labels per abstract: supports, refutes, inconclusive.
- Per-abstract reasoning order: cite `evidence` first (≥1 short quote, ≤300 chars), THEN pick the `label` that follows from the cited quote.
- The final 'decision' is one of: supports, refutes, insufficient_evidence; choose based on the guidelines and the provided evidence set only.
"""


def skim_with_gpt_json_schema():
    return {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "per_abstract": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pmid": {"type": "string", "pattern": "^[0-9]+$"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                        "minItems": 1
                    },
                    "label": {
                        "type": "string",
                        "enum": ["supports","refutes","inconclusive"]
                    },
                },
                "required": ["pmid","evidence","label"]
            }
        },
        "score_rationale": {
            "type": "array",
            "items": {"type": "string", "maxLength": 1000, "description": "Evidence-based rationale with PMIDs that uses the scoring guidelines to justify the score."},
            "minItems": 1,
            "maxItems": 6
        },
        "tallies": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "support": {"type": "integer", "minimum": 0},
                "refute": {"type": "integer", "minimum": 0},
                "inconclusive": {"type": "integer", "minimum": 0}
            },
            "required": ["support","refute","inconclusive"]
        },
        "score": {"type": "number", "minimum": -2, "maximum": 2},
        "decision": {"type": "string", "enum": ["supports","refutes","insufficient_evidence"]}
    },
    "required": ["per_abstract","score_rationale","tallies","score","decision"]
    }


def skim_with_gpt_system_instructions():
    return """\
You evaluate a single mediated biomedical hypothesis (AB/BC/AC as applicable) using ONLY the provided PubMed abstracts.
Rules:
- Use ONLY the provided abstracts. Do not use outside knowledge or any PMIDs not provided.
- Every claim must map to at least one provided PMID from the input set.
- Output MUST be a single JSON object matching the required schema exactly. response_format guarantees raw JSON; do NOT wrap the output in Markdown code fences.
- Follow the provided discrete scoring guidelines verbatim (-2..+2). Do not derive or use any explicit scoring formula.
- Tally counts as requested: number supporting, number refuting, and number that are inconclusive.
- Labels per abstract: supports, refutes, inconclusive.
- Per-abstract reasoning order: cite `evidence` first (≥1 short quote, ≤300 chars), THEN pick the `label` that follows from the cited quote.
- The final 'decision' is one of: supports, refutes, insufficient_evidence; choose based on the guidelines and the provided evidence set only.
"""