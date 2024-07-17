def gpt_customized_scoring_system(b_term, a_term):
    return (
        "Scoring Guidelines:\n"
        f"   - -1 Point: There is sufficient direct evidence in the provided texts suggesting that {b_term} does not effectively alleviate or target key pathogenic mechanisms of {a_term}, or that it does not offer therapeutic benefits or slow disease progression.\n"
        f"   - 0 Points: The provided texts either offer no direct evidence or only indirect evidence regarding the effectiveness of {b_term} in alleviating or targeting key pathogenic mechanisms of {a_term}. This includes situations where the evidence is related but does not specifically address the hypothesis or focuses on associated subjects without directly supporting or refuting the primary subject.\n"
        f"   - 1 Point: There is weak but direct evidence supporting the hypothesis that {b_term} may alleviate or target key pathogenic mechanisms of {a_term}, potentially offering therapeutic benefits or slowing disease progression. The studies or findings may have limitations or provide mixed results.\n"
        f"   - 2 Points: The evidence supporting the hypothesis that {b_term} can alleviate or target key pathogenic mechanisms of {a_term} is moderate and direct in the provided texts. The studies are well conducted and provide fairly consistent results.\n"
        f"   - 3 Points: There is strong and direct evidence in the provided texts supporting the hypothesis that {b_term} effectively alleviates or targets key pathogenic mechanisms of {a_term}, offering therapeutic benefits or slowing disease progression. Multiple high-quality studies provide consistent and direct support.\n\n"
    )


def original_scoring_guidelines():
    return (
        "   - -1 Point: There is sufficient evidence in the provided texts to say that the hypothesis is probably not true. "
        "The evidence described in the texts directly contradict the hypothesis.\n"
        "   - 0 Points: There is no evidence to support the hypothesis in the provided texts. "
        "The experiments described in the texts may or may not be well conducted, but they do not have any bearing on the hypothesis.\n"
        "   - 1 Point: Evidence that supports the hypothesis is weak in the provided texts. "
        "The experiments seem poorly conducted, or there is mixed evidence that the hypothesis is true.\n"
        "   - 2 Points: Evidence that supports the hypothesis is moderate in the provided texts. "
        "The experiments are well conducted, and there is a fair amount of evidence in the texts that the hypothesis may be true.\n"
        "   - 3 Points: Evidence that supports the hypothesis is strong in the provided texts. "
        "There is a large amount of direct evidence in the texts that the hypothesis is true.\n\n"
    )


def updated_KM_scoring_guidelines():
    return (
        "   - -2 Points: The evidence from the PubMed abstracts strongly contradicts the hypothesis. "
        "This includes multiple high-quality studies or meta-analyses that provide robust, "
        "consistent evidence against the hypothesis with statistically significant results "
        "and no notable methodological flaws.\n"
        
        "   - -1 Point: The evidence in the PubMed abstracts moderately contradicts the hypothesis. "
        "This includes studies or analyses that generally show negative results towards the hypothesis, "
        "with minor limitations in study design, sample size, or statistical analysis, but "
        "the overall body of evidence leans against the hypothesis.\n"
        
        "   - 0 Points: The evidence from the PubMed abstracts is inconclusive or neutral regarding the hypothesis. "
        "This score applies to situations where studies show mixed results, or the existing research "
        "is characterized by a significant lack of data, insufficient statistical power, or methodological "
        "variance, indicating an urgent need for further high-quality research.\n"
        
        "   - 1 Point: There is weak support for the hypothesis in the PubMed abstracts. "
        "This includes initial findings or smaller-scale studies that suggest a positive correlation or effect, "
        "with notable limitations like small sample sizes, the need for more extensive replication, "
        "or preliminary methodological concerns that call for cautious interpretation of results.\n"
        
        "   - 2 Points: There is strong support for the hypothesis in the PubMed abstracts. "
        "Evidence comes from rigorously conducted research, including well-designed studies or meta-analyses, "
        "demonstrating clear, statistically significant results in favor of the hypothesis with high "
        "methodological quality and minimal biases.\n\n"
    )


def updated_scoring_guidelines_for_mediated_relationships():
    return (
        "   - -2 Points: Strong evidence contradicts the hypothesis that an interaction with b_term facilitates a beneficial "
        "relationship between a_term and c_term. This includes multiple, high-quality studies or meta-analyses showing robust, "
        "consistent evidence against either the connection between a_term and b_term, the connection between b_term and c_term, "
        "or the overall proposed mediated relationship, with statistically significant results and rigorous methodological quality.\n"
        
        "   - -1 Point: Moderate evidence contradicts the hypothesis. Studies or analyses generally show negative results for "
        "the connections between a_term and b_term or b_term and c_term, or they question the efficacy of the mediated relationship. "
        "Some limitations in study design, sample size, or statistical analysis may exist, but the preponderance of evidence leans "
        "against the hypothesis.\n"
        
        "   - 0 Points: The evidence is inconclusive or neutral regarding the hypothesis. Results are mixed, or the research does "
        "not directly address the mediated relationship between a_term, b_term, and c_term, showing a significant gap in the literature. "
        "This indicates a need for more targeted, high-quality research to clarify the proposed interactions.\n"
        
        "   - 1 Point: There is weak support for the hypothesis. Initial findings, possibly from smaller studies or indirect evidence, "
        "suggest a potential mediated relationship between a_term and c_term via b_term, with limitations such as small sample sizes, "
        "indirect evidence linkage, or preliminary methodological concerns. These findings necessitate cautious interpretation and further "
        "research for validation.\n"
        
        "   - 2 Points: Strong support exists for the hypothesis. Evidence from well-conducted research supports the mediated relationship "
        "between a_term and c_term via b_term, showing clear, statistically significant results favoring the hypothesis, with high "
        "methodological quality and minimal biases. This may include studies directly addressing the mediated pathway or converging "
        "evidence from related research areas.\n\n"
    )


def updated_scoring_guidelines_for_mediated_relationships(hypothesis):
    """
    Provides scoring guidelines for assessing the validity of a given hypothesis about mediated relationships.
    
    Parameters:
    hypothesis (str): A description of the hypothesis to be evaluated, typically involving a mediated relationship between terms.

    Returns:
    str: A set of scoring guidelines based on the level of evidence supporting or contradicting the hypothesis.
    """
    return (
        f"   - -2 Points: Strong evidence contradicts the hypothesis that {hypothesis}. This includes multiple, high-quality studies or "
        "meta-analyses showing robust, consistent evidence against the proposed relationship, with statistically significant results "
        "and rigorous methodological quality.\n"
        
        f"   - -1 Point: Moderate evidence contradicts the hypothesis that {hypothesis}. Studies or analyses generally show negative "
        "results for the proposed relationship, or they question its efficacy. Some limitations in study design, sample size, or "
        "statistical analysis may exist, but the preponderance of evidence leans against the hypothesis.\n"
        
        f"   - 0 Points: The evidence is inconclusive or neutral regarding the hypothesis that {hypothesis}. Results are mixed, or the "
        "research does not directly address the proposed relationship, showing a significant gap in the literature. This indicates a need "
        "for more targeted, high-quality research to clarify the proposed interactions.\n"
        
        f"   - 1 Point: There is weak support for the hypothesis that {hypothesis}. Initial findings, possibly from smaller studies or "
        "indirect evidence, suggest a potential relationship, with limitations such as small sample sizes, indirect evidence linkage, or "
        "preliminary methodological concerns. These findings necessitate cautious interpretation and further research for validation.\n"
        
        f"   - 2 Points: Strong support exists for the hypothesis that {hypothesis}. Evidence from well-conducted research supports the "
        "proposed relationship, showing clear, statistically significant results favoring the hypothesis, with high methodological quality "
        "and minimal biases. This may include studies directly addressing the pathway or converging evidence from related research areas.\n\n"
    )



def skim_old(b_term, a_term, c_term):
    """
    Provides scoring guidelines for assessing the validity of a given hypothesis about mediated relationships,
    with a focus on the indirect relationships between A and B, B and C, and the co-occurrence of A and C.
    
    Parameters:
    b_term, a_term, c_term (str): Terms representing the entities involved in the hypothesis to be evaluated.

    Returns:
    str: A set of scoring guidelines based on the level of evidence supporting or contradicting the hypothesis, with particular
         emphasis on the indirect relationships and co-occurrence evidence.
    """
    return (
        f"   - -2 Points: Strong evidence directly contradicts the hypothesis or robust evidence is against the "
        f"effectiveness of the indirect relationship between {a_term} and {b_term}, {b_term} and {c_term}, or lack of co-occurrence significance of {a_term} and {c_term} in the context of {b_term}, with rigorous methodological quality.\n"
        
        f"   - -1 Point: Moderate evidence raises substantial doubts about the hypothesis. Studies typically show weak or "
        f"negative interactions between {a_term} and {b_term}, {b_term} and {c_term}, or insufficient co-occurrence of {a_term} and {c_term} in relevant contexts, albeit with some "
        f"methodological limitations or biases.\n"
        
        f"   - 0 Points: The evidence is inconclusive, neutral or irrelevant. Research may show mixed results "
        f"for the interactions between {a_term} and {b_term}, {b_term} and {c_term}, or, when present, of {a_term} and {c_term}, indicating gaps in the literature "
        f"and the need for further, targeted research.\n"
        
        f"   - 1 Point: Initial evidence supports the hypothesis, showing weak to moderate support for the interactions "
        f"between {a_term} and {b_term}, {b_term} and {c_term}, and some evidence of {a_term} and {c_term} co-occurring in meaningful contexts. However, limitations such as small "
        f"sample sizes or preliminary methodological concerns necessitate cautious interpretation.\n"
        
        f"   - 2 Points: Strong evidence supports the hypothesis, with clear results demonstrating "
        f"effective interactions between {a_term} and {b_term}, {b_term} and {c_term} involving large sample sizes, strong methodology, and, when present, clear support of the {a_term} and {c_term} relationship with respect to the hypothesis. Evidence is of high methodological quality, with minimal biases.\n\n"
    )

def skim_classify():
    return (
                "-2 Points: The hypothesis can very likely be refuted based on the provided texts. There is a substantial amount of evidence in the provided texts that the hypothesis is false.\n" 
                "-1 Point: The hypothesis can likely be refuted based on the provided texts. There is some direct or indirect evidence in the provided texts that the hypothesis is false.\n" 
                "0 Points: The hypothesis is not supported or directly refuted by the provided texts. There is either no evidence to support or directly refute the hypothesis in the provided texts, or it is inconclusive, neutral, or irrelevant.\n" 
                "1 Point: The hypothesis is likely true based on the provided texts. There is some direct or indirect evidence in the provided texts that the hypothesis is true.\n" 
                "2 Points: The hypothesis is very likely true based on the provided texts. There is a good amount of direct or indirect evidence in the provided texts that the hypothesis is true.\n"
    )

def skim():
    return (
        "-2 Points: The hypothesis is strongly refuted by the provided texts. There is substantial evidence suggesting the hypothesis is false, offering little to no potential for further research or exploration.\n"
        "-1 Point: The hypothesis is likely refuted based on the provided texts. The evidence leans towards negating the hypothesis, diminishing its potential for further valuable inquiry.\n"
        "0 Points: The hypothesis is neither strongly supported nor refuted by the provided texts. The evidence is inconclusive, making its potential for further research unclear.\n"
        "1 Point: The hypothesis is likely true based on the provided texts. The available evidence is promising and suggests that further exploration could yield valuable insights or discoveries.\n"
        "2 Points: The hypothesis is strongly supported by the provided texts. The evidence is compelling and indicates significant potential for novel discoveries or important theoretical advancements.\n"
    )



def skim_finer():
    return (
        "-2.0 Points: The hypothesis can very likely be refuted based on the provided texts. There is substantial evidence in the provided texts that the hypothesis is false.\n"
        "-1.5 Points: The hypothesis can probably be refuted based on the provided texts. There is considerable evidence in the provided texts that the hypothesis is less likely to be true.\n"
        "-1.0 Point: The hypothesis is likely refutable based on the provided texts. There is some direct or indirect evidence in the provided texts that the hypothesis is false.\n"
        "-0.5 Point: The evidence against the hypothesis in the provided texts is slight, but it slightly leans towards refutation rather than support.\n"
        "0 Points: The hypothesis is not supported or directly refuted by the provided texts. There is either no evidence to support or directly refute the hypothesis in the provided texts, or it is inconclusive, neutral, or irrelevant.\n"
        "0.5 Point: The evidence supporting the hypothesis in the provided texts is slight, but it slightly leans towards support rather than refutation.\n"
        "1.0 Point: The hypothesis is likely true based on the provided texts. There is some direct or indirect evidence in the provided texts that the hypothesis is true.\n"
        "1.5 Points: The hypothesis is probably true based on the provided texts. There is considerable evidence in the provided texts that the hypothesis is more likely to be true.\n"
        "2.0 Points: The hypothesis is very likely true based on the provided texts. There is a good amount of direct or indirect evidence in the provided texts that the hypothesis is true.\n"
    )


