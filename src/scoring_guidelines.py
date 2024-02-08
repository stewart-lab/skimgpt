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
