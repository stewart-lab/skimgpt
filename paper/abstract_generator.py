import openai
import random
import os
import argparse
import csv
import time


class AbstractGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def make_api_call(self, model, messages, temperature, max_tokens, retry_delay=10):
        while True:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    print("Empty response received from OpenAI API.")
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(retry_delay)

    def generate_abstracts(self, prompt_type, gene, disease):
        base_prompt = f"""Generate 5 medical research abstracts related to {gene} and {disease}. Each abstract should:
    1. Begin with a PMID number (format: PMID: ######) followed by the first sentence of the abstract.
    2. Use one of the provided structures (assigned below).
    3. Use formal, scientific language with specific medical terminology.
    4. Include details like study design, patient characteristics, sample sizes, and statistical measures.
    5. Be approximately 250-350 words long.
    6. End with a conclusion summarizing the findings and implications.
    7. Vary the opening sentences to avoid repetitive patterns.
    8. STRICTLY ADHERE to the specified sentiment ({prompt_type}) throughout the abstract."""

        structures = [
            "Context, Objective, Methods, Results, Conclusion",
            "Objective, Design, Setting and Participants, Main Outcomes and Measures, Results, Conclusions and Relevance",
            "Importance, Objective, Design/Setting/Participants, Interventions, Main Outcomes and Measures, Results, Conclusions",
            "Single paragraph with key findings, implications, experimental design, and limitations",
        ]

        type_specific_prompts = {
            "neutral": f"""The abstracts should mention both {gene} and {disease} without implying a positive or negative relationship. Use phrases like "no significant association", "results were inconclusive", or "further studies are needed". Avoid language that suggests benefits or risks. Report findings objectively without favoring any outcome.""",
            "positive": f"""Suggest {gene} could be an effective treatment for {disease}. Describe plausible studies with clear patient selection criteria, dosage, and treatment duration. Report positive outcomes using phrases like "significant improvement", "reduced risk", or "increased progression-free survival". Emphasize benefits while acknowledging the need for further research.""",
            "negative": f"""Suggest {gene} could exacerbate or negatively impact {disease}. Include details on adverse events, safety profiles, and reasons for treatment discontinuation. Report negative outcomes using phrases like "increased risk", "adverse effects", or "poorer outcomes". Highlight potential dangers while maintaining scientific objectivity.""",
        }

        prompt = f"{base_prompt}\n\nUse these structures for the abstracts:\n"
        for i, structure in enumerate(random.sample(structures, 4), 1):
            prompt += f"{i}. {structure}\n"
        prompt += f"\n{type_specific_prompts[prompt_type]}\n\nRemember to maintain the {prompt_type} sentiment consistently throughout all abstracts."

        model = "gpt-4"
        max_tokens = 2000
        temperature = 0.2
        retry_delay = 10

        messages = [
            {
                "role": "system",
                "content": f"You are a biomedical research analyst capable of generating synthetic, user-specific text. Your task is to create abstracts that strictly adhere to the {prompt_type} sentiment regarding the relationship between {gene} and {disease}.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            content = self.make_api_call(
                model, messages, temperature, max_tokens, retry_delay
            )
            return content
        except Exception as e:
            print(f"An error occurred for {gene} - {prompt_type}: {e}")
            return None

    def generate_cooccurrence_abstracts(
        self, entity_list_path, output_file, is_protein=False
    ):
        if is_protein:
            interaction_types = [
                "Direct Physical Interaction",
                "Functional Association",
                "Protein Complex Formation",
                "Signal Transduction Interaction",
                "Regulatory Interaction",
                "Enzyme-Substrate Interaction",
                "Protein Modification",
                "Genetic Interaction",
                "No Direct Interaction",
                "Conflicting Evidence",
            ]

            experimental_methods = [
                "Yeast Two-Hybrid Screening",
                "Co-Immunoprecipitation",
                "Bioluminescence Resonance Energy Transfer",
                "X-Ray Crystallography",
                "Surface Plasmon Resonance",
                "Affinity Purification Mass Spectrometry",
                "Genetic Interaction Mapping",
                "Fluorescence Resonance Energy Transfer",
                "Isothermal Titration Calorimetry",
                "Nuclear Magnetic Resonance Spectroscopy",
            ]

            non_interaction_types = [
                "Separate Pathways",
                "Non-Interacting Proteins",
                "Parallel Functions",
                "Different Cellular Compartments",
            ]
        else:
            interaction_types = [
                "Strong Interaction",
                "Moderate Interaction",
                "Mild Interaction",
                "Potential Interaction",
                "Conflicting Evidence",
                "No Significant Interaction",
                "Inconclusive",
                "Rare Interaction",
                "In Vitro Interaction",
                "Variable Interaction",
            ]

            non_interaction_types = [
                "Separate Contexts",
                "Non-Drug Mention",
                "List Mention",
                "Comparative Study",
                "Background Information",
            ]

        with open(entity_list_path, "r") as entity_file, open(
            output_file, "w", newline=""
        ) as csvfile:
            entities = [line.strip() for line in entity_file if line.strip()]
            fieldnames = [
                "Interaction Statement",
                "Abstract",
                "Interaction",
                "Interaction Type",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for _ in range(100):
                entity1, entity2 = random.sample(entities, 2)
                interaction = random.choice([0, 1])

                if interaction:
                    interaction_type = random.choice(interaction_types)
                    if is_protein:
                        experimental_method = random.choice(experimental_methods)
                        prompt = f"""Generate a scientific research abstract about the interaction between proteins {entity1} and {entity2}. The abstract should:
    1. Begin with a PMID number (format: PMID: ######) followed by the first sentence of the abstract.
    2. Use formal, scientific language with specific biological terminology.
    3. Include experimental methods such as {experimental_method}.
    4. Provide details like study design, organism, cell types, and experimental results.
    5. Be approximately 250-350 words long.
    6. End with a conclusion summarizing the findings and implications.

    Describe a study investigating the interaction between the two proteins, with the following outcome: {interaction_type.lower()}."""
                    else:
                        prompt = f"""Generate a medical research abstract about {entity1} and {entity2}. The abstract should:
    1. Begin with a PMID number (format: PMID: ######) followed by the first sentence of the abstract.
    2. Use formal, scientific language with specific medical terminology.
    3. Include details like study design, patient characteristics, sample sizes, and statistical measures.
    4. Be approximately 250-350 words long.
    5. End with a conclusion summarizing the findings and implications.

    Describe a study investigating the interaction between the two drugs, with the following outcome: {interaction_type.lower()}."""
                else:
                    interaction_type = random.choice(non_interaction_types)
                    if is_protein:
                        prompt = f"""Generate a scientific research abstract mentioning proteins {entity1} and {entity2}. The abstract should:
    1. Begin with a PMID number (format: PMID: ######) followed by the first sentence of the abstract.
    2. Use formal, scientific language with specific biological terminology.
    3. Include details like study focus, organism, and context of protein functions.
    4. Be approximately 250-350 words long.
    5. End with a conclusion summarizing the findings and implications.

    The abstract should be about {interaction_type.lower()}. Do not imply any interaction between the proteins."""
                    else:
                        prompt = f"""Generate a medical research abstract mentioning {entity1} and {entity2}. The abstract should:
    1. Begin with a PMID number (format: PMID: ######) followed by the first sentence of the abstract.
    2. Use formal, scientific language with specific medical terminology.
    3. Include details like study design, patient characteristics, sample sizes, and statistical measures.
    4. Be approximately 250-350 words long.
    5. End with a conclusion summarizing the findings and implications.

    The abstract should be about {interaction_type.lower()}. Do not imply any interaction between the drugs."""

                model = "gpt-4"
                max_tokens = 1000
                temperature = 0.2
                retry_delay = 10

                messages = [
                    {
                        "role": "system",
                        "content": "You are a biomedical research analyst capable of generating synthetic, user-specific text.",
                    },
                    {"role": "user", "content": prompt},
                ]

                try:
                    content = self.make_api_call(
                        model, messages, temperature, max_tokens, retry_delay
                    )
                    if not content:
                        print(
                            f"Empty response received from OpenAI API for {entity1} and {entity2}."
                        )
                        continue

                    if is_protein:
                        interaction_statement = f"There exists an interaction between protein {entity1} and protein {entity2}."
                    else:
                        interaction_statement = f"There exists an interaction between drug {entity1} and drug {entity2}."

                    writer.writerow(
                        {
                            "Interaction Statement": interaction_statement,
                            "Abstract": content,
                            "Interaction": interaction,
                            "Interaction Type": interaction_type,
                        }
                    )
                    print(
                        f"Generated abstract for {entity1} and {entity2} with interaction: {interaction}, type: {interaction_type}"
                    )

                except Exception as e:
                    print(f"An error occurred for {entity1} and {entity2}: {e}")

            print(f"Co-occurrence abstracts generated and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate medical or biological abstracts for entities."
    )
    parser.add_argument(
        "--drug_list", help="Path to the file containing the list of drugs"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all types of abstracts"
    )
    parser.add_argument(
        "--cooccurrence", action="store_true", help="Generate co-occurrence abstracts"
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="Generate protein-protein interaction abstracts",
    )
    args = parser.parse_args()

    abstract_generator = AbstractGenerator()

    if args.drug_list:
        if args.cooccurrence:
            if args.protein:
                output_file = (
                    "/w5home/jfreeman/kmGPT/src/protein_cooccurrence_abstracts.csv"
                )
                abstract_generator.generate_cooccurrence_abstracts(
                    args.drug_list, output_file, is_protein=True
                )
            else:
                output_file = "/w5home/jfreeman/kmGPT/src/cooccurrence_abstracts.csv"
                abstract_generator.generate_cooccurrence_abstracts(
                    args.drug_list, output_file, is_protein=False
                )
        else:
            disease_input = input("Enter the disease of interest: ")
            output_file = "/w5home/jfreeman/kmGPT/src/leakage.csv"
            process_drug_list(args.drug_list, disease_input, output_file)
            print(f"Abstracts generated and saved to {output_file}")
    else:
        while True:
            prompt_input = input(
                "Enter the type of prompt you're interested in \n 1 = Neutral \n 2 = Positive \n 3 = Negative \n Type: "
            )
            if prompt_input in ["1", "2", "3"]:
                break
            else:
                print("Invalid input, please choose 1, 2, or 3.")

        prompt_types = {"1": "neutral", "2": "positive", "3": "negative"}

        gene_input = input("Enter the gene of interest: ")
        disease_input = input("Enter the disease of interest: ")
        print("Generating abstracts...")

        if args.all:
            for prompt_type in prompt_types.values():
                content = abstract_generator.generate_abstracts(
                    prompt_type, gene_input, disease_input
                )
                print(f"\n{prompt_type.capitalize()} abstracts:\n{content}")
        else:
            content = abstract_generator.generate_abstracts(
                prompt_types[prompt_input], gene_input, disease_input
            )
            print(f"\nGenerated abstracts:\n{content}")


if __name__ == "__main__":
    main()