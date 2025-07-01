#!/usr/bin/env python3
# scores.py

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse
from collections import defaultdict
from typing import List, Dict
import pandas as pd
from matplotlib.lines import Line2D

# Scores for each person (Standard - A-B-C Connection)
ron_scores = [-1, 1, 2, -2, 0, 2, 1, 1, 0, 0, -2, 0, 0, -1]
rob_scores = [-2, 1, 2, -2, 0, 2, 1, 0, 0, -2, -2, 1, 0, -1]
jack_scores = [-2, 1, 2, -2, 0, 2, 0, 1, 1, -2, -1, 1, 0, -2]
ishaan_scores = [-2, 2, 2, -2, 0, 2, 1, 1, 1, -2, -2, -1, 0, -1]
ishaan_scores_abc_prerel = [-2, 1, 1, 0, 0, -1, -2, 0, 0, 1]
ron_scores_abc_prerel = [-1, 1, 1, 0, 0, 0, -1, 0, 0, 0]
rob_scores_abc_prerel = [-2, 1, 0, 0, -1, 0, -1, -1, 0, 1]
jack_scores_abc_prerel = [-1, 1, 0, 0, 0, 0, -1, 0, 0, 0]
# Scores for each person (A-C Connection)
ron_scores_ac = [0, -1, 2, -2, -2, 2, 0, 1, 1, -2, 0, 0, 2, 1]
rob_scores_ac = [0, 0, 2, -2, -2, 2, -1, -1, 1, -2, 0, 0, 2, 1]
jack_scores_ac = [0, 0, 2, -2, -2, 2, -1, 0, 1, -2, 0, 0, 2, 1]
ishaan_scores_ac = [0, 0, 2, -2, -2, 2, 0, 0, 1, -2, 0, 0, 2, 1]
ishaan_scores_ac_prerel = [0, 0, 1, 0, 0, 0, 1, 2, -2, 0]
ron_scores_ac_prerel = [0, 0, 1, 0, 0, 0, 1, 2, -2, 0]
rob_scores_ac_prerel = [-2, -2, 0, 0, 0, 0, 1, 2, -2, 0]
jack_scores_ac_prerel = [-1, 0, 0, 0, 0, 0, 1, 2, -2, 0]
# Labels for each score point
labels_abc = [
    "Alzheimer's - BCHE - 2 pam",
    "Alzheimer's - FYN - pp2",
    "Breast cancer - CDK4 - abemaciclib",
    "Breast cancer - ESR1 - estrogens",
    "Diabetes - AHR - stavudine",
    "Diabetes - GIPR - tirzepatide",
    "Diabetes - LOX - zileuton",
    "Heart disease - EGFR - erlotinib",
    "Heart disease - RAF1 - gefitinib",
    "Lupus - NAT2 - isoniazid",
    "Non-alcoholic fatty liver disease - AHR - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - ULK1 - pp242",
    "Pancreatic cancer - CCK - gant61",
    "Pancreatic cancer - NAMPT - ginsenoside rb1",
]

# Labels for A-C connection (removing gene)
labels_ac = [
    "Alzheimer's - 2 pam",
    "Alzheimer's - pp2",
    "Breast cancer - abemaciclib",
    "Breast cancer - estrogens",
    "Diabetes - stavudine",
    "Diabetes - tirzepatide",
    "Diabetes - zileuton",
    "Heart disease - erlotinib",
    "Heart disease - gefitinib",
    "Lupus - isoniazid",
    "Non-alcoholic fatty liver disease - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - pp242",
    "Pancreatic cancer - gant61",
    "Pancreatic cancer - ginsenoside rb1",
]

prerel_labels_ac = [
    "Alzheimer's - 2 pam",
    "Alzheimer's - pp2",
    "Heart disease - gefitinib",
    "Heart disease - erlotinib",
    "Non-alcoholic fatty liver disease - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - pp242",
    "Pancreatic cancer - ginsenoside rb1",
    "Pancreatic cancer - gant61",
    "Diabetes - stavudine",
    "Diabetes - zileuton",
]

prerel_labels_abc = [
    "Alzheimer's - BCHE - 2 pam",
    "Alzheimer's - FYN - pp2",
    "Heart disease - RAF1 - gefitinib",
    "Heart disease - EGFR - erlotinib",
    "Non-alcoholic fatty liver disease - AHR - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - ULK1 - pp242",
    "Pancreatic cancer - NAMPT - ginsenoside rb1",
    "Pancreatic cancer - CCK - gant61",
    "Diabetes - AHR - stavudine",
    "Diabetes - LOX - zileuton",
    ]

# X-axis positions
x_pos = np.arange(len(labels_abc))

# Define smaller offsets to reduce horizontal jitter for human markers
offsets = {"Ron": -0.15, "Rob": -0.05, "Jack": 0.05, "Ishaan": 0.15}

# Original prerel scores (in their original order)
ishaan_scores_abc_prerel_orig = [-2, 1, 1, 0, 0, -1, -2, 0, 0, 1]
ron_scores_abc_prerel_orig = [-1, 1, 1, 0, 0, 0, -1, 0, 0, 0]
rob_scores_abc_prerel_orig = [-2, 1, 0, 0, -1, 0, -1, -1, 0, 1]
jack_scores_abc_prerel_orig = [-1, 1, 0, 0, 0, 0, -1, 0, 0, 0]

# Original prerel AC scores (in their original order)
ishaan_scores_ac_prerel_orig = [0, 0, 1, 0, 0, 0, 1, 2, -2, 0]
ron_scores_ac_prerel_orig = [0, 0, 1, 0, 0, 0, 1, 2, -2, 0]
rob_scores_ac_prerel_orig = [-2, -2, 0, 0, 0, 0, 1, 2, -2, 0]
jack_scores_ac_prerel_orig = [-1, 0, 0, 0, 0, 0, 1, 2, -2, 0]

# Original prerel labels (for reference)
prerel_labels_abc_orig = [
    "Alzheimer's - BCHE - 2 pam",
    "Alzheimer's - FYN - pp2",
    "Heart disease - RAF1 - gefitinib",
    "Heart disease - EGFR - erlotinib",
    "Non-alcoholic fatty liver disease - AHR - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - ULK1 - pp242",
    "Pancreatic cancer - NAMPT - ginsenoside rb1",
    "Pancreatic cancer - CCK - gant61",
    "Diabetes - AHR - stavudine",
    "Diabetes - LOX - zileuton",
]

prerel_labels_ac_orig = [
    "Alzheimer's - 2 pam",
    "Alzheimer's - pp2",
    "Heart disease - gefitinib",
    "Heart disease - erlotinib",
    "Non-alcoholic fatty liver disease - 6-formylindolo(3,2-b)carbazole",
    "Non-alcoholic fatty liver disease - pp242",
    "Pancreatic cancer - ginsenoside rb1",
    "Pancreatic cancer - gant61",
    "Diabetes - stavudine",
    "Diabetes - zileuton",
]

# Mapping from original prerel indices to new full indices
prerel_abc_mapping = {
    0: 0,  # Alzheimer's - BCHE - 2 pam
    1: 1,  # Alzheimer's - FYN - pp2
    2: 8,  # Heart disease - RAF1 - gefitinib
    3: 7,  # Heart disease - EGFR - erlotinib
    4: 10, # Non-alcoholic fatty liver disease - AHR - 6-formylindolo(3,2-b)carbazole
    5: 11, # Non-alcoholic fatty liver disease - ULK1 - pp242
    6: 13, # Pancreatic cancer - NAMPT - ginsenoside rb1
    7: 12, # Pancreatic cancer - CCK - gant61
    8: 4,  # Diabetes - AHR - stavudine
    9: 6,  # Diabetes - LOX - zileuton
}

prerel_ac_mapping = {
    0: 0,  # Alzheimer's - 2 pam
    1: 1,  # Alzheimer's - pp2
    2: 8,  # Heart disease - gefitinib
    3: 7,  # Heart disease - erlotinib
    4: 10, # Non-alcoholic fatty liver disease - 6-formylindolo(3,2-b)carbazole
    5: 11, # Non-alcoholic fatty liver disease - pp242
    6: 13, # Pancreatic cancer - ginsenoside rb1
    7: 12, # Pancreatic cancer - gant61
    8: 4,  # Diabetes - stavudine
    9: 6,  # Diabetes - zileuton
}

# Initialize prerel scores with the non-prerel scores
ron_scores_abc_prerel = ron_scores.copy()
rob_scores_abc_prerel = rob_scores.copy()
jack_scores_abc_prerel = jack_scores.copy()
ishaan_scores_abc_prerel = ishaan_scores.copy()

ron_scores_ac_prerel = ron_scores_ac.copy()
rob_scores_ac_prerel = rob_scores_ac.copy()
jack_scores_ac_prerel = jack_scores_ac.copy()
ishaan_scores_ac_prerel = ishaan_scores_ac.copy()

# Replace with the original prerel scores where available
for orig_idx, new_idx in prerel_abc_mapping.items():
    ron_scores_abc_prerel[new_idx] = ron_scores_abc_prerel_orig[orig_idx]
    rob_scores_abc_prerel[new_idx] = rob_scores_abc_prerel_orig[orig_idx]
    jack_scores_abc_prerel[new_idx] = jack_scores_abc_prerel_orig[orig_idx]
    ishaan_scores_abc_prerel[new_idx] = ishaan_scores_abc_prerel_orig[orig_idx]

for orig_idx, new_idx in prerel_ac_mapping.items():
    ron_scores_ac_prerel[new_idx] = ron_scores_ac_prerel_orig[orig_idx]
    rob_scores_ac_prerel[new_idx] = rob_scores_ac_prerel_orig[orig_idx]
    jack_scores_ac_prerel[new_idx] = jack_scores_ac_prerel_orig[orig_idx]
    ishaan_scores_ac_prerel[new_idx] = ishaan_scores_ac_prerel_orig[orig_idx]

# Updated prerel labels to include all items in the same order as non-prerel
prerel_labels_ac = labels_ac.copy()
prerel_labels_abc = labels_abc.copy()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create violin plots with model and human scores"
    )
    parser.add_argument(
        "parent_dir",
        help="Parent directory containing output directories with results.txt files",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory to save output plots (default: current directory)",
    )
    parser.add_argument(
        "--prerel",
        action="store_true",
        help="Use preliminary release scores and labels",
    )
    return parser.parse_args()


def build_label_mapping(args):
    label_map = {}
    if args.prerel:
        labels_to_use = prerel_labels_abc  # Use preliminary release labels if --prerel is set
    else:
        labels_to_use = labels_abc         # Otherwise, use standard labels

    for idx, label in enumerate(labels_to_use):
        # More aggressive cleaning using regex to remove non-alphanumeric and extra spaces
        clean_label = re.sub(r'[^a-z0-9\s\-]', '', label.lower()).strip()
        clean_label = re.sub(r'\s+', ' ', clean_label) # Normalize spaces
        clean_label = clean_label.replace('3 2b', '32b') # Specifically remove space in '3 2b'
        
        # Fix the "alzheimer's" vs "alzheimers" issue
        clean_label = clean_label.replace('alzheimers', 'alzheimer')
        
        print(f"Label {idx+1} cleaned: '{clean_label}'") # Debug print for cleaned label
        label_map[clean_label] = idx

        # Handle A-C mapping - create ac_key by removing gene (middle part)
        parts = clean_label.split(' - ')
        if len(parts) == 3:
            disease, gene, drug = parts
            ac_key = f"{disease} - {drug}"
            label_map[ac_key] = idx
            # Add variant without hyphen in AC key as well, just in case
            ac_key_alt = f"{disease}  {drug}" # double space to be removed in next step
            ac_key_alt = re.sub(r'\s+', ' ', ac_key_alt).strip()
            label_map[ac_key_alt] = idx

    print("\nLabel Map Contents:") # Debug print for label_map content
    for key, value in label_map.items():
        print(f"  '{key}': {value}")
    print("\n")

    return label_map


def parse_results_file(filepath, label_map):
    scores_abc = [[] for _ in range(len(labels_abc))]
    scores_ac = [[] for _ in range(len(labels_ac))]

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("Relationship_Type"):
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Warning: Line {line_num} format incorrect in {filepath}: {line}")
                continue

            relationship_type, relationship, score_part = parts

            # First normalize the relationship string - be very aggressive with cleaning
            relationship = relationship.strip().lower()
            
            # Remove all non-alphanumeric characters except hyphens and spaces
            relationship = re.sub(r"[^a-z0-9\s\-]", "", relationship)
            
            # Fix the "alzheimers" issue - standardize to "alzheimer"
            relationship = relationship.replace('alzheimers', 'alzheimer')
            
            # Special handling for 6-formylindolo compound
            relationship = re.sub(r"6-formylindolo.*carbazole", "6-formylindolo(3,2-b)carbazole", relationship)
            
            # Remove double spaces
            relationship = re.sub(r"\s+", " ", relationship).strip()

            score_match = re.match(r"^([+-]?\d+)", score_part)
            if not score_match:
                print(f"Warning: No numerical score found in line {line_num}: {line}")
                continue
            score = int(score_match.group(1))

            if "a_b_c_relationship" in relationship_type.lower():
                found_match = False
                for label_key, idx in label_map.items():
                    # Normalize the key in the exact same way as the relationship
                    normalized_key = label_key.strip().lower()
                    normalized_key = re.sub(r"[^a-z0-9\s\-]", "", normalized_key)
                    normalized_key = normalized_key.replace('alzheimers', 'alzheimer')
                    normalized_key = re.sub(r"6-formylindolo.*carbazole", "6-formylindolo(3,2-b)carbazole", normalized_key)
                    normalized_key = re.sub(r"\s+", " ", normalized_key).strip()
                    
                    if normalized_key == relationship:
                        scores_abc[idx].append(score)
                        found_match = True
                        break
                
                if not found_match:
                    print(f"Warning: A-B-C Relationship not found in mapping: {relationship}")
                    # Debug: Print relationship after normalization
                    print(f"Looking for (after normalization): '{relationship}'")
                    
            elif "a_c_relationship" in relationship_type.lower():
                found_match = False
                for label_key, idx in label_map.items():
                    # Normalize in exactly the same way
                    normalized_key = label_key.strip().lower()
                    normalized_key = re.sub(r"[^a-z0-9\s\-]", "", normalized_key)
                    normalized_key = normalized_key.replace('alzheimers', 'alzheimer')
                    normalized_key = re.sub(r"6-formylindolo.*carbazole", "6-formylindolo(3,2-b)carbazole", normalized_key)
                    normalized_key = re.sub(r"\s+", " ", normalized_key).strip()
                    
                    if normalized_key == relationship:
                        scores_ac[idx].append(score)
                        found_match = True
                        break
                
                if not found_match:
                    print(f"Warning: A-C Relationship not found in mapping: {relationship}")
                    # Debug: Print relationship after normalization
                    print(f"Looking for (after normalization): '{relationship}'")

    return scores_abc, scores_ac


def create_violin_plot(
    individual_scores, model_scores, labels, offsets, relationship_type, save_path
):
    # Set publication-quality figure parameters
    plt.rcParams.update({
        'pdf.fonttype': 42,  # Ensures text is editable in PDF
        'ps.fonttype': 42,
        'font.family': 'sans-serif',  # Use system sans-serif font
        'font.size': 14,
        'axes.linewidth': 1,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Create figure sized to full-page width (170 mm ~ 6.69") and comfortable height
    fig = plt.figure(figsize=(6.69, 4))
    
    # Replace long compound name with abbreviation
    labels = [label.replace(
        "6-formylindolo(3,2-b)carbazole",
        "6-FICZ"
    ) for label in labels]

    min_length = min(
        len(labels),
        min(len(scores) for scores in individual_scores.values())
    )
    
    # Dynamic spacing so x-tick labels have more breathing room
    base = max(1.2, 10 / max(min_length, 1))
    spacing = base * 1.6
    within_shift = spacing * 0.3
    x_pos_human = np.arange(min_length) * spacing
    x_pos_model = x_pos_human + within_shift
    x_jitter = 0.25

    # Human violins with improved styling
    human_scores = [
        [score[i] for score in individual_scores.values()] 
        for i in range(min_length)
    ]
    parts_human = plt.violinplot(
        human_scores,
        positions=x_pos_human,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for pc in parts_human["bodies"]:
        pc.set_facecolor("#4575B4")  # Professional blue color
        pc.set_alpha(0.3)
        pc.set_edgecolor("#4575B4")
    parts_human["cmedians"].set_color("#4575B4")
    parts_human["cmedians"].set_linewidth(1.5)

    # Model violins with improved styling
    valid_model_scores = [
        scores for i, scores in enumerate(model_scores) 
        if i < min_length and len(scores) > 0
    ]
    valid_positions = [
        x_pos_model[i] for i, scores in enumerate(model_scores[:min_length]) 
        if len(scores) > 0
    ]
    
    if valid_model_scores and valid_positions:
        parts_model = plt.violinplot(
            valid_model_scores,
            positions=valid_positions,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts_model["bodies"]:
            pc.set_facecolor("#D73027")  # Professional red color
            pc.set_alpha(0.3)
            pc.set_edgecolor("#D73027")
        parts_model["cmedians"].set_color("#D73027")
        parts_model["cmedians"].set_linewidth(1.5)

        # Plot model points
        for pos, scores in zip(valid_positions, valid_model_scores):
            x_jittered = pos + np.random.uniform(-x_jitter, x_jitter, len(scores))
            plt.scatter(x_jittered, scores, color="#D73027", alpha=0.5, s=30, linewidth=0)

    # Plot human scores with improved markers
    markers = {"Ron": "o", "Rob": "s", "Jack": "^", "Ishaan": "D"}
    name_mapping = {
        "Ron": "Human 1",
        "Rob": "Human 2",
        "Jack": "Human 3",
        "Ishaan": "Human 4",
    }

    for person, scores in individual_scores.items():
        x = x_pos_human + offsets.get(person, 0)
        plt.scatter(
            x[:min_length],
            scores[:min_length],
            color="#4575B4",
            marker=markers[person],
            s=30,  # slightly larger markers for better visibility
            label=name_mapping[person],
            alpha=0.9,
            edgecolors="white",
            linewidth=0.3,
        )

    # Legend with improved styling
    human_violin = plt.Rectangle((0, 0), 1, 1, fc="#4575B4", alpha=0.3)
    model_violin = plt.Rectangle((0, 0), 1, 1, fc="#D73027", alpha=0.3)
    median_proxy = Line2D([0], [0], color='black', linewidth=1.2, label='Median')
    legend_handles_full = [human_violin, model_violin, median_proxy]
    legend_labels_full = ["Human Distribution", "SKiM-GPT Distribution", "Median"]

    fig.legend(
        handles=legend_handles_full,
        labels=legend_labels_full,
        loc='lower right',
        fontsize=8,
        frameon=True,
        fancybox=True,
        edgecolor='0.5',
        borderaxespad=0.2,
        markerscale=1.2
    )

    # Set y-ticks with compact numeric labels to save horizontal space
    plt.yticks(np.arange(-2, 3), fontsize=9)
    plt.gca().set_yticklabels(["-2", "-1", "0", "1", "+2"])
    # Remove y-axis label; keep grid for orientation
    plt.ylabel("")

    # X-axis description removed for A_B_C to save space
    if relationship_type == "A_C":
        ax.set_xlabel("Disease - Drug", fontsize=10)
    else:
        ax.set_xlabel("")

    if relationship_type == "A_B_C":
        title = "Degree of support for hypothesis:\n{drug} treats {disease} through its effect on {gene}"
    else:
        title = "Degree of support for hypothesis:\n{drug} treats {disease}"

    plt.title(title, fontsize=10, pad=8)

    # Add grid with improved styling
    plt.grid(axis="y", linestyle='--', alpha=0.3)
    
    # Adjust layout and save as PDF
    plt.tight_layout()
    
    # Save full-page version
    full_page_path = save_path.replace('_600dpi', '_300dpi').replace('.pdf', '_fullpage.pdf')
    fig.savefig(
        full_page_path,
        format='pdf',
        dpi=300,
        bbox_inches='tight',
        metadata={'Creator': '', 'Producer': ''}
    )
    plt.close()


def compare_model_to_human_variance(
    individual_scores, model_scores, labels, relationship_type, save_path
):
    """
    Analyze if SKiM-GPT evaluations fall within the range of human evaluations.
    """
    results = []
    
    for i in range(len(labels)):
        human_values = [scores[i] for scores in individual_scores.values()]
        human_min = min(human_values)
        human_max = max(human_values)
        human_range = human_max - human_min
        human_mean = np.mean(human_values)
        human_std = np.std(human_values)
        
        # Skip if no model scores for this relationship
        if not model_scores[i]:
            results.append({
                'Relationship': labels[i],
                'Human_Range': f"{human_min} to {human_max}",
                'Human_Mean': human_mean,
                'Human_Std': human_std,
                'Model_Median': None,
                'Within_Human_Range': None,
                'Distance_From_Range': None,
                'Within_1_Std': None,
                'Distance_In_Std_Units': None
            })
            continue
            
        model_median = np.median(model_scores[i])
        
        # Check if model median is within human range
        within_range = human_min <= model_median <= human_max
        
        # Calculate distance from range (0 if within range)
        if within_range:
            distance_from_range = 0
        else:
            distance_from_range = min(abs(model_median - human_min), 
                                     abs(model_median - human_max))
        
        # Check if model median is within 1 std dev of human mean
        within_1_std = abs(model_median - human_mean) <= human_std
        
        # Calculate distance in standard deviation units
        if human_std > 0:
            distance_in_std = abs(model_median - human_mean) / human_std
        else:
            # All humans gave same score
            distance_in_std = 0 if model_median == human_mean else float('inf')
        
        results.append({
            'Relationship': labels[i],
            'Human_Range': f"{human_min} to {human_max}",
            'Human_Mean': human_mean,
            'Human_Std': human_std,
            'Model_Median': model_median,
            'Within_Human_Range': within_range,
            'Distance_From_Range': distance_from_range,
            'Within_1_Std': within_1_std,
            'Distance_In_Std_Units': distance_in_std
        })
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    
    # Print summary statistics
    relationships_with_scores = sum(1 for r in results if r['Model_Median'] is not None)
    within_range_count = sum(1 for r in results if r.get('Within_Human_Range', False) == True)
    within_std_count = sum(1 for r in results if r.get('Within_1_Std', False) == True)
    
    print(f"\nComparison with Human Variance for {relationship_type} relationships:")
    print(f"Total relationships with model scores: {relationships_with_scores}")
    print(f"SKiM-GPT median within human range: {within_range_count} ({within_range_count/relationships_with_scores*100:.1f}%)")
    print(f"SKiM-GPT median within 1 std dev of human mean: {within_std_count} ({within_std_count/relationships_with_scores*100:.1f}%)")
    
    return results_df


def create_combined_violin_plot(
    individual_scores_prerel, model_scores_prerel, labels_prerel,
    individual_scores_postrel, model_scores_postrel, labels_postrel,
    offsets, relationship_type, save_path
):
    # Set publication-quality figure parameters
    plt.rcParams.update({
        'pdf.fonttype': 42,  # Ensures text is editable in PDF
        'ps.fonttype': 42,
        'font.family': 'sans-serif',  # Use system sans-serif font
        'font.size': 8,
        'axes.linewidth': 1,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Create full-page figure (stacked vertically) to maximize horizontal space per panel
    fig_full, (ax1_full, ax2_full) = plt.subplots(2, 1, figsize=(6.69, 8), sharex=True)  # Full page width (170mm)
    
    # Create half-page figure (stacked vertically)
    fig_half, (ax1_half, ax2_half) = plt.subplots(2, 1, figsize=(3.35, 8), sharex=True)  # Half page width (85mm)
    
    # Replace long compound name with abbreviation in both label sets
    labels_prerel = [label.replace(
        "6-formylindolo(3,2-b)carbazole",
        "6-FICZ"
    ) for label in labels_prerel]
    
    labels_postrel = [label.replace(
        "6-formylindolo(3,2-b)carbazole",
        "6-FICZ"
    ) for label in labels_postrel]

    # Function to create a single violin plot
    def create_single_violin_plot(ax, individual_scores, model_scores, labels, offsets, title):
        min_length = min(
            len(labels),
            min(len(scores) for scores in individual_scores.values())
        )
        
        # Dynamic spacing so x-tick labels have more breathing room
        base = max(1.2, 10 / max(min_length, 1))
        spacing = base * 1.6
        within_shift = spacing * 0.3
        x_pos_human = np.arange(min_length) * spacing
        x_pos_model = x_pos_human + within_shift
        x_jitter = 0.25

        # Human violins
        human_scores = [
            [score[i] for score in individual_scores.values()] 
            for i in range(min_length)
        ]
        parts_human = ax.violinplot(
            human_scores,
            positions=x_pos_human,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts_human["bodies"]:
            pc.set_facecolor("#4575B4")
            pc.set_alpha(0.3)
            pc.set_edgecolor("#4575B4")
        parts_human["cmedians"].set_color("#4575B4")
        parts_human["cmedians"].set_linewidth(1.5)

        # Model violins
        valid_model_scores = [
            scores for i, scores in enumerate(model_scores) 
            if i < min_length and len(scores) > 0
        ]
        valid_positions = [
            x_pos_model[i] for i, scores in enumerate(model_scores[:min_length]) 
            if len(scores) > 0
        ]
        
        if valid_model_scores and valid_positions:
            parts_model = ax.violinplot(
                valid_model_scores,
                positions=valid_positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts_model["bodies"]:
                pc.set_facecolor("#D73027")
                pc.set_alpha(0.3)
                pc.set_edgecolor("#D73027")
            parts_model["cmedians"].set_color("#D73027")
            parts_model["cmedians"].set_linewidth(1.5)

            # Plot model points
            for pos, scores in zip(valid_positions, valid_model_scores):
                x_jittered = pos + np.random.uniform(-x_jitter, x_jitter, len(scores))
                ax.scatter(x_jittered, scores, color="#D73027", alpha=0.5, s=30, linewidth=0)

        # Plot individual human scores so that their marker size matches model point size
        markers = {"Ron": "o", "Rob": "s", "Jack": "^", "Ishaan": "D"}
        for person, scores_person in individual_scores.items():
            x_vals = x_pos_human + offsets.get(person, 0)
            ax.scatter(
                x_vals[:min_length],
                scores_person[:min_length],
                color="#4575B4",
                marker=markers.get(person, "o"),
                s=30,  # match model point size
                alpha=0.9,
                edgecolors="white",
                linewidth=0.3,
            )

        # Draw median as small horizontal black line (only if variation exists)
        for idx in range(min_length):
            seg = spacing * 0.15  # half-length of the median line
            # Human median
            hs = [score_set[idx] for score_set in individual_scores.values()]
            if len(set(hs)) > 1:
                m = np.median(hs)
                ax.plot([x_pos_human[idx] - seg, x_pos_human[idx] + seg], [m, m], color='black', linewidth=1.0, zorder=4)
            # Model median
            if idx < len(model_scores) and model_scores[idx] and len(set(model_scores[idx])) > 1:
                m = np.median(model_scores[idx])
                ax.plot([x_pos_model[idx] - seg, x_pos_model[idx] + seg], [m, m], color='black', linewidth=1.0, zorder=4)

        # Set x-ticks with disease abbreviations for space
        ax.set_xticks(x_pos_human + (spacing / 4))
        display_labels = []
        for lab in labels[:min_length]:
            lab = lab.replace("Non-alcoholic fatty liver disease", "NAFLD")
            lab = lab.replace("Alzheimer's", "AD").replace("Alzheimer", "AD")
            display_labels.append(lab)
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
        
        # Set y-ticks with compact numeric labels to save horizontal space
        ax.set_yticks(np.arange(-2, 3))
        ax.set_yticklabels(["-2", "-1", "0", "1", "+2"], fontsize=9)
        # Add y-axis label once per panel (depends on caller)
        ax.set_ylabel("Score", fontsize=9)
        
        # Set labels and title
        if relationship_type == "A_C":
            ax.set_xlabel("Disease - Drug", fontsize=10)
        else:
            ax.set_xlabel("")
        ax.set_title(title, fontsize=10, pad=8)

        # Add grid
        ax.grid(axis="y", linestyle='--', alpha=0.3)
        
        return ax

    # Create both plots for full-page version
    create_single_violin_plot(
        ax1_full, individual_scores_prerel, model_scores_prerel, 
        labels_prerel, offsets, "A) Without Relevance Filtering"
    )
    create_single_violin_plot(
        ax2_full, individual_scores_postrel, model_scores_postrel, 
        labels_postrel, offsets, "B) With Relevance Filtering"
    )

    # Hide x-tick labels only on top subplot (shared axis) without deleting them
    ax1_full.tick_params(axis='x', which='both', labelbottom=False)
    ax1_full.set_xlabel("")

    # Create both plots for half-page version
    create_single_violin_plot(
        ax1_half, individual_scores_prerel, model_scores_prerel, 
        labels_prerel, offsets, "A) Without Relevance Filtering"
    )
    create_single_violin_plot(
        ax2_half, individual_scores_postrel, model_scores_postrel, 
        labels_postrel, offsets, "B) With Relevance Filtering"
    )

    # Create legends (full-page and half-page) with consistent ordering
    human_violin = plt.Rectangle((0, 0), 1, 1, fc="#4575B4", alpha=0.3)
    model_violin = plt.Rectangle((0, 0), 1, 1, fc="#D73027", alpha=0.3)
    median_proxy = Line2D([0], [0], color='black', linewidth=1.2, label='Median')

    legend_handles = [human_violin, model_violin, median_proxy]
    legend_labels = ["Human Distribution", "SKiM-GPT Distribution", "Median"]

    fig_full.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower right',
        fontsize=8,
        frameon=True,
        fancybox=True,
        edgecolor='0.5',
        borderaxespad=0.2,
        markerscale=1.2
    )

    fig_half.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower right',
        fontsize=8,
        frameon=True,
        fancybox=True,
        edgecolor='0.5',
        borderaxespad=0.2,
        markerscale=1.2
    )

    # Adjust layout and save both versions (use tight_layout after legend placement)
    fig_full.tight_layout(rect=[0, 0, 0.95, 1])
    fig_half.tight_layout(rect=[0, 0, 0.95, 1])

    # Save full-page version
    full_page_path = save_path.replace('_600dpi', '_300dpi').replace('.pdf', '_fullpage.pdf')
    fig_full.savefig(
        full_page_path,
        format='pdf',
        dpi=300,
        bbox_inches='tight',
        metadata={'Creator': '', 'Producer': ''}
    )

    # Save half-page version
    half_page_path = save_path.replace('_600dpi', '_300dpi').replace('.pdf', '_halfpage.pdf')
    fig_half.savefig(
        half_page_path,
        format='pdf',
        dpi=300,
        bbox_inches='tight',
        metadata={'Creator': '', 'Producer': ''}
    )


def main():
    args = parse_args()

    # Get all subdirectories in the parent directory
    if not os.path.isdir(args.parent_dir):
        print(f"Error: '{args.parent_dir}' is not a valid directory.")
        return

    valid_output_dirs = []
    for item in os.listdir(args.parent_dir):
        dir_path = os.path.join(args.parent_dir, item)
        if os.path.isdir(dir_path) and os.path.isfile(os.path.join(dir_path, "results.txt")):
            valid_output_dirs.append(dir_path)

    if not valid_output_dirs:
        print(f"Error: No valid output directories with results.txt found in '{args.parent_dir}'")
        return

    print(f"Found {len(valid_output_dirs)} valid output directories")

    # Process both pre-release and post-release data
    os.makedirs(args.save_dir, exist_ok=True)

    # Process pre-release data
    label_map_prerel = build_label_mapping(argparse.Namespace(prerel=True))
    model_scores_abc_prerel = [[] for _ in range(len(prerel_labels_abc))]
    model_scores_ac_prerel = [[] for _ in range(len(prerel_labels_ac))]

    # Process post-release data
    label_map_postrel = build_label_mapping(argparse.Namespace(prerel=False))
    model_scores_abc_postrel = [[] for _ in range(len(labels_abc))]
    model_scores_ac_postrel = [[] for _ in range(len(labels_ac))]

    # Process all directories
    for dir_path in valid_output_dirs:
        results_file = os.path.join(dir_path, "results.txt")
        if not os.path.isfile(results_file):
            continue
            
        print(f"Processing '{results_file}'...")
        
        # Process for pre-release
        scores_abc_prerel, scores_ac_prerel = parse_results_file(results_file, label_map_prerel)
        for i in range(len(prerel_labels_abc)):
            if i < len(scores_abc_prerel):
                model_scores_abc_prerel[i].extend(scores_abc_prerel[i])
        for i in range(len(prerel_labels_ac)):
            if i < len(scores_ac_prerel):
                model_scores_ac_prerel[i].extend(scores_ac_prerel[i])
        
        # Process for post-release
        scores_abc_postrel, scores_ac_postrel = parse_results_file(results_file, label_map_postrel)
        for i in range(len(labels_abc)):
            if i < len(scores_abc_postrel):
                model_scores_abc_postrel[i].extend(scores_abc_postrel[i])
        for i in range(len(labels_ac)):
            if i < len(scores_ac_postrel):
                model_scores_ac_postrel[i].extend(scores_ac_postrel[i])

    # Create combined plots with descriptive names
    abc_plot_path = os.path.join(args.save_dir, "figure2_combined_scores_A_B_C_connection_600dpi.pdf")
    create_combined_violin_plot(
        individual_scores_prerel={
            "Ron": ron_scores_abc_prerel,
            "Rob": rob_scores_abc_prerel,
            "Jack": jack_scores_abc_prerel,
            "Ishaan": ishaan_scores_abc_prerel,
        },
        model_scores_prerel=model_scores_abc_prerel,
        labels_prerel=prerel_labels_abc,
        individual_scores_postrel={
            "Ron": ron_scores,
            "Rob": rob_scores,
            "Jack": jack_scores,
            "Ishaan": ishaan_scores,
        },
        model_scores_postrel=model_scores_abc_postrel,
        labels_postrel=labels_abc,
        offsets=offsets,
        relationship_type="A_B_C",
        save_path=abc_plot_path,
    )

    ac_plot_path = os.path.join(args.save_dir, "figure2_combined_scores_A_C_connection_600dpi.pdf")
    create_combined_violin_plot(
        individual_scores_prerel={
            "Ron": ron_scores_ac_prerel,
            "Rob": rob_scores_ac_prerel,
            "Jack": jack_scores_ac_prerel,
            "Ishaan": ishaan_scores_ac_prerel,
        },
        model_scores_prerel=model_scores_ac_prerel,
        labels_prerel=prerel_labels_ac,
        individual_scores_postrel={
            "Ron": ron_scores_ac,
            "Rob": rob_scores_ac,
            "Jack": jack_scores_ac,
            "Ishaan": ishaan_scores_ac,
        },
        model_scores_postrel=model_scores_ac_postrel,
        labels_postrel=labels_ac,
        offsets=offsets,
        relationship_type="A_C",
        save_path=ac_plot_path,
    )

    # Create variance comparison files with descriptive names
    abc_variance_path = os.path.join(args.save_dir, "figure2_human_variance_comparison_A_B_C.csv")
    abc_variance = compare_model_to_human_variance(
        individual_scores={
            "Ron": ron_scores,
            "Rob": rob_scores,
            "Jack": jack_scores,
            "Ishaan": ishaan_scores,
        },
        model_scores=model_scores_abc_postrel,
        labels=labels_abc,
        relationship_type="A_B_C",
        save_path=abc_variance_path,
    )
    
    ac_variance_path = os.path.join(args.save_dir, "figure2_human_variance_comparison_A_C.csv")
    ac_variance = compare_model_to_human_variance(
        individual_scores={
            "Ron": ron_scores_ac,
            "Rob": rob_scores_ac,
            "Jack": jack_scores_ac,
            "Ishaan": ishaan_scores_ac,
        },
        model_scores=model_scores_ac_postrel,
        labels=labels_ac,
        relationship_type="A_C",
        save_path=ac_variance_path,
    )


if __name__ == "__main__":
    main()
