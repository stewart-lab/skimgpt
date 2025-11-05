import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ------------------  Global style  ------------------
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.size': 10,  # Increased from 8 to 10
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

# ---------- Figure container (2x2 grid) ----------
# Row 1: Baseline Confusion (left), Fine-tuned Confusion (right)
# Row 2: DDI Dataset (left), PPI Dataset (right)
# Legend below the grid
fig = plt.figure(figsize=(11.0, 10.0), dpi=300)  # Increased height for bigger appearance
# Create two separate gridspecs for different row layouts
gs_top = fig.add_gridspec(1, 2, height_ratios=[1], 
                          width_ratios=[1, 1],
                          hspace=0,
                          wspace=0.5,    # Normal spacing for confusion matrices (A, B)
                          left=0.10,     # Wide margins for confusion matrices
                          right=0.90,
                          top=0.96,
                          bottom=0.62)   # Bottom of top row (raised from 0.56 to make row smaller)

gs_bottom = fig.add_gridspec(1, 2, height_ratios=[1], 
                             width_ratios=[1, 1],
                             hspace=0,
                             wspace=1.8,    # Much more spacing between C and D
                             left=0.22,     # Narrower margins for bar charts
                             right=0.85,    # Extended right margin (was 0.78) to move D to the right
                             top=0.52,      # Top of bottom row (raised from 0.46 to make row larger)
                             bottom=0.16)   # Bottom of bottom row

gs_legend = fig.add_gridspec(1, 1,
                             left=0.10,
                             right=0.90,
                             top=0.12,      # Top of legend (was 0.10) - more space above legend
                             bottom=0.03)

ax_cm_baseline = fig.add_subplot(gs_top[0, 0])
ax_cm_finetuned = fig.add_subplot(gs_top[0, 1])
ax_ddi = fig.add_subplot(gs_bottom[0, 0])
ax_ppi = fig.add_subplot(gs_bottom[0, 1])
ax_leg = fig.add_subplot(gs_legend[0, 0])

def calculate_precision_recall(categories, is_interaction, accuracy_values, total_samples_per_category=100):
    """
    Calculate precision and recall based on accuracy values and interaction classifications.
    
    Assumptions:
    - Each category has equal sample size (total_samples_per_category)
    - Accuracy represents correct classifications for that category
    - For interaction categories: accuracy = TP / (TP + FN)
    - For non-interaction categories: accuracy = TN / (TN + FP)
    """
    
    results = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    for i, (category, is_interact, acc) in enumerate(zip(categories, is_interaction, accuracy_values)):
        if is_interact:  # True interaction category
            tp = int(acc * total_samples_per_category)  # Correctly classified as interaction
            fn = total_samples_per_category - tp        # Incorrectly classified as non-interaction
            fp = 0  # Not applicable for this category
            tn = 0  # Not applicable for this category
        else:  # True non-interaction category
            tn = int(acc * total_samples_per_category)  # Correctly classified as non-interaction
            fp = total_samples_per_category - tn        # Incorrectly classified as interaction
            tp = 0  # Not applicable for this category
            fn = 0  # Not applicable for this category
        
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        
        results.append({
            'category': category,
            'is_interaction': is_interact,
            'accuracy': acc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # Calculate overall precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return results, precision, recall, f1_score, total_tp, total_fp, total_tn, total_fn

# Data for DDI Dataset with interaction/non-interaction classification
ddi_categories = [
    "Variable Interaction",
    "Strong Interaction",
    "Separate Contexts",
    "Rare Interaction",
    "Potential Interaction",
    "Non-Drug Mention",
    "No Significant Interaction",
    "Moderate Interaction",
    "Mild Interaction",
    "List Mention",
    "Inconclusive",
    "In Vitro Interaction",
    "Conflicting Evidence",
    "Comparative Study",
    "Background Information"
]

# Identify which are interaction vs non-interaction types
ddi_is_interaction = [
    True,  # Variable Interaction
    True,  # Strong Interaction
    False, # Separate Contexts
    True,  # Rare Interaction
    True,  # Potential Interaction
    False, # Non-Drug Mention
    True,  # No Significant Interaction
    True,  # Moderate Interaction
    True,  # Mild Interaction
    False, # List Mention
    True,  # Inconclusive
    True,  # In Vitro Interaction
    True,  # Conflicting Evidence
    False, # Comparative Study
    False  # Background Information
]

# Approximate values based on the provided image
ddi_accuracy = [
    1, 1, 1, 1, 1, 0.7, 
    1, 1, 1, 0.75, 1, 1, 
    1, 0.42, 0.3
]

# Calculate precision and recall for DDI dataset
ddi_results, ddi_precision, ddi_recall, ddi_f1, ddi_tp, ddi_fp, ddi_tn, ddi_fn = calculate_precision_recall(
    ddi_categories, ddi_is_interaction, ddi_accuracy
)

# Sort the data by interaction type (true first) and then by accuracy
ddi_combined = list(zip(ddi_categories, ddi_is_interaction, ddi_accuracy))
ddi_combined.sort(key=lambda x: (-int(x[1]), -x[2]))  # Sort by interaction type (true first), then by descending accuracy
ddi_categories, ddi_is_interaction, ddi_accuracy = zip(*ddi_combined)

# Data for PPI Dataset (renamed from Protein Dataset)
ppi_categories = [
    "Signal Transduction Interaction",
    "Separate Pathways",
    "Regulatory Interaction",
    "Protein Modification",
    "Protein Complex Formation",
    "Parallel Functions",
    "Non-Interacting Proteins",
    "No Direct Interaction",
    "Genetic Interaction",
    "Functional Association",
    "Enzyme-Substrate Interaction",
    "Direct Physical Interaction",
    "Different Cellular Compartments",
    "Conflicting Evidence"
]

# Identify which are interaction vs non-interaction types
ppi_is_interaction = [
    True,  # Signal Transduction Interaction
    False, # Separate Pathways
    True,  # Regulatory Interaction
    True,  # Protein Modification
    True,  # Protein Complex Formation
    False, # Parallel Functions
    False, # Non-Interacting Proteins
    True,  # No Direct Interaction
    True,  # Genetic Interaction
    True,  # Functional Association
    True,  # Enzyme-Substrate Interaction
    True,  # Direct Physical Interaction
    False, # Different Cellular Compartments
    True,  # Conflicting Evidence
]

# Approximate values based on the provided image
ppi_accuracy = [
    1, 1, 1, 1, 1, 0.6,
    1, 1, 1, 1, 1, 1,
    0.95, 1
]

# Calculate precision and recall for PPI dataset
ppi_results, ppi_precision, ppi_recall, ppi_f1, ppi_tp, ppi_fp, ppi_tn, ppi_fn = calculate_precision_recall(
    ppi_categories, ppi_is_interaction, ppi_accuracy
)

# Sort the data by interaction type (true first) and then by accuracy
ppi_combined = list(zip(ppi_categories, ppi_is_interaction, ppi_accuracy))
ppi_combined.sort(key=lambda x: (-int(x[1]), -x[2]))  # Sort by interaction type (true first), then by descending accuracy
ppi_categories, ppi_is_interaction, ppi_accuracy = zip(*ppi_combined)

# Print precision and recall results
print("=== DDI Dataset Metrics ===")
print(f"Precision: {ddi_precision:.4f}")
print(f"Recall: {ddi_recall:.4f}")
print(f"F1-Score: {ddi_f1:.4f}")
print(f"True Positives: {ddi_tp}")
print(f"False Positives: {ddi_fp}")
print(f"True Negatives: {ddi_tn}")
print(f"False Negatives: {ddi_fn}")
print()

print("=== PPI Dataset Metrics ===")
print(f"Precision: {ppi_precision:.4f}")
print(f"Recall: {ppi_recall:.4f}")
print(f"F1-Score: {ppi_f1:.4f}")
print(f"True Positives: {ppi_tp}")
print(f"False Positives: {ppi_fp}")
print(f"True Negatives: {ppi_tn}")
print(f"False Negatives: {ppi_fn}")
print()

# Define improved colors for interaction and non-interaction types
interaction_color = '#1f77b4'     # Blue
non_interaction_color = '#ff7f0e' # Orange

def plot_dataset(ax, categories, is_interaction, accuracy, panel_label, dataset_name, precision, recall, f1):
    colors = [interaction_color if is_int else non_interaction_color for is_int in is_interaction]
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, accuracy, color=colors, height=0.5)  # Reduced bar height from 0.6 to 0.5
    ax.set_title(f'{panel_label}) {dataset_name}', fontsize=12, pad=10)
    ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylabel('Category', fontsize=11, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlim(0, 1.0)  # Narrowed from 1.05 to 1.0
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    # Adjust subplot to make room for y-labels
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', labelsize=10)

# Plot full-page version
plot_dataset(ax_ddi, ddi_categories, ddi_is_interaction, ddi_accuracy,
            'C', 'Drug–Drug Interaction (DDI) Dataset', ddi_precision, ddi_recall, ddi_f1)

plot_dataset(ax_ppi, ppi_categories, ppi_is_interaction, ppi_accuracy,
            'D', 'Functional Gene Interaction (FGI) Dataset', ppi_precision, ppi_recall, ppi_f1)

# Add legend for interaction types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=interaction_color, label='Relevant Category'),
    Patch(facecolor=non_interaction_color, label='Irrelevant Category')
]

# --- Legend row (spans full width) ---
ax_leg.axis('off')
ax_leg.legend(handles=legend_elements, loc='center', ncol=2, fontsize=11, frameon=False,
              columnspacing=3.5)  # Increased spacing between legend items

# -------- Confusion matrix data --------
# Baseline (non fine-tuned) confusion matrix - Panel A
# From baseline_results.json: TN=5, FP=19, FN=4, TP=44
cm_baseline = np.array([[44, 4], [19, 5]])

# Fine-tuned confusion matrix - Panel B
cm_finetuned = np.array([[46, 2], [8, 16]])

def plot_confusion(ax, cm, panel_label, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=False,
                xticklabels=["Relevant","Irrelevant"],
                yticklabels=["Relevant","Irrelevant"], ax=ax, linewidths=0.5, linecolor='gray',
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=11, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=11, fontweight='bold')
    ax.set_title(f"{panel_label}) {title}", pad=12, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_aspect('auto')

# ---------- Render confusion matrices BEFORE saving ----------

plot_confusion(ax_cm_baseline, cm_baseline, "A", "Pretrained Model: Relevance Classification")
plot_confusion(ax_cm_finetuned, cm_finetuned, "B", "Fine-tuned Model: Relevance Classification")

# Set y-axis labels rotation for both confusion matrices
ax_cm_baseline.set_yticklabels(ax_cm_baseline.get_yticklabels(), rotation=0)
ax_cm_finetuned.set_yticklabels(ax_cm_finetuned.get_yticklabels(), rotation=0)

# Titles are already set in plot_dataset function, no need to override

# Save figure
fig.savefig('figure3_combined_accuracy_charts_300dpi_fullpage.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
fig.savefig('figure3_combined_accuracy_charts_300dpi_fullpage.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05, metadata={'Creator': '', 'Producer': ''})

print("Figure 3 saved as PNG/PDF.")

plt.show() 