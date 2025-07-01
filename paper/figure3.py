import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ------------------  Global style  ------------------
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.size': 8,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

# ---------- Figure container (single column) ----------
# 4 rows: Confusion, DDI, PPI, Legend
fig, (ax_cm, ax_ddi, ax_ppi, ax_leg) = plt.subplots(
    4, 1,
    figsize=(3.35, 9.5),   # slight shrink to stay <250 mm
    dpi=300,
    gridspec_kw={'height_ratios': [1, 1, 1, 0.18]}
)
fig.subplots_adjust(hspace=0.75)

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
    ax.barh(y_pos, accuracy, color=colors, height=0.6)
    ax.set_title(f'{panel_label}) {dataset_name}', fontsize=8, pad=4)
    ax.set_xlabel('Accuracy', fontsize=7)
    ax.set_ylabel('Relevance Category', fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=6)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

# Plot full-page version
plot_dataset(ax_ddi, ddi_categories, ddi_is_interaction, ddi_accuracy,
            'B', 'DDI Dataset', ddi_precision, ddi_recall, ddi_f1)

plot_dataset(ax_ppi, ppi_categories, ppi_is_interaction, ppi_accuracy,
            'C', 'PPI Dataset', ppi_precision, ppi_recall, ppi_f1)

# Add legend for interaction types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=interaction_color, label='Interaction Category (Relevant to Hypothesis)'),
    Patch(facecolor=non_interaction_color, label='Non-Interaction Category (Not Relevant)')
]

# --- Legend row ---
ax_leg.axis('off')
# Render legend and then shift legend axis upwards slightly
ax_leg.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=6, frameon=False)

# Move legend axis closer to panel C (reduce bottom padding)
leg_pos = ax_leg.get_position()
ax_leg.set_position([leg_pos.x0, leg_pos.y0 + 0.03, leg_pos.width, leg_pos.height])

# -------- Confusion matrix data (Panel A) --------
cm = np.array([[46,2],[8,16]])

def plot_confusion(ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=False,
                xticklabels=["Relevant","Irrelevant"],
                yticklabels=["Relevant","Irrelevant"], ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("A) Confusion Matrix: Relevance Classification", pad=6)
    ax.set_aspect('auto')

# ---------- Render confusion matrix BEFORE saving ----------

plot_confusion(ax_cm)

if ax_cm.collections and ax_cm.collections[0].colorbar:
    ax_cm.collections[0].colorbar.set_label('Count')

ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)

# Confusion matrix already titled concisely above; ensure fontsize small

ax_ddi.set_title("B) DDI Dataset Accuracy", pad=6)
ax_ppi.set_title("C) PPI Dataset Accuracy", pad=6)

# Adjust layout with minimal padding
fig.tight_layout(pad=0.2)
# Extra bottom padding now handled by legend row; small bottom margin
fig.subplots_adjust(bottom=0.02)

# Manually stretch plotting axes horizontally to occupy most of figure width
for ax in [ax_cm, ax_ddi, ax_ppi]:
    pos = ax.get_position()
    ax.set_position([0.16, pos.y0, 0.78, pos.height])  # x0, y0, width, height

# Save figure
fig.savefig('figure3_combined_accuracy_charts_300dpi_fullpage.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
fig.savefig('figure3_combined_accuracy_charts_300dpi_fullpage.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05, metadata={'Creator': '', 'Producer': ''})

print("Figure 3 saved as PNG/PDF.")

plt.show() 