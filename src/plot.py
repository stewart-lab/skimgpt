import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to normalize the relationship text
def normalize_text(text):
    return ' '.join(text.strip().lower().replace('’', "'").replace('-', ' ').split())

if __name__ == "__main__":
    # Accept parent directory as a command-line argument
    if len(sys.argv) != 2:
        print('Usage: python plot.py <parent_dir>')
        sys.exit(1)
    
    parent_dir = sys.argv[1]

    # Prepare an empty DataFrame to store results
    df = pd.DataFrame(columns=['Full Relationship', 'Score'])

    # Iterate over each directory
    for child_dir in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, child_dir)
        if os.path.isdir(dir_path):
            file_path = os.path.join(dir_path, 'results.txt')
            if os.path.exists(file_path):
                # Read the results.txt file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    data_list = []  # List to collect data
                    for line in lines:
                        parts = line.strip().split(', Score: ')
                        if len(parts) == 2:
                            full_relationship = normalize_text(parts[0].replace('Relationship: ', ''))
                            score = float(parts[1])
                            data_list.append({'Full Relationship': full_relationship, 'Score': score})

                    # Append all collected data at once using concat
                    if data_list:
                        df = pd.concat([df, pd.DataFrame(data_list)], ignore_index=True)

    # Manually loaded data for individual scores with normalization applied
    scores_data = {
        'Full Relationship': [
            'Alzheimer’s - BCHE - 2 pam', 'Alzheimer’s - FYN - PP2', 'heart disease - RAF1 - gefitinib',
            'heart disease - EGFR - erlotinib', 'Non-alcoholic fatty liver disease - AHR - 6-formylindolo(3,2-b)carbazole',
            'Non-alcoholic fatty liver disease - ULK1 - pp242', 'pancreatic cancer - NAMPT - ginsenoside rb1',
            'pancreatic cancer - CCK - gant61', 'diabetes - AHR - stavudine', 'diabetes - LOX - zileuton',
            'breast cancer - CDK4 - ABEMACICLIB', 'breast cancer - ESR1 - estrogens', 'diabetes - GIPR - tirzepatide',
            'lupus - NAT2 - Isoniazid'
        ] * 3,  # Repeat for each score type
        'Score': [-1.5, 1, 0, 0, -1, -1, 0, 0, -1, 1, 2, -2, 2, -2,  # Rob scores
                  -1, 0, 1, 1, 0, 0, 0.5, 0, -1, 0.5, 2, -2, 2, -2,  # Ron scores
                  -1, 1, 0, 1, 0, 0, 0, 1, -1, 0, 2, -2, 2, -2]      # Jack scores
    }
    scores_df = pd.DataFrame(scores_data)
    scores_df['Full Relationship'] = scores_df['Full Relationship'].apply(normalize_text)

    # Calculate mean differences
    df_mean = df.groupby('Full Relationship')['Score'].mean()
    scores_df_mean = scores_df.groupby('Full Relationship')['Score'].mean()
    mean_diffs = (df_mean - scores_df_mean.loc[df_mean.index]).abs().sort_values(ascending=False)
    sum_diffs = (df_mean - scores_df_mean.loc[df_mean.index]).abs().sum()

    print(mean_diffs)
    print("Sum of differences:", sum_diffs)

    # Plotting with violin plots for both datasets
    plt.figure(figsize=(18, 10))

    # Plot overall scores
    sns.violinplot(data=df, x='Full Relationship', y='Score', scale='width', inner='quartile', color='lightblue', alpha=0.5)

    # Adding jitter to both distributions to avoid overlap
    sns.stripplot(data=df, x='Full Relationship', y='Score', color='blue', size=6, jitter=0.3, dodge=True)
    sns.stripplot(data=scores_df, x='Full Relationship', y='Score', color='red', size=6, jitter=0.3, dodge=True)

    plt.title('Score Distribution by Full Relationship with Individual Scores')
    plt.xlabel('Full Relationship')
    plt.ylabel('Score')
    plt.xticks(rotation=90, fontsize=8)  # Rotate and resize x-axis labels for better visibility

    plt.tight_layout()
    plt.show()
    plt.savefig('score_distribution_double.png')