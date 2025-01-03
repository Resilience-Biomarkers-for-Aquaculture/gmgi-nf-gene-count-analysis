import argparse
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Parse arguments
parser = argparse.ArgumentParser(description="Process gene counts and metadata for PCA and LDA analysis.")
parser.add_argument('--gene_counts', required=True, help="Path to the gene count matrix file.")
parser.add_argument('--metadata', required=True, help="Path to the metadata file.")
parser.add_argument('--output_dir', required=True, help="Directory to save output files.")
parser.add_argument('--low_expression_threshold', type=int, default=10, help="Threshold for filtering low-expression genes. Default is 10.")
parser.add_argument('--n_repeats', type=int, default=10, help="Number of repetitions for mutual information calculation. Default is 10.")
args = parser.parse_args()

gene_counts_path = args.gene_counts
metadata_path = args.metadata
output_dir = args.output_dir
low_expression_threshold = args.low_expression_threshold
n_repeats = args.n_repeats

# Read the gene counts and metadata files
gene_counts = pd.read_csv(gene_counts_path, sep='\t', index_col=0)
metadata = pd.read_csv(metadata_path)

# Filter gene counts to include only samples present in the metadata
samples_in_metadata = metadata['sample'].tolist()
filtered_gene_counts = \
    gene_counts.loc[:, gene_counts.columns.intersection(samples_in_metadata)]

# Filter low-expression genes
gene_sums = filtered_gene_counts.sum(axis=1)
filtered_gene_counts_high_expression = \
    filtered_gene_counts[gene_sums > low_expression_threshold]

# Normalize and log-transform data
cpm = (filtered_gene_counts_high_expression.T /
       filtered_gene_counts_high_expression.sum(axis=1)).T * 1e6
log_cpm = cpm.apply(lambda x: np.log2(x + 1), axis=1)

# Calculate variance and prepare for mutual information analysis
gene_variances = log_cpm.var(axis=1)
# Select the top 50 most variable genes
top_genes = gene_variances.nlargest(50).index
# Subset the log-transformed CPM data for these top genes
top_genes_data = log_cpm.loc[top_genes]
metadata_sorted = metadata.sort_values(by=['thermal.tolerance', 'day'])
sorted_samples = metadata_sorted['sample']

# Subset and reorder the heatmap data to match the sorted samples
top_genes_data_sorted = top_genes_data[sorted_samples]

# Calculate mutual information for subsets of 10 genes
sample_metadata = metadata.set_index('sample')
expression_data = top_genes_data_sorted.T.values
labels_resilience_day = metadata_sorted[['thermal.tolerance', 'day']]
labels_combined = labels_resilience_day['thermal.tolerance'] + "-Day" + \
    labels_resilience_day['day'].astype(str)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_combined)

mi_scores_list = []
for _ in range(n_repeats):
    mi_scores = mutual_info_classif(expression_data, encoded_labels,
                                    discrete_features=False,
                                    random_state=None)
    mi_scores_list.append(mi_scores)

avg_mi_scores = np.mean(mi_scores_list, axis=0)

mutual_info_results = pd.DataFrame({
    'Gene': top_genes_data_sorted.index,
    'Mutual_Information': avg_mi_scores
}).sort_values(by='Mutual_Information', ascending=False)

mutual_info_results.to_csv(f"{output_dir}/mutual_information_results.csv", index=False)

# PCA Plot for Top 10 Genes
top_genes_to_visualize = mutual_info_results.iloc[:10]['Gene']
scaled_top_10_data = StandardScaler().fit_transform(top_genes_data.T)  # Transpose
pca_top_10 = PCA(n_components=2).fit_transform(scaled_top_10_data)
plt.figure(figsize=(12, 10))
color_map = {'resistant': 'blue', 'susceptible': 'red'}
colors = [color_map[sample_metadata.loc[sample, 'thermal.tolerance']] for sample in filtered_gene_counts_high_expression.columns]
days = sample_metadata.loc[filtered_gene_counts_high_expression.columns, 'day'].tolist()
for i, sample in enumerate(filtered_gene_counts_high_expression.columns):
    plt.scatter(pca_top_10[i, 0], pca_top_10[i, 1], color=colors[i], s=50, alpha=0.7)
    plt.text(pca_top_10[i, 0], pca_top_10[i, 1], f"Day {days[i]}", fontsize=8, ha='right')
plt.title("PCA of Samples Based on Top 10 Genes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Resistant'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Susceptible')
]
plt.legend(handles=legend_handles, title="Thermal Tolerance")
plt.savefig(f"{output_dir}/PCA_top_10_mutual_info_genes.png")

# Additional plots and outputs can follow a similar pattern.
