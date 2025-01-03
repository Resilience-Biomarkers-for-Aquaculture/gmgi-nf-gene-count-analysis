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


# Reload the data files
gene_counts_path = '/mnt/c/Users/inter/Downloads/GMGI/ChatGPT/salmon.merged.gene_counts_2023_seqera.tsv'
metadata_path = '/mnt/c/Users/inter/Downloads/GMGI/ChatGPT/deseq_metadata2.csv'

# Read the gene counts and metadata files
gene_counts = pd.read_csv(gene_counts_path, sep='\t', index_col=0)
metadata = pd.read_csv(metadata_path)

# Filter gene counts to include only samples present in the metadata
samples_in_metadata = metadata['sample'].tolist()
filtered_gene_counts = \
    gene_counts.loc[:, gene_counts.columns.intersection(samples_in_metadata)]

# Filter low-expression genes
low_expression_threshold = 10
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
# Prepare data for mutual information calculation
# Flatten the top genes data for each sample to compare with categorical labels
expression_data = top_genes_data_sorted.T.values  # Shape: (samples, genes)

# Prepare data for mutual information calculation
# Flatten the top genes data for each sample to compare with categorical labels
expression_data = top_genes_data_sorted.T.values  # Shape: (samples, genes)

# Encode thermal tolerance and day as labels for mutual information analysis
labels_resilience_day = metadata_sorted[['thermal.tolerance', 'day']]
labels_combined = labels_resilience_day['thermal.tolerance'] + "-Day" + \
    labels_resilience_day['day'].astype(str)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_combined)

# Calculate mutual information multiple times and average, because
# it's a stochastic algorithm
mi_scores_list = []
n_repeats = 5
for _ in range(n_repeats):
    mi_scores = mutual_info_classif(expression_data, encoded_labels,
                                    discrete_features=False,
                                    # Each repetition uses a different
                                    # random initialization
                                    random_state=None)
    mi_scores_list.append(mi_scores)

avg_mi_scores = np.mean(mi_scores_list, axis=0)

# Create a DataFrame for averaged results
mutual_info_results = pd.DataFrame({
    'Gene': top_genes_data_sorted.index,
    'Mutual_Information': avg_mi_scores
}).sort_values(by='Mutual_Information', ascending=False)

print(mutual_info_results.head(10))


# Do PCA plot for the top 10, genes in mutual information score
top_genes_to_visualize = mutual_info_results.iloc[:10]['Gene']

# Subset the log-transformed CPM data for these genes
top_genes_data = log_cpm.loc[top_genes_to_visualize]

# Perform PCA for the top 10 genes
scaler = StandardScaler()
scaled_top_10_data = scaler.fit_transform(top_genes_data.T)  # Transpose so genes are features
pca_top_10 = PCA(n_components=2).fit_transform(scaled_top_10_data)
# PCA plot for Top 10 Genes
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

# Add legend for thermal tolerance
# Add legend
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Resistant'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Susceptible')
]
plt.legend(handles=legend_handles, title="Thermal Tolerance")

plt.savefig("ChatGPT/PCA_top_10_mutual_info_genes.png")



# Encode thermal tolerance labels
labels = metadata.set_index('sample').loc[log_cpm.columns, 'thermal.tolerance']
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # 0 = susceptible, 1 = resistant
colors = ['red' if label == 0 else 'blue' for label in encoded_labels]  # Red for susceptible, blue for resistant

# Perform PCA on the selected genes
pca = PCA(n_components=2)
pca_result = pca.fit_transform(top_genes_data.T.values)

# Perform LDA on the selected genes
lda = LinearDiscriminantAnalysis(n_components=1)
lda_result = lda.fit_transform(top_genes_data.T.values, encoded_labels)
# Evaluate LDA performance
# Predict class labels using LDA
lda_predictions = lda.predict(top_genes_data.T.values)

# Evaluate LDA performance
lda_accuracy = accuracy_score(encoded_labels, lda_predictions)
lda_conf_matrix = confusion_matrix(encoded_labels, lda_predictions)
lda_report = classification_report(encoded_labels, lda_predictions,
                                   target_names=label_encoder.classes_)

print(lda_accuracy)
print(lda_conf_matrix)
print(lda_report)

# Create side-by-side comparison plots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# PCA Plot
axs[0].scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7, s=100)
for i, sample in enumerate(log_cpm.columns):
    axs[0].text(pca_result[i, 0], pca_result[i, 1], sample, fontsize=8, ha='right')
axs[0].set_title("PCA Plot: Samples Based on Top Genes")
axs[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
axs[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
axs[0].grid(True)

# LDA Plot
axs[1].scatter(lda_result, [0] * len(lda_result), c=colors, alpha=0.7, s=100)
for i, sample in enumerate(log_cpm.columns):
    axs[1].text(lda_result[i], 0, sample, fontsize=8, ha='right', va='center')
axs[1].set_title("LDA Plot: Samples Based on Top Genes")
axs[1].set_xlabel("LDA Component 1")
axs[1].set_yticks([])
axs[1].grid(True)

# Add legends
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Susceptible'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Resistant')
]
fig.legend(handles=legend_handles, title="Thermal Tolerance", loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig("ChatGPT/PCA_LDA_top_10_mutual_info_genes.png")

