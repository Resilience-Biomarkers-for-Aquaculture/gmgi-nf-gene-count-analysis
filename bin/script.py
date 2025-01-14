#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')

import io
import boto3
from botocore.exceptions import NoCredentialsError

def save_figure(fig, path, format='png', **kwargs):
    """
    Save a matplotlib figure to a local file or an S3 bucket using in-memory storage.

    Parameters:
        fig (matplotlib.figure.Figure): The matplotlib figure to save.
        path (str): The file path or S3 bucket path (e.g., 's3://bucket-name/key-name').
        format (str): The format of the file (e.g., 'png', 'jpg', 'pdf').
        **kwargs: Additional keyword arguments passed to plt.savefig().
    """
    if path.startswith("s3://"):
        # Parse the S3 bucket and key
        s3_path = path.replace("s3://", "").split("/", 1)
        if len(s3_path) != 2:
            raise ValueError("Invalid S3 path. Format should be 's3://bucket-name/key-name'.")
        bucket_name, key_name = s3_path

        # Save the figure to an in-memory buffer
        buffer = io.BytesIO()
        try:
            fig.savefig(buffer, format=format, **kwargs)
            buffer.seek(0)  # Reset the buffer's pointer to the beginning

            # Upload the buffer content to S3
            s3_client = boto3.client('s3')
            s3_client.upload_fileobj(buffer, bucket_name, key_name)

            print(f"Figure saved to S3 at {path}")
        except NoCredentialsError:
            raise NoCredentialsError("AWS credentials not found. Please configure your AWS environment.")
        except Exception as e:
            raise Exception(f"Failed to save figure to S3: {e}")
        finally:
            buffer.close()
    else:
        # Save to the local filesystem
        try:
            fig.savefig(path, format=format, **kwargs)
            print(f"Figure saved locally at {path}")
        except Exception as e:
            raise Exception(f"Failed to save figure locally: {e}")


# Parse arguments
parser = argparse.ArgumentParser(description="Process gene counts and metadata for PCA and LDA analysis.")
parser.add_argument('--gene_counts', required=True, help="Path to the gene count matrix file.")
parser.add_argument('--metadata', required=True, help="Path to the metadata file.")
parser.add_argument('--output_dir', required=True, help="Directory to save output files.")
parser.add_argument('--mutual_information_threshold', type=int, default=10, help="Top N highest mutual information genes. Default is 10.")
parser.add_argument('--low_expression_threshold', type=int, default=10, help="Threshold for filtering low-expression genes. Default is 10.")
parser.add_argument('--n_repeats', type=int, default=10, help="Number of repetitions for mutual information calculation. Default is 10.")
args = parser.parse_args()

gene_counts_path = args.gene_counts
metadata_path = args.metadata
output_dir = args.output_dir
mutual_information_threshold = args.mutual_information_threshold
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

# Create heatmap with samples grouped by thermal resilience and day
# Create custom labels for the x-axis showing both thermal resilience and day
x_labels = [
    f"{metadata_sorted.loc[metadata_sorted['sample'] == sample, 'thermal.tolerance'].values[0]}-Day{metadata_sorted.loc[metadata_sorted['sample'] == sample, 'day'].values[0]}"
    for sample in sorted_samples
]

# Create heatmap with labeled groupings
plt.figure(figsize=(16, 10))
sns.heatmap(
    top_genes_data_sorted, 
    cmap="viridis", 
    yticklabels=False, 
    xticklabels=x_labels, 
    cbar_kws={"label": "Log2(CPM + 1)"}
)
plt.title("Heatmap of Top 50 Most Variable Genes (Grouped by Thermal Resilience and Day)")
plt.xlabel("Samples (Resilience-Day)")
plt.ylabel("Genes")
plt.xticks(rotation=90, fontsize=8)
save_figure(plt, os.path.join(output_dir, "heatmap_top_50_variable_genes.png"))

# Calculate mutual information
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

mutual_info_results.to_csv(os.path.join(output_dir,"mutual_information_results.csv"), index=False)

top_genes_mi = mutual_info_results.iloc[:mutual_information_threshold]['Gene']
top_data_mi = log_cpm.loc[top_genes_mi]

# Reorder samples based on metadata (optional)
top_data_mi_sorted = top_data_mi[sorted_samples]

# PCA Plot for Top 10 Genes
# Select the top genes based on averaged mutual information scores

scaled_top_mi_data = StandardScaler().fit_transform(top_data_mi.T)  # Transpose
pca_top_mi = PCA(n_components=2).fit_transform(scaled_top_mi_data)
plt.figure(figsize=(12, 10))
color_map = {'resistant': 'blue', 'susceptible': 'red'}
colors = [color_map[sample_metadata.loc[sample, 'thermal.tolerance']] for sample in filtered_gene_counts_high_expression.columns]
days = sample_metadata.loc[filtered_gene_counts_high_expression.columns, 'day'].tolist()
for i, sample in enumerate(filtered_gene_counts_high_expression.columns):
    plt.scatter(pca_top_mi[i, 0], pca_top_mi[i, 1], color=colors[i], s=50, alpha=0.7)
    plt.text(pca_top_mi[i, 0], pca_top_mi[i, 1], f"Day {days[i]}", fontsize=8, ha='right')
plt.title(f"PCA of Samples Based on Top {mutual_information_threshold} Genes")
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
save_figure(plt, os.path.join(output_dir, f"PCA_top_{mutual_information_threshold}_mutual_info_genes.png"))


# Create heatmap with grouped samples
plt.figure(figsize=(12, 8))
sns.heatmap(
    top_data_mi_sorted,
    cmap="viridis",
    xticklabels=sorted_samples,
    yticklabels=top_genes_mi,
    cbar_kws={"label": "Log2(CPM + 1)"}
)
plt.title(f"Heatmap of Top {mutual_information_threshold} Genes with Highest Mutual Information (Grouped by Thermal Tolerance and Day)")
plt.xlabel("Samples (Grouped by Resilience and Day)")
plt.ylabel("Genes")
plt.xticks(rotation=90, fontsize=8)
save_figure(plt, os.path.join(output_dir, f"heatmap_top_{mutual_information_threshold}_mutual_info_genes.png"))

#---

# Create heatmap with hierarchical clustering
plt.figure(figsize=(12, 10))
sns.clustermap(
    top_data_mi_sorted,
    cmap="viridis",
    metric="euclidean",
    method="average",
    col_cluster=True,  # Cluster samples
    row_cluster=True,  # Cluster genes
    cbar_kws={"label": "Log2(CPM + 1)"}
)
plt.title(f"Hierarchical Clustering Heatmap of Top {mutual_information_threshold} Mutual Information Genes")
save_figure(plt, os.path.join(output_dir, f"heatmap_hierarchical_top_{mutual_information_threshold}_mutual_info_genes.png"))

# Additional plots and outputs can follow a similar pattern.
