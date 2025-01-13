# Seqera-Compatible Nextflow Pipeline

This repository contains a Nextflow pipeline for analyzing gene count data and metadata using PCA and LDA visualizations.

## Repository Structure
- `bin/script.py`: Python script for data processing.
- `main.nf`: Nextflow workflow definition.
- `nextflow.config`: Configuration for pipeline parameters and AWS Batch.
- `results/`: Output directory (auto-created by the pipeline).

## Usage
1. Install Nextflow:
   ```bash
   curl -s https://get.nextflow.io | bash
   ```

2. Launch the pipeline on Seqera Tower:
   - Connect this repository to Seqera Tower.
   - Use the `awsbatch` profile for running on AWS Batch.

3. Outputs will be stored in your S3 bucket (`results`).

## Parameters
- `gene_counts`: Path to the gene count matrix.
- `metadata`: Path to the metadata file.
- `output_dir`: S3 directory for results.
- `mutual_information_threshold`: Consider the top N genes with respect to mutual information scores.
- `low_expression_threshold`: Threshold for filtering low-expression genes.
- `n_repeats`: Number of repetitions for mutual information calculation.

## Example Run
```bash
nextflow run main.nf -profile awsbatch
```
