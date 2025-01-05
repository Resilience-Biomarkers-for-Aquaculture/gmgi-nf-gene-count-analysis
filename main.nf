#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.gene_counts = 's3://your-bucket-name/data/gene_counts.tsv'
params.metadata = 's3://your-bucket-name/data/metadata.csv'
params.output_dir = 's3://your-bucket-name/results'
params.low_expression_threshold = 10
params.n_repeats = 10


workflow {
    main:
    def gene_counts_file = Channel.fromPath(params.gene_counts)
    def metadata_file = Channel.fromPath(params.metadata)
    process_script(gene_counts_file, metadata_file, params.output_dir, params.low_expression_threshold, params.n_repeats)
}

process process_script {
    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
        path gene_counts_file
        path metadata_file
        val output_dir
        val low_expression_threshold
        val n_repeats

    output:
        path("${output_dir}/*")

    script:
    """
    script.py \
        --gene_counts ${gene_counts_file} \
        --metadata ${metadata_file} \
        --output_dir ${output_dir} \
        --low_expression_threshold ${low_expression_threshold} \
        --n_repeats ${n_repeats}
    """
}

