#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.gene_counts = 's3://your-bucket-name/data/gene_counts.tsv'
params.metadata = 's3://your-bucket-name/data/metadata.csv'
params.output_dir = 's3://your-bucket-name/results'
params.low_expression_threshold = 10
params.n_repeats = 10

workflow {
    main:
    process_script(params.gene_counts, params.metadata, params.output_dir, params.low_expression_threshold, params.n_repeats)
}

process process_script {
    input:
    path gene_counts_file from Channel.value(params.gene_counts)
    path metadata_file from Channel.value(params.metadata)
    val output_dir from params.output_dir
    val low_expression_threshold from params.low_expression_threshold
    val n_repeats from params.n_repeats

    output:
    path("${output_dir}/*") into results_channel

    script:
    """
    python bin/script.py \
        --gene_counts ${gene_counts_file} \
        --metadata ${metadata_file} \
        --output_dir ${output_dir} \
        --low_expression_threshold ${low_expression_threshold} \
        --n_repeats ${n_repeats}
    """
}

