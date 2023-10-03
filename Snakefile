from Bio import SeqIO


# input/output file names
data_folder="data/"
input_alignment="input_alignment.fasta"
plots_folder="plots/"
IQTREE_SUFFIXES=["iqtree", "log", "treefile", "ckp.gz"]


# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


def get_attachment_edge_indices(input_file):
    num_sequences = 0
    for record in SeqIO.parse(input_file, "fasta"):
        num_sequences += 1
    return range(1, 2*(num_sequences-1)-2)


# Define the workflow
rule all:
    input:
        plots_folder+"edpl_vs_tii.pdf",
        plots_folder+"likelihood_swarmplots.pdf",
        plots_folder+"seq_distance_vs_tii.pdf",
        plots_folder+"bootstrap_vs_tii.pdf",
        data_folder+"benchmarking_data.csv"

# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        temp(touch(data_folder+"model-test-iqtree.done")),
        modeltest=data_folder+input_alignment+"_model.iqtree"
    benchmark:
        temp(data_folder+"benchmarking/benchmark_model_test_iqtree.txt")
    shell:
        """
        if [[ -f "{data_folder}{input.msa}_model.iqtree" ]]; then
          echo "Ignoring iqtree ModelFinder run on {input.msa}, since it is already done."
        else
          iqtree -s {input.msa} --prefix {data_folder}{input.msa}_model -m MF
        fi
        """

# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        data_folder+"model-test-iqtree.done",
        msa=rules.model_test_iqtree.output.modeltest
    output:
        model=data_folder+"iqtree-model.txt"
    shell:
        """
        echo $(grep "Best-fit model" {input.msa} | cut -d ":" -f 2) > {output.model}
        """

# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa=input_alignment
    output:
        reduced_msa=temp(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta")
    benchmark:
        temp(data_folder+"benchmarking/benchmark_remove_sequence_{seq_id}.txt")
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/remove_sequence.py"

# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        msa=input_alignment,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        temp(touch(data_folder+"run_iqtree_on_full_dataset.done")),
        tree=data_folder+input_alignment+".treefile",
        mldist=data_folder+input_alignment+".mldist"
    benchmark:
        temp(data_folder+"benchmarking/benchmark_run_iqtree_on_full_dataset.txt")
    shell:
        """
        if [[ -f "{data_folder}{input.msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.msa}, since it is already done."
        else
          cp {input.msa} {data_folder}
          iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {data_folder}{input.msa} -bb 1000
        fi
        """

rule get_restricted_trees:
    input:
        data_folder+"run_iqtree_on_full_dataset.done",
        full_tree=data_folder+input_alignment+".treefile"
    output:
        restricted_trees=expand(data_folder+"reduced_alignments/{seq_id}/restricted_tree.treefile", seq_id=get_seq_ids(input_alignment))
    script:
        "scripts/create_restricted_trees.py"

# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        done=temp(touch(data_folder+"reduced_alignments/{seq_id}/run_iqtree_restricted_alignments.done")),
        tree=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        mlfile=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.iqtree"
    benchmark:
        temp(data_folder+"benchmarking/benchmark_run_iqtree_restricted_alignments_{seq_id}.txt")
    shell:
        """
        if [[ -f "{input.reduced_msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.reduced_msa}, since it is already done."
        else
          iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000
        fi
        """

# Define the rule to attach the pruned taxon at each edge
rule reattach_removed_sequence:
    input:
        rules.run_iqtree_restricted_alignments.output.done,
        reduced_tree_nwk=rules.run_iqtree_restricted_alignments.output.tree
    output:
        topologies=expand(data_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk", edge=get_attachment_edge_indices(input_alignment))
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    benchmark:
        temp(data_folder+"benchmarking/benchmark_reattach_removed_sequence_{seq_id}.txt")
    script:
        "scripts/reattach_removed_sequence.py"

rule run_iqtree_on_augmented_topologies:
    input:
        msa=input_alignment,
        topology_file=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk",
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        alldone=temp(touch(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.run_iqtree.done")),
        treefile=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile",
        mlfile=temp(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.iqtree"),
        other=temp(expand(data_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{{edge}}.nwk_branch_length.{suffix}",
        suffix=[suf for suf in IQTREE_SUFFIXES if suf not in ["iqtree", "treefile"]]))
    benchmark:
        temp(data_folder+"benchmarking/benchmark_run_iqtree_on_augmented_topologies_{seq_id}_{edge}.txt")
    shell:
        """
        if test -f "{input.topology_file}_branch_length.iqtree"; then
          echo "Ignoring iqtree run on {input.topology_file}_branch_length, since it is already done."
        else
          iqtree -s {input.msa} -m $(cat {input.full_model}) -te {input.topology_file} --prefix {input.topology_file}_branch_length
        fi
        """

# this rule adds a specific key to the global dictionary
rule extract_reattachment_data_per_taxon_and_edge:
    input:
        ready_to_run=expand(data_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.run_iqtree.done", edge=get_attachment_edge_indices(input_alignment)),
        treefiles=expand(data_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile", edge=get_attachment_edge_indices(input_alignment)),
        mlfiles=expand(data_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.iqtree", edge=get_attachment_edge_indices(input_alignment)),
        full_tree_file=rules.run_iqtree_on_full_dataset.output.tree
    output:
        csv_name=data_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv"
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/extract_reattachment_data_per_taxon_and_edge.py"

rule aggregate_reattachment_data_per_taxon:
    input:
        full_treefile=data_folder+input_alignment+".treefile",
        treefiles=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile", edge=get_attachment_edge_indices(input_alignment), seq_id=get_seq_ids(input_alignment)),
        reduced_treefile=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        reduced_tree_mlfile=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.iqtree", seq_id=get_seq_ids(input_alignment)),
        taxon_dictionary=expand(data_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv", seq_id=get_seq_ids(input_alignment))
    output:
        output_csv=data_folder+"reduced_alignments/reattachment_data_per_taxon.csv"
    params:
        seq_ids=get_seq_ids(input_alignment),
        edges=get_attachment_edge_indices(input_alignment),
    script:
        "scripts/aggregate_reattachment_data_per_taxon.py"

# create plots
rule create_plots:
    input:
        taxon_df_csv=rules.aggregate_reattachment_data_per_taxon.output.output_csv,
        taxon_edge_df_csv=expand(data_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv", seq_id=get_seq_ids(input_alignment)),
        reduced_trees=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", 
        seq_id=get_seq_ids(input_alignment)),
        mldist_file=data_folder+input_alignment+".mldist"
    output:
        plots_folder+"edpl_vs_tii.pdf",
        plots_folder+"likelihood_swarmplots.pdf",
        plots_folder+"seq_distance_vs_tii.pdf",
        plots_folder+"bootstrap_vs_tii.pdf",
        plots_folder+"taxon_height_vs_tii.pdf"
    params:
        plots_folder=plots_folder
    script:
        "scripts/create_plots.py"

# create single file with timing breakdowns for the rules
rule combine_benchmark_outputs:
    input:
        iqtree_model=data_folder+"benchmarking/benchmark_model_test_iqtree.txt",
        remove_taxon=expand(data_folder+"benchmarking/benchmark_remove_sequence_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        iqtree_whole_taxon_set=data_folder+"benchmarking/benchmark_run_iqtree_on_full_dataset.txt",
        iqtree_restricted_taxon_set=expand(data_folder+"benchmarking/benchmark_run_iqtree_restricted_alignments_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        reattach_taxon=expand(data_folder+"benchmarking/benchmark_reattach_removed_sequence_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        iqtree_augmented_topologies=expand(data_folder+"benchmarking/benchmark_run_iqtree_on_augmented_topologies_{seq_id}_{edge}.txt", seq_id=get_seq_ids(input_alignment), edge=get_attachment_edge_indices(input_alignment))
    output:
        output_file=data_folder+"benchmarking_data.csv"
    params:
        iqtree_model=data_folder+"benchmarking/benchmark_model_test_iqtree.txt",
        remove_taxon=expand(data_folder+"benchmarking/benchmark_remove_sequence_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        iqtree_whole_taxon_set=data_folder+"benchmarking/benchmark_run_iqtree_on_full_dataset.txt",
        iqtree_restricted_taxon_set=expand(data_folder+"benchmarking/benchmark_run_iqtree_restricted_alignments_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        reattach_taxon=expand(data_folder+"benchmarking/benchmark_reattach_removed_sequence_{seq_id}.txt", seq_id=get_seq_ids(input_alignment)),
        iqtree_augmented_topologies=expand(data_folder+"benchmarking/benchmark_run_iqtree_on_augmented_topologies_{seq_id}_{edge}.txt", seq_id=get_seq_ids(input_alignment), edge=get_attachment_edge_indices(input_alignment))
    script:
        "scripts/combine_benchmark_outputs.py"
