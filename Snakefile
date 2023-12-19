configfile: "config.yaml"

num_cores = config["num_cores"]
data_folder = config["data_folder"]

module subsampling:
    snakefile:
        "Subsampling_snakefile"
    config:
        config


rule this_all:
    input:
        "run_analysis.done"

rule run_analysis:
    input:
        data_folder+"/convert_input_to_fasta.done"
    output:
        temp(touch("run_analysis.done"))
    shell:
        """
        snakemake --snakefile Analysis_snakefile --cores {num_cores} --rerun-incomplete
        rm {data_folder}/convert_input_to_fasta.done
        """


use rule * from subsampling as subsampling_*