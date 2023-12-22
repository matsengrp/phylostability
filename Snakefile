import os

# Construct paths relative to the working directory
snakefile_dir = workflow.basedir
current_working_directory = os.getcwd()

config_path = os.path.join(snakefile_dir, "config.yaml")
subsampling_snakefile_path = os.path.join(snakefile_dir, "Subsampling_snakefile")
analysis_snakefile_path = os.path.join(snakefile_dir, "Analysis_snakefile")

configfile: config_path

num_cores = config["num_cores"]
data_folder = config["data_folder"]


module subsampling:
    snakefile:
        subsampling_snakefile_path
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
        snakemake --snakefile {analysis_snakefile_path} -d {current_working_directory} --cores {num_cores}
        rm {current_working_directory}/{data_folder}/convert_input_to_fasta.done
        """

use rule * from subsampling as subsampling_*
