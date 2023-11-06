from Bio import SeqIO

input_msa_file = snakemake.input.msa
without_taxon_msa_files = snakemake.output.without_taxon_msa
taxon_msa_files = snakemake.output.taxon_msa

print("extract single taxon msa")

with open(input_msa_file, "r") as infile:
    records = list(SeqIO.parse(infile, "fasta"))

for record in records:
    without_taxon_msa_file = [
        f for f in without_taxon_msa_files if "/" + record.id + "/" in f
    ][0]
    taxon_msa_file = [f for f in taxon_msa_files if "/" + record.id + "/" in f][0]

    # Save the specific taxon to a separate file
    with open(without_taxon_msa_file, "w") as outfile_taxon:
        for other_record in records:
            if other_record.id != record.id:
                SeqIO.write(other_record, outfile_taxon, "fasta")

    # Save the alignment without the specific taxon
    with open(taxon_msa_file, "w") as outfile_removed:
        SeqIO.write(record, outfile_removed, "fasta")
