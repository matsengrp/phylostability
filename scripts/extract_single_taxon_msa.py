from Bio import SeqIO
import os

input_msa_file = snakemake.input.msa
seq_id = snakemake.params.seq_id

# name of dir for this dataset
ds = "/".join(input_msa_file.split("/")[:-1]) + "/reduced_alignments"

with open(input_msa_file, "r") as infile:
    records = list(SeqIO.parse(infile, "fasta"))

record = [record for record in records if seq_id == record.id][0]

seq_id_dir = ds + "/" + seq_id + "/"
if not os.path.exists(seq_id_dir):
    os.makedirs(seq_id_dir)

without_taxon_msa_file = seq_id_dir + "without_taxon.fasta"
taxon_msa_file = seq_id_dir + "single_taxon.fasta"

print(without_taxon_msa_file)
print(taxon_msa_file)

# Save the specific taxon to a separate file
with open(without_taxon_msa_file, "w") as outfile_taxon:
    for other_record in records:
        if other_record.id != seq_id:
            SeqIO.write(other_record, outfile_taxon, "fasta")

# Save the alignment without the specific taxon
with open(taxon_msa_file, "w") as outfile_removed:
    SeqIO.write(record, outfile_removed, "fasta")
