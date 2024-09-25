from Bio import SeqIO

input_fasta = snakemake.input.alignment_name
output_fasta = snakemake.output.output_file

seen = []
records = []

for record in SeqIO.parse(input_fasta, "fasta"):  
    if str(record.seq) not in seen:
        seen.append(str(record.seq))
        records.append(record)

SeqIO.write(records, output_fasta, "fasta")

