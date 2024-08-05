import os

def read2seq(sequence_list, directory):
    tsv_files = []
    for file in os.listdir(directory):
        if file.endswith(".tsv"):
            tsv_files.append(file[:-4])
    final_tsv = [seq for seq in sequence_list if seq not in tsv_files]
    return final_tsv

peptide_directory = './datasets/DA_TSVFiles'

# Assuming DF is defined somewhere in your actual code
# DF should be a pandas DataFrame that contains a column named 'CDR3b'
sequence_list = read2seq(DF.CDR3b.unique(), peptide_directory)

output_file = "./LST_FORRUN.fasta"

with open(output_file, 'w') as file:
    for sequence in sequence_list:
        file.write(">%s\n" % sequence)
        file.write("%s\n" % sequence)
    print('Done')
