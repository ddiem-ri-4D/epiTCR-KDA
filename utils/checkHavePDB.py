#Check have in PDB

import os
thu_muc_pep =  './datasets/DA_TSVFiles'

def read2seq(LST_SEQ, thu_muc):
    danh_sach_tep_tsv = []
    for tep in os.listdir(thu_muc):
        if tep.endswith(".pdb"):
            danh_sach_tep_tsv.append(tep[:-4])
    FINAL_TSV = [cdr3 for cdr3 in LST_SEQ if cdr3 not in danh_sach_tep_tsv]
    return FINAL_TSV

LST_SEQ = read2seq(df_CDR3b, thu_muc_pep)


FILE = "./LST_SEQ_REMAIN_COVID.fasta"
DATA = LST_SEQ

with open(FILE, 'w') as fp:
    for item in DATA:
        fp.write(">%s\n" % item)
        fp.write("%s\n" % item)
    print('Done')
