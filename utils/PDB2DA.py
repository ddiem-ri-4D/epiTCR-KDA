import numpy as np
import pandas as pd
import random
import glob
import itertools
from biopandas.pdb import PandasPdb
import os
import Bio.PDB
from math import degrees

def ramachandran_type(phi, psi):
    if -180 <= phi <= 180 and -180 <= psi <= 180:
        if -180 <= phi <= -60 and -180 <= psi <= 180:
            return "A"
        elif -180 <= phi <= -60 and -90 <= psi <= 180:
            return "B"
        elif -180 <= phi <= -60 and -180 <= psi <= -90:
            return "C"
        elif -60 <= phi <= 0 and -180 <= psi <= 180:
            return "D"
        elif -60 <= phi <= 0 and -90 <= psi <= 180:
            return "E"
        elif -60 <= phi <= 0 and -180 <= psi <= -90:
            return "F"
        elif 0 <= phi <= 180 and -180 <= psi <= 180:
            return "G"
        elif 0 <= phi <= 180 and -90 <= psi <= 180:
            return "H"
        elif 0 <= phi <= 180 and -180 <= psi <= -90:
            return "I"
        else:
            return "Unknown"
    else:
        return "Unknown"

folder_path = "./datasets/DA_TSVFiles"
file_extension = ".pdb"

# Lấy danh sách các file trong thư mục FINAL_PEP/
file_list = [filename for filename in os.listdir(folder_path) if filename.endswith(file_extension)]

for pdb_file in file_list:
    pdb_code = os.path.splitext(pdb_file)[0]
    link_code = "./datasets/3DS_PDBFiles" + pdb_code
    
    print("About to load Bio.PDB and the PDB file...")
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, os.path.join(folder_path, pdb_file))
    print("Done")

    print("About to save angles to file...")
    output_file = open("%s.tsv" % link_code, "w")
    print(output_file)
    for model in structure:
        for chain in model:
            print("Chain %s" % str(chain.id))
            polypeptides = Bio.PDB.CaPPBuilder().build_peptides(chain)
            for poly_index, poly in enumerate(polypeptides):
                phi_psi = poly.get_phi_psi_list()
                for res_index, residue in enumerate(poly):
                    phi, psi = phi_psi[res_index]
                    if phi is not None and psi is not None:
                        # Don't write output when missing an angle
                        phi_angle = degrees(phi)
                        psi_angle = degrees(psi)
                        output_file.write("%s:Chain%s:%s%i\t%f\t%f\t%s\n" \
                            % (pdb_code, str(chain.id), residue.resname,
                               residue.id[1], phi_angle, psi_angle,
                               ramachandran_type(phi_angle, psi_angle)))
    output_file.close()
    print("Done")
