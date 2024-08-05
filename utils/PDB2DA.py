import numpy as np
import pandas as pd
import os
from Bio.PDB import PDBParser, CaPPBuilder
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

pdb_folder_path = "./datasets/DA_TSVFiles"
output_folder_path = "./datasets/3DS_PDBFiles"
file_extension = ".pdb"

# List of PDB files in the folder
pdb_file_list = [filename for filename in os.listdir(pdb_folder_path) if filename.endswith(file_extension)]

for pdb_file in pdb_file_list:
    pdb_code = os.path.splitext(pdb_file)[0]
    output_file_path = os.path.join(output_folder_path, f"{pdb_code}.tsv")
    
    print("Loading PDB file using Bio.PDB...")
    structure = PDBParser().get_structure(pdb_code, os.path.join(pdb_folder_path, pdb_file))
    print("Done loading PDB file.")

    print("Saving angles to file...")
    with open(output_file_path, "w") as output_file:
        for model in structure:
            for chain in model:
                print(f"Processing Chain {chain.id}...")
                polypeptides = CaPPBuilder().build_peptides(chain)
                for poly_index, poly in enumerate(polypeptides):
                    phi_psi = poly.get_phi_psi_list()
                    for res_index, residue in enumerate(poly):
                        phi, psi = phi_psi[res_index]
                        if phi is not None and psi is not None:
                            phi_angle = degrees(phi)
                            psi_angle = degrees(psi)
                            output_file.write(f"{pdb_code}:Chain{chain.id}:{residue.resname}{residue.id[1]}\t{phi_angle}\t{psi_angle}\t{ramachandran_type(phi_angle, psi_angle)}\n")
    print("Done saving angles.")
