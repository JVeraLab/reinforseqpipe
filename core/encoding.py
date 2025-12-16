# core/encoding.py

import torch
from Bio.Seq import Seq

# ==========================================================================================================================================================
# Encoding of Codons
# ==========================================================================================================================================================
class CodonEmbedding:
    """
    Handles the encoding of nucleotides and codons into a numerical vector representation
    for the DQN model's state input.
    """
    def __init__(self):
        # One-hot encoding for each nucleotide (A=0001, T=0010, C=0100, G=1000)
        self.nucleotide_encoding = {
            "A": "0001", "T": "0010", "C": "0100", "G": "1000"
        }
        # The codon vector size is 3 nucleotides * 4 bits/nucleotide = 12 bits.
        self.VECTOR_SIZE = 12
        
        # Standard genetic code mapping codons to amino acids.
        self.genetic_code = {
            "AAA": "K", "AAG": "K", "AAC": "N", "AAT": "N",
            "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
            "AGA": "R", "AGC": "S", "AGG": "R", "AGT": "S",
            "ATA": "I", "ATC": "I", "ATG": "M", "ATT": "I",
            "CAA": "Q", "CAG": "Q", "CAC": "H", "CAT": "H",
            "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
            "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
            "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
            "GAA": "E", "GAG": "E", "GAC": "D", "GAT": "D",
            "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
            "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
            "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
            "TAA": "*", "TAG": "*", "TAC": "Y", "TAT": "Y",
            "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
            "TGA": "*", "TGC": "C", "TGG": "W", "TGT": "C",
            "TTA": "L", "TTC": "F", "TTG": "L", "TTT": "F"
        }
        self.codons = list(self.genetic_code.keys())
        
    def _compute_synonymous_groups(self) -> dict[str, list[str]]:
        """Creates a mapping from amino acid to its synonymous codons."""
        syn_dict: dict[str, list[str]] = {}
        for codon, aa in self.genetic_code.items():
            if aa not in syn_dict:
                syn_dict[aa] = []
            syn_dict[aa].append(codon)
        return syn_dict

    def compute_codon_vector(self, codon: str) -> list[int]:
        """
        Encodes a 3-nucleotide codon into a 12-bit vector by concatenating 
        the 4-bit one-hot encoding of each nucleotide.
        """
        # Concatenate the 4-bit encoding for N1, N2, and N3.
        bin12_str = "".join(self.nucleotide_encoding[nt] for nt in codon)
        # Convert the string of '0's and '1's into a list of integers.
        bin12 = [int(char) for char in bin12_str]
        return bin12 

# ==========================================================================================================================================================
# Verification of Final Sequence
# ==========================================================================================================================================================
class Verification:
    """
    Accessory class to verify that the original and optimized sequences code 
    for the same polypeptide. All other metrics (CAI, MFE, GC) are calculated 
    and reported in the worker function.
    """
    def __init__(self, original_sequence: str, final_sequence: str):
        self.original_sequence = original_sequence
        self.final_sequence = final_sequence

    def compare_dna_sequences(self) -> bool:
        """
        Translates both DNA sequences to check if they result in the same peptide sequence.
        This is the only necessary verification for sequence validity.
        """
        # Seq(dna).translate() automatically handles the genetic code
        original_peptide = Seq(self.original_sequence).translate()
        final_peptide = Seq(self.final_sequence).translate()

        return original_peptide == final_peptide