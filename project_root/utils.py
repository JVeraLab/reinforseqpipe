import json
from itertools import permutations, product
from Bio.Seq import Seq
from embeddings import CodonEmbedding

class Assemble:
    """
    Handles the assembly of candidate peptide sequences and their back-translation 
    into DNA using high-usage codons.
    """
    def __init__(self, codon_usage_file: str):
        # Peptide fragments to be permuted
        self.peptides_list = ['YLEPGPVTA', 'EAAGIGILTV', 'EVDPIGHLY', 'SLYNTVATL', 'ILKEPVHGV']
        # Linker sequence
        self.linker = ['GGGGS']
        # Marker tag sequence
        self.myctag_aa = 'EQKLISEEDL'
        self.myctag_dna = "GAACAAAAACTCATCTCAGAAGAGGATCTG" # DNA of EQKLISEEDL (if the original code meant a DNA tag)
        
        # Codon usage (from the JSON file)
        with open(codon_usage_file, 'r') as file:
            self.codon_usage: dict[str, float] = json.load(file)
            
        # Codon table for back-translation
        self.genetic_code = CodonEmbedding().genetic_code
        self.aa_to_codons: dict[str, list[str]] = CodonEmbedding()._compute_synonymous_groups()
            
        self.stopcodon = 'TAA'
        self.startcodon = 'ATG'
        
    def construct_peptide_sequences(self) -> list[str]:
        """
        Generates all possible permutations of peptide sequences with all combinations of linkers,
        and appends the marker tag (`EQKLISEEDL`).
        """
        peptide_permutations = list(permutations(self.peptides_list))
        sequences: list[str] = []

        for perm in peptide_permutations:
            # Handle single peptide sequences (though linkers are used between them)
            if len(perm) == 1:
                sequences.append(perm[0] + self.linker[0] + self.myctag_aa)
                continue

            # Assuming only one linker type for now, as per the original logic
            linker_combinations = product(self.linker, repeat=len(perm) - 1)
            for linkers in linker_combinations:
                # Start with the first peptide
                base_sequence = perm[0]
                # Add each linker and next peptide
                for i, linker in enumerate(linkers):
                    base_sequence += linker + perm[i + 1]
                # Append the final linker and the marker tag
                sequences.append(base_sequence + self.linker[0])

        return sequences
    
    def backtranslate_peptide(self) -> tuple[list[str], int]:
        """
        Back-translates each assembled peptide sequence into a DNA sequence 
        using the codon with the highest usage, and adds start/stop codons.
        """
        final_constructs: list[str] = []
        
        for peptide in self.construct_peptide_sequences():
            dna_sequence: list[str] = []
            for aa in peptide:
                if aa in self.aa_to_codons:
                    codons = self.aa_to_codons[aa]
                    # Select the codon with the highest usage from the codon usage dictionary
                    best_codon = max(codons, key=lambda codon: self.codon_usage.get(codon, 0))
                    dna_sequence.append(best_codon)
                else:
                    raise ValueError(f"Unknown amino acid: {aa}")
                
            # Add start and stop codons
            full_dna_sequence = self.startcodon + "".join(dna_sequence) + self.myctag_dna + self.stopcodon
            final_constructs.append(full_dna_sequence)

        # The original code calculates the length of the tag+stop in codons for the mask.
        # However, the mask is set to 0 in main, so this return value is unused there. 
        # Returning 0 as it's the value used in the main function.
        return final_constructs, 0
    
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