# data/assemble.py

import json
from itertools import permutations, product
from core.encoding import CodonEmbedding

# ==========================================================================================================================================================
# Assembling of Candidate Peptides
# ==========================================================================================================================================================
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
        self.myctag_dna = "GAACAAAAACTCATCTCAGAAGAGGATCTG" # DNA of EQKLISEEDL
        
        # Codon usage (from the JSON file)
        with open(codon_usage_file, 'r') as file:
            self.codon_usage: dict[str, float] = json.load(file)
            
        # Codon table for back-translation
        codon_embedder = CodonEmbedding()
        self.genetic_code = codon_embedder.genetic_code
        self.aa_to_codons: dict[str, list[str]] = codon_embedder._compute_synonymous_groups()
            
        self.stopcodon = 'TAA'
        self.startcodon = 'ATG'
        
    def construct_peptide_sequences(self) -> list[str]:
        """
        Generates all possible permutations of peptide sequences with all combinations of linkers,
        and appends the marker tag (`EQKLISEEDL`).
        """
        # 
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
                sequences.append(base_sequence + self.linker[0] + self.myctag_aa)

        return sequences
    
    def backtranslate_constructed_peptides(self) -> str:
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
            full_dna_sequence = self.startcodon + "".join(dna_sequence) + self.stopcodon
            final_constructs.append(full_dna_sequence)

        # The original code returns an unused value for mask_last_n_codons, 
        # which is then overridden to 0 in main. We return 0 here to match the logic.
        return final_constructs
    
    def backtranslate_peptide(self, given_peptide) -> str:
        dna_sequence: list[str] = []
        for aa in given_peptide:
            if aa in self.aa_to_codons:
                codons = self.aa_to_codons[aa]
                # Select the codon with the highest usage from the codon usage dictionary
                best_codon = max(codons, key=lambda codon: self.codon_usage.get(codon, 0))
                dna_sequence.append(best_codon)
            else:
                raise ValueError(f"Unknown amino acid: {aa}")
        dna_sequence = "".join(dna_sequence)
        return dna_sequence