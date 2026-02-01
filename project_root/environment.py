import torch
import math
import RNA
from embeddings import CodonEmbedding
from Bio.SeqUtils import CodonAdaptationIndex as CAI_Bio

class MRNAEnv:
    """
    Environment for codon optimization via synonymous substitutions.
    """
    def __init__(self, original_mrna, weights, embedding_function, max_steps, factor, mask_last_n_codons):
        self.factor = factor
        self.original_mrna = original_mrna
        self.length = len("".join(self.original_mrna))
        self.current_mrna = original_mrna.copy()
        self.max_steps = max_steps
        self.steps_taken = 0
        self.mask_last_n_codons = mask_last_n_codons
        self.genetics_code = {
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
        self.codons = list(self.genetics_code.keys())
        self.synonymous_dict = self._compute_synonymous_groups()
        self.prev_cai = None 
        self.prev_mfe = None 
        self.embedding_function = embedding_function
        self.weights = weights

    def _compute_synonymous_groups(self):
        syn_dict = {}
        for codon, aa in self.genetics_code.items():
            if aa not in syn_dict:
                syn_dict[aa] = []
            syn_dict[aa].append(codon)
        return syn_dict

    def calculate_cai(self, mrna_sequence):
        """
        MANUAL CAI FIX: Calculates geometric mean using self.weights.
        """
        codons = [mrna_sequence[i:i+3] for i in range(0, len(mrna_sequence), 3)]
        score = 0.0
        count = 0
        for codon in codons:
            # Use 0.0001 as floor to prevent log(0) errors
            weight = self.weights.get(codon, 0.0001)
            score += math.log(weight)
            count += 1
        return math.exp(score / count) if count > 0 else 0

    def calculate_mfe(self, mrna_sequence):
        return RNA.fold(mrna_sequence.replace("T", "U"))[1]

    def get_initial_state(self):
        dna_seq = "".join(self.original_mrna)
        return self.calculate_cai(dna_seq), self.calculate_mfe(dna_seq)

    def reset(self):
        self.current_mrna = self.original_mrna.copy()
        self.steps_taken = 0
        dna_seq = "".join(self.current_mrna)
        self.prev_cai = self.calculate_cai(dna_seq)
        self.prev_mfe = self.calculate_mfe(dna_seq)
        self.cai_changes = []
        self.mfe_changes = []
        self.cai_pct_changes = []
        self.mfe_pct_changes = []
        return self.get_state()

    def get_state(self):
        state = [self.embedding_function(codon) for codon in self.current_mrna]
        return torch.tensor(state, dtype=torch.float)

    def calculate_percentage_change(self, old_value, new_value):
        if old_value == 0: return 0 
        return (new_value - old_value) / (old_value) * 100

    def calculate_gc_content(self, mrna_sequence):
        if not mrna_sequence: return 0
        return 100 * (mrna_sequence.count('G') + mrna_sequence.count('C')) / len(mrna_sequence)

    def calculate_mfecaiscore(self, mfe, cai, l, factor):
        # Safety check for log10
        if cai <= 0: cai = 0.0001
        return abs(mfe - l * factor * math.log10(cai))

    def step(self, position, new_codon):
        info = {} # Pre-initialize to avoid UnboundLocalError
        old_codon = self.current_mrna[position]
        old_aa = self.genetics_code[old_codon]
        new_aa = self.genetics_code[new_codon]

        if old_aa != new_aa:
            reward = -1.0
            done = True
            info = {"error": "Non-synonymous substitution attempted"}
            return self.get_state(), reward, done, info

        self.current_mrna[position] = new_codon
        self.steps_taken += 1
        
        dna_seq = "".join(self.current_mrna)
        new_cai = self.calculate_cai(dna_seq)
        new_mfe = self.calculate_mfe(dna_seq)

        cai_change = new_cai - self.prev_cai
        mfe_change = -(new_mfe - self.prev_mfe)

        self.cai_changes.append(abs(cai_change))
        self.mfe_changes.append(abs(mfe_change))

        # Reward logic as you provided
        total_reward = (self.calculate_mfecaiscore(new_mfe, new_cai, self.length/3, self.factor) - 
                        self.calculate_mfecaiscore(self.prev_mfe, self.prev_cai, self.length/3, self.factor)) 

        self.prev_cai = new_cai
        self.prev_mfe = new_mfe
        done = self.steps_taken >= self.max_steps
        
        info = {
            "cai_change": cai_change,
            "mfe_change": mfe_change,
        }
        return self.get_state(), total_reward, done, info

    def get_allowed_actions(self, position):
        if position >= len(self.current_mrna) - self.mask_last_n_codons:
            return [self.current_mrna[position]]                
        current_codon = self.current_mrna[position]
        aa = self.genetics_code[current_codon]
        syn_codons = self.synonymous_dict[aa]
        return [codon for codon in syn_codons if codon != current_codon]


def compute_allowed_mask(env: MRNAEnv) -> torch.Tensor:
    """
    Computes a boolean mask for legal actions based on the current environment state.
    
    The mask is flattened to shape (seq_length * num_codons,).
    An action is legal if it corresponds to a synonymous codon change (or no-op for fixed codons).
    """
    seq_length = len(env.current_mrna)
    num_codons = len(env.codons)
    
    # Initialize mask to all False
    allowed_mask = torch.zeros(seq_length, num_codons, dtype=torch.bool)
    
    for i in range(seq_length):
        allowed_codons = env.get_allowed_actions(i)
        # Note: get_allowed_actions returns a list of *substitutions*, 
        # but for positions that cannot be changed (masked or no alternatives), 
        # it returns the current codon. This distinction is crucial for RL action space.
        
        # If no substitutions are available (e.g., Met/Trp, or masked codon), 
        # the no-op (choosing the current codon) is the only legal action.
        current_codon = env.current_mrna[i]
        
        # If the only 'allowed' codon is the current one, it means no substitution is legal.
        is_fixed = (len(allowed_codons) == 1 and allowed_codons[0] == current_codon)
        
        for j in range(num_codons):
            codon = env.codons[j]
            # If the codon is one of the allowed synonymous substitutions
            if codon in allowed_codons:
                allowed_mask[i, j] = True
            # If the position is fixed, the only legal action is the no-op (choosing the current codon again).
            elif is_fixed and codon == current_codon:
                allowed_mask[i, j] = True
            
    return allowed_mask.view(-1)

