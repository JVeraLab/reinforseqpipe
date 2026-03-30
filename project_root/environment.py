
import math
import torch
import RNA
#from CAI import CAI

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
        synonymous_dictionary = {}
        for codon, amino_acid in self.genetics_code.items():
            if amino_acid not in synonymous_dictionary:
                synonymous_dictionary[amino_acid] = []
            synonymous_dictionary[amino_acid].append(codon)
        return synonymous_dictionary

    """
    def calculate_cai(self, mrna_sequence):
        if isinstance(mrna_sequence, list):
            mrna_sequence = "".join(mrna_sequence)
            
        # The CAI library function takes the sequence and the precalculated weights dictionary
        return CAI(mrna_sequence, weights=self.weights)
    """  
    def calculate_cai(self, mrna_sequence):
        """
        Manual CAI calculation without external libraries.
        To avoid the problems with environment setup this approach was used.
        Calculates the geometric mean of codon weights using logarithms to prevent numerical underflow.
        """
        if isinstance(mrna_sequence, list):
            mrna_sequence = "".join(mrna_sequence)
            
        sequence_length_in_bases = len(mrna_sequence)
        
        sum_of_logarithms = 0.0
        valid_codon_count = 0
        
        for codon_start_index in range(0, sequence_length_in_bases, 3):
            current_codon = mrna_sequence[codon_start_index : codon_start_index + 3]
            
            if len(current_codon) == 3 and current_codon in self.weights:
                codon_weight = self.weights[current_codon]
                if codon_weight == 0: # In very rare instances the weight can be zero. To avoid the error code returnz zero.
                    return 0.0
                if codon_weight > 0:
                    sum_of_logarithms += math.log(codon_weight)
                valid_codon_count += 1
                
        if valid_codon_count == 0:
            return 0.0
            
        codon_adaptation_index = math.exp(sum_of_logarithms / valid_codon_count)
        
        return codon_adaptation_index

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

    def calculate_mfecaiscore(self, mfe, cai, sequence_length, factor):
        # Safety check for log10 because in a very rare scenario the CAI can be loaded as zero value
        if cai <= 0: 
            cai = 0.0001 
        return abs(mfe - sequence_length * factor * math.log10(cai))
    
    def step(self, position, new_codon):
        info = {} # Pre-initialize to avoid UnboundLocalError
        old_codon = self.current_mrna[position]
        old_amino_acid = self.genetics_code[old_codon]
        new_amino_acid = self.genetics_code[new_codon]

        # Check for non-synonymous substitution and terminate episode if it occurs.
        if old_amino_acid != new_amino_acid:
            reward = -1.0
            done = True
            info = {"error": "Non-synonymous substitution attempted"}
            return self.get_state(), reward, done, info

        # Update the codon at the specified position
        self.current_mrna[position] = new_codon
        self.steps_taken += 1
        
        dna_sequence = "".join(self.current_mrna)
        new_cai = self.calculate_cai(dna_sequence)
        new_mfe = self.calculate_mfe(dna_sequence)

        # Calcaulte the absolute changes in CAI and MFE for tracking and stats.
        cai_change = new_cai - self.prev_cai
        mfe_change = -(new_mfe - self.prev_mfe)

        self.cai_changes.append(abs(cai_change))
        self.mfe_changes.append(abs(mfe_change))

        # Reward equation based on MFECAIlambda score change from Zhang et al., 2021 paper
        total_reward = (self.calculate_mfecaiscore(new_mfe, new_cai, self.length/3, self.factor) - 
                        self.calculate_mfecaiscore(self.prev_mfe, self.prev_cai, self.length/3, self.factor)) 

        # Update previous CAI and MFE for the next step
        self.prev_cai = new_cai
        self.prev_mfe = new_mfe
        done = self.steps_taken >= self.max_steps
        
        info = {
            "cai_change": cai_change,
            "mfe_change": mfe_change,
        }
        return self.get_state(), total_reward, done, info

    def get_allowed_actions(self, position):
        """
        Returns a list of legal codons that can be substituted at the given position.
        """

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

