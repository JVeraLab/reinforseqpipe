from Bio import SeqIO
from Bio.Seq import Seq

def read_fasta_sequences(file_path: str) -> list[str]:
    """
    Reads a FASTA file and returns the string variable to the pipeline.
    """
    PROTEIN_ONLY_CHARS = set('NDRWMEFILPQYSGHKV')
    sequences = {}
    # SeqIO.parse returns an iterator of SeqRecord objects
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            seq_obj = record.seq.upper()
            sequence_id = record.id 
            sequence_string = str(seq_obj)
                
            if not sequence_string:
                continue

            # 1. Check for Definitive Protein Signature
            if any(char in sequence_string for char in PROTEIN_ONLY_CHARS):
                mol_type = 'Protein'
                
            # 2. Check for RNA Signature
            elif 'U' in sequence_string:
                mol_type = 'RNA'
                
            # 3. Check for DNA Signature
            elif 'T' in sequence_string:
                mol_type = 'DNA'
                
            # 4. Ambiguous Case (Only A, C, G, N)
            else:
                mol_type = 'Ambiguous (Assumed DNA)'
                
            sequences[sequence_id] = (sequence_string, mol_type)

    except FileNotFoundError:
        print(f"Error: FASTA file not found at path: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while parsing the FASTA file: {e}")
        return []
        
    print(f"Successfully read {len(sequences)} sequence(s) from {file_path}")
    return sequences
