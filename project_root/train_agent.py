import torch 
from environment import MRNAEnv, compute_allowed_mask
from models import DQNAgent
from embeddings import CodonEmbedding

def train_agent_dqn(original_dna: str, device: torch.device, weights: dict, num_episodes: int, 
                    max_steps: int, factor: float, mask_last_n_codons: int) -> tuple:
    """
    Main training function for the DQN agent.
    
    Returns: A tuple containing various initial/final metrics and episode data lists.
    """
    # Split DNA sequence into a list of codons.
    original_mrna = [original_dna[i:i+3] for i in range(0, len(original_dna), 3)]
    
    # Initialize embedding helper
    computing_codon = CodonEmbedding() 
    
    # Initialize the Environment
    env = MRNAEnv(
        original_mrna,
        weights,
        computing_codon.compute_codon_vector,
        max_steps=max_steps, # Max steps is length in nucleotides.
        factor=factor,
        mask_last_n_codons=mask_last_n_codons 
    )
    
    initial_cai, initial_mfe = env.get_initial_state()
    seq_length = len(original_mrna)
    
    # Hyperparameters (preserved from hyperparameters optimziation for specific usage case)
    input_size = 12 
    num_codons = 64
    lr = 1e-2
    gamma=0.95
    target_update = 30
    batch_size = 64
    hidden_size = 128
    epsilon_decay = 1.5 * len(original_dna)
    
    # Initialize the Agent
    agent = DQNAgent(input_size, hidden_size, seq_length, num_codons, device, lr, gamma, 
                     target_update, batch_size, epsilon_decay, epsilon_start=1.0, 
                     epsilon_final=0.05, buffer_capacity=10000)
    
    # Tracking variables
    total_steps = 0
    best_cai = initial_cai
    best_mfe = initial_mfe
    best_score = env.calculate_mfecaiscore(initial_mfe, initial_cai, seq_length, factor)
    best_sequence = "".join(original_mrna)
    best_sequences: list[str] = []
    
    # Data logging
    episode_number: list[int] = []
    episode_cai: list[float] = []
    episode_mfe: list[float] = []
    episode_score: list[float] = []

    for episode in range(1, num_episodes + 1):
        # Reset environment
        state_tensor = env.reset() 
        state = state_tensor.unsqueeze(0).to(device) # shape: (1, seq_length, vector_size)
        episode_reward = 0.0
        
        for t in range(max_steps):
            # 1. Select action
            action = agent.select_action(state, env)
            position, codon_idx = agent.action_to_components(action)
            chosen_codon = env.codons[codon_idx]
            
            # 2. Take the step in the environment
            next_state_tensor, reward, done, info = env.step(position, chosen_codon)
            episode_reward += reward
            next_state = next_state_tensor.unsqueeze(0).to(device)
            
            # 3. Compute allowed mask for the *next* state 
            # (required for target Q-value calculation in update_policy)
            allowed_mask = compute_allowed_mask(env)
            
            # 4. Store transition in replay memory
            agent.memory.push(state, action, reward, next_state, allowed_mask.cpu(), done)
            
            state = next_state
            total_steps += 1
            
            # 5. Perform optimization
            agent.update_policy()
            
            if done:
                if "error" in info:
                    print(f"Episode {episode} terminated early: {info['error']}")
                break

        # 6. Periodically update the target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # 7. Log episode metrics
        current_sequence = "".join(env.current_mrna)
        current_cai = env.calculate_cai(current_sequence)
        current_mfe = env.calculate_mfe(current_sequence)
        current_score = env.calculate_mfecaiscore(current_mfe, current_cai, seq_length, factor)
        
        episode_number.append(episode)
        episode_cai.append(current_cai)
        episode_mfe.append(current_mfe)
        episode_score.append(current_score)
        print(f'THe episode MFE:{episode_mfe} andepisodeCAI:{episode_cai}')
        # 8. Update best sequence
        # The goal is to MINIMIZE the score, so we check for < best_score.
        if current_score > best_score:
            best_cai = current_cai
            best_mfe = current_mfe
            best_score = current_score
            best_sequence = current_sequence
            best_sequences.append(best_sequence) # Log all best sequences (optional: only store the final one)
            

            
    return (initial_cai, initial_mfe, env.calculate_gc_content(original_dna), 
            best_sequence, best_cai, best_mfe, env.calculate_gc_content(best_sequence), 
            episode_number, episode_cai, episode_mfe, episode_score)