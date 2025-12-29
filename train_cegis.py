import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
import gymnasium.spaces as spaces
from types import SimpleNamespace

sys.path.append(os.getcwd())

from utils.env import make_env
from certificate import CertificateNet
from model import ACModel
from data_gen import generate_transitions

# --- CONFIGURATION ---
ENV_NAME = "MiniGrid-Dynamic-Obstacles-8x8-v0"
MODEL_PATH = "storage/Phase1-random/status.pt" 

# Paper Parameters
P_SATISFACTION = 0.99
UNSAFE_BARRIER = 1.0 / (1.0 - P_SATISFACTION) # ~100
EPSILON = 0.05
MAX_ITERATIONS = 20  # Max number of CEGIS loops
EPOCHS_PER_ITER = 1000 # How long to train per loop

def get_initial_datasets(states, masks, ratio=0.1):
    """
    Paper Sec 4.1: Initialized with a coarse discretization.
    We simulate this by picking a random subset of the full grid.
    """
    total = len(states)
    size = int(total * ratio)
    indices = torch.randperm(total)[:size]
    
    # We maintain INDICES of the states to train on
    C_indices = indices.tolist()
    return set(C_indices) # Use a set for easy adding

def train_learner(policy, certificate, optimizer, states, next_states, masks, train_indices, iteration):
    """
    The LEARNER Module (Paper Sec 4.2)
    Minimizes Eq (3) on the current training set (C_indices).
    """
    policy.train()
    certificate.train()
    
    # Convert set to tensor for indexing
    idx_tensor = torch.tensor(list(train_indices), device=states.device).long()
    
    # Subsample the universe to get current Training Set
    C_states = states[idx_tensor]
    C_next = next_states[idx_tensor]
    
    # Masks need to be sliced to match the subset
    C_masks = {k: v[idx_tensor] for k, v in masks.items()}
    
    print(f"\n--- Learner Loop {iteration} (Training on {len(idx_tensor)} states) ---")
    
    for epoch in range(EPOCHS_PER_ITER):
        optimizer.zero_grad()
        
        # Annealing (Soft -> Hard)
        temp = max(0.01, 1.0 - (epoch / EPOCHS_PER_ITER))

        # 1. Forward Pass
        V_curr = certificate(C_states)
        policy_input = SimpleNamespace(image=C_states)
        dist, _ , _= policy(policy_input,None)
        probs = F.softmax(dist.logits / temp, dim=1)
        
        # 2. Expected Next V
        N, n_actions, H, W, C = C_next.shape
        flat_next = C_next.view(-1, H, W, C)
        V_next_all = certificate(flat_next).view(N, n_actions)
        V_next_expected = (probs * V_next_all).sum(dim=1, keepdim=True)
        
        # 3. RASM LOSSES (Eq 3)
        
        # L_Init: Max(V(start) - 1, 0)
        if C_masks["start"].any():
            loss_init = torch.max(torch.relu(V_curr[C_masks["start"]] - 1.0))
        else:
            loss_init = torch.tensor(0.0, device=states.device)

        # L_Unsafe: Max(Barrier - V(unsafe), 0)
        if C_masks["unsafe"].any():
            loss_unsafe = torch.max(torch.relu(UNSAFE_BARRIER - V_curr[C_masks["unsafe"]]))
        else:
            loss_unsafe = torch.tensor(0.0, device=states.device)

        # L_Expected: Mean(Relu(V_next - V_curr + eps))
        if C_masks["safe"].any():
            # Note: Paper Eq 3 sums violations. We use mean for gradient stability.
            violation = (V_next_expected[C_masks["safe"]] + EPSILON) - V_curr[C_masks["safe"]]
            loss_expected = torch.mean(torch.relu(violation))
        else:
            loss_expected = torch.tensor(0.0, device=states.device)
            
        loss = loss_init + loss_unsafe + loss_expected
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Ep {epoch} | Loss: {loss.item():.4f} | Init: {loss_init.item():.4f} | Unsafe: {loss_unsafe.item():.4f} | Decr: {loss_expected.item():.4f}")

    return policy, certificate

def run_verifier(policy, certificate, states, next_states, masks):
    """
    The VERIFIER Module (Paper Sec 4.3)
    Checks ALL states. Returns indices of violations (Counterexamples).
    """
    policy.eval()
    certificate.eval()
    
    new_counterexamples = []
    
    with torch.no_grad():
        # --- FIX: flatten V_curr to (N) so it matches masks ---
        V_curr = certificate(states).view(-1) 
        
        # Wrap input for policy
        policy_input = SimpleNamespace(image=states)
            
        # Unpack 3 values
        dist, _, _ = policy(policy_input, None) 
        best_actions = dist.logits.argmax(dim=1)
        
        # Select specific next state
        batch_indices = torch.arange(len(states), device=states.device)
        chosen_next = next_states[batch_indices, best_actions]
        
        # --- FIX: flatten V_next to (N) ---
        V_next = certificate(chosen_next).view(-1)
        
        # --- CHECK CONDITIONS GLOBALLY ---
        
        # 1. Init Violations
        fail_init = (masks["start"] & (V_curr > 1.0)).nonzero(as_tuple=True)[0]
        
        # 2. Unsafe Violations (V < Barrier)
        fail_unsafe = (masks["unsafe"] & (V_curr < UNSAFE_BARRIER)).nonzero(as_tuple=True)[0]
        
        # 3. Decrease Violations
        # V_next - V_curr + eps > 0
        violation_gap = (V_next - V_curr + EPSILON)
        fail_decr = (masks["safe"] & (violation_gap > 1e-5)).nonzero(as_tuple=True)[0]
        
        # Combine unique indices
        all_fails = torch.cat((fail_init, fail_unsafe, fail_decr)).unique()
        new_counterexamples = all_fails.tolist()
        
    print(f"Verifier Found: {len(new_counterexamples)} unique violations.")
    print(f"  - Init: {len(fail_init)} | Unsafe: {len(fail_unsafe)} | Decr: {len(fail_decr)}")
    
    return new_counterexamples

def main():
    # 1. Setup Universe (All possible states)
    env = make_env(ENV_NAME, seed=1)
    states, next_states, masks = generate_transitions(env)
    
    # Assume (1,1) is start if not defined
    if "start" not in masks:
        # Create a dummy start mask if your generate_transitions doesn't return one
        # For now, let's treat 'safe' states as potential starts or define logic
        pass 

    # Check if observation_space is a Box (standard for ImgObsWrapper)
    if hasattr(env.observation_space, "shape"):
        image_shape = env.observation_space.shape
    else:
        # Fallback for unexpected types
        image_shape = (8, 8, 3)
    dummy_obs_space = {"image": image_shape}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = states.to(device); next_states = next_states.to(device)
    for k in masks: masks[k] = masks[k].to(device)

    # 2. Initialize Networks
    policy = ACModel(dummy_obs_space, env.action_space)
    checkpoint = torch.load(MODEL_PATH)
    if "model_state" in checkpoint:
        policy.load_state_dict(checkpoint["model_state"])
    else:
        policy.load_state_dict(checkpoint)
        
    certificate = CertificateNet(dummy_obs_space)
    certificate.to(device); policy.to(device)
    
    # Optimizer for BOTH
    optimizer = optim.Adam(list(policy.parameters()) + list(certificate.parameters()), lr=0.0003)

    # 3. Initialize Dataset (C_expected, etc.)
    # Start with a random subset (Coarse abstraction)
    train_indices = get_initial_datasets(states, masks, ratio=0.2) 
    
    # --- CEGIS LOOP ---
    for i in range(MAX_ITERATIONS):
        print(f"\n================ ITERATION {i+1} ================")
        
        # Step A: Learner (Train on current set)
        policy, certificate = train_learner(
            policy, certificate, optimizer, 
            states, next_states, masks, train_indices, i
        )
        
        # Step B: Verifier (Check ALL states)
        counterexamples = run_verifier(policy, certificate, states, next_states, masks)
        
        # Step C: Convergence Check
        if len(counterexamples) == 0:
            print("\n CEGIS CONVERGED! No counterexamples found.")
            break
            
        # Step D: Aggregation (Add Counterexamples to Training Set)
        # Paper: "extended by counterexamples computed by the verifier" 
        initial_len = len(train_indices)
        train_indices.update(counterexamples)
        print(f"Added {len(train_indices) - initial_len} new unique states to training set.")

    # Save
    torch.save(policy.state_dict(), "cegis_policy.pt")
    torch.save(certificate.state_dict(), "cegis_certificate.pt")

if __name__ == "__main__":
    main()