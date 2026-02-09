#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
import torch
print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")


# In[2]:


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
from data_gen_mod import generate_transitions
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper


# In[3]:


import matplotlib.pyplot as plt

def plot_learning_curve(history):
    """
    Plots Total Loss and the 3 Component Losses.
    """
    epochs = range(len(history["total"]))

    plt.figure(figsize=(15, 6))

    # 1. Total Loss (Log Scale is often better for big drops)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["total"], label="Total Loss", color='black', linewidth=2)
    plt.yscale('log') # Optional: Makes it easier to see small improvements later
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Total Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Breakdown of Components
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["init"], label="Init (V<=1)", alpha=0.6)
    plt.plot(epochs, history["unsafe"], label="Unsafe (V>=Barrier)", alpha=0.6)
    plt.plot(epochs, history["decr"], label="Decrease (V'<V)", alpha=0.6)
    plt.xlabel("Epochs")
    plt.title("Component Losses (Linear Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# In[7]:





# In[ ]:





# In[ ]:





# In[8]:


# def scratch():
#     V_curr = certificate(C_states)
#     policy_input = SimpleNamespace(image=C_states)
#     dist, _ , _= policy(policy_input,None)
#     probs = F.softmax(dist.logits , dim=1)
#     return 


# In[4]:


def check_if_unsafe(C_decrease):
    obj_layer = C_decrease[:, :, :, 0]
    is_unsafe=(obj_layer == 16).view(C_decrease.size(0), -1).any(dim=1)
    return is_unsafe


# In[5]:


class ObsBatch:
    def __init__(self, tensor):
        self.image = tensor


# In[13]:


def train_learner(env, policy, certificate, optimizer, C_init,C_unsafe,C_decrease, iteration):
    """
    The LEARNER Module (Paper Sec 4.2)
    Minimizes Eq (3) on the current training set (C_indices).
    """
    for param in policy.parameters():
        param.requires_grad = False
    policy.eval()
    certificate.train()

    history = {"total": [], "init": [], "unsafe": [], "decr": []}

    # Probability Constants (Based on your Env randomness)

    print(f"\n--- Learner Loop {iteration} (Training ---")

    for epoch in range(EPOCHS_PER_ITER):
        optimizer.zero_grad()


        # 3. RASM LOSSES (Eq 3)

        # L_Init: Max(V(start) - 1, 0)
        if len(C_init)>0:
            loss_init = torch.max(torch.relu(certificate(C_init) - 1.0))
        else:
            loss_init = torch.tensor(0.0, device=states.device)

        # L_Unsafe: Max(Barrier - V(unsafe), 0)
        if len(C_unsafe)>0:
            loss_unsafe = torch.max(torch.relu(UNSAFE_BARRIER - certificate(C_unsafe)))
        else:
            loss_unsafe = torch.tensor(0.0, device=states.device)

        # L_Expected: Mean(Relu(V_next - V_curr + eps))
        if len(C_decrease)>0:
            # Note: Paper Eq 3 sums violations. We use mean for gradient stability.
            ########## OLD IMPL ##################
            # is_unsafe=check_if_unsafe(C_decrease)
            # safe=~is_unsafe
            # next_states=get_next_states_via_env(env,C_decrease,policy)
            # V_next=certificate(next_states)
            # target=UNSAFE_BARRIER-EPSILON
            # violation=((V_next * safe) + (target * is_unsafe))+EPSILON-certificate(C_decrease)
            ##########################################
            # next_states=get_next_states_via_env(env,C_decrease,policy)
            # V_next=certificate(next_states)
            # violation=V_next +EPSILON-certificate(C_decrease)
            # loss_expected = torch.mean(torch.relu(violation))

            V_curr = certificate(C_decrease)

            # C. Calculate E[V(next)]
            # expected_V_next=[]
            policy_input=ObsBatch(C_decrease)
            dist, _, _ = policy(policy_input,None)
            intended_actions = dist.probs.argmax(dim=1).cpu().numpy()

            next_list=[]
            for i in range(len(C_decrease)):
                state_tensor = C_decrease[i] # (16, 16, 3)
                action = intended_actions[i]
                # state_batch = state.unsqueeze(0)
                # policy_input = ObsBatch(state_batch)
                # dist, _, _ = policy(policy_input,None)
                # intended_action = dist.probs.max(1,keepdim=True)[1]

                exp=float(0.0)
                for _ in range(16):
                    env.load_state_from_tensor(state_tensor)
                    obs,_,_,_,_=env.step(action)
                    obs=obs['image']
                    next_list.append(obs)

                #### MODEL INPUT = env_obs->env_obs['image'] -> tensor for cnn -> to the model ####################

            huge_batch_np = np.stack(next_list)
            huge_batch = torch.tensor(huge_batch_np, dtype=torch.float32, device=C_decrease.device)
            v_next_all = certificate(huge_batch).view(-1)
            v_next_reshaped = v_next_all.view(len(C_decrease), 16)
            expected_V_next = v_next_reshaped.mean(dim=1, keepdim=True)
            # expected_V_next.append(exp)

            # expected_V_next = torch.cat(expected_V_next, dim=0)

            # D. Compute Violation
            violation = expected_V_next + EPSILON - V_curr
            loss_expected = torch.mean(torch.relu(violation))
        else:
            loss_expected = torch.tensor(0.0, device=states.device)

        # loss = loss_init + loss_unsafe + loss_expected
        # W_INIT = 5.0
        # W_UNSAFE = 10   # Increased from 0.05 to 0.5 (10x stronger)
        # W_DECR = 1.0     # Reduced from 5.0 (Stop punishing the jump so hard)

        loss = (loss_init) + (loss_unsafe) + (loss_expected)
        # history["total"].append(loss.item())
        # history["init"].append(loss_init.item())
        # history["unsafe"].append(loss_unsafe.item())
        # history["decr"].append(loss_expected.item())

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            last_layer_weights = certificate.actor[-1].weight
            w_mean = last_layer_weights.mean().item()
            w_grad = last_layer_weights.grad.abs().mean().item() if last_layer_weights.grad is not None else 0.0

            print(f"  Ep {epoch} | Loss: {loss.item():.4f} | "
                  f"Init: {loss_init.item():.4f} | "
                  f"Unsafe: {loss_unsafe.item():.4f} | "
                  f"Decr: {loss_expected.item():.4f}")
            print(f"           > Weights: Mean={w_mean:.5f} | Grad={w_grad:.5f}")
            checkpoint = {
                "model_state": certificate.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, CERTI_PATH)    
    return policy, certificate, history


# In[14]:


def run_verifier(env, policy, certificate, C_init, C_unsafe, C_decrease):
    """
    Checks the Validity of the Learned Certificate against the 3 Conditions.

    1. Init Condition:     V(x) <= 1  for all x in C_init
    2. Unsafe Condition:   V(x) >= Barrier for all x in C_unsafe
    3. Decrease Condition: V(next) + Epsilon <= V(curr) 
                           (Only for states in C_decrease where V(curr) < Barrier)
    """

    print("="*30)
    print("STARTING VERIFICATION")
    print("="*30)

    certificate.eval()
    policy.eval()


    # --- 1. Verify Init Condition ---
    # Requirement: V(start) should be low (e.g. <= 1.0)
    with torch.no_grad():
        v_init = certificate(C_init)

    # Check how many satisfy V(x) <= 1.0
    valid_init_mask = (v_init <= 1.0).view(-1)
    failed_init_mask = ~valid_init_mask
    acc_init = valid_init_mask.float().mean().item() * 100

    print(f"[1] Init Condition (V <= 1.0):")
    print(f"    Passed: {acc_init:.2f}% ({valid_init_mask.sum()}/{len(C_init)})")

    # --- 2. Verify Unsafe Condition ---
    # Requirement: V(unsafe) should be high (>= UNSAFE_BARRIER)
    with torch.no_grad():
        v_unsafe = certificate(C_unsafe)

    valid_unsafe_mask = (v_unsafe >= UNSAFE_BARRIER).view(-1)
    failed_unsafe_mask = ~valid_unsafe_mask
    acc_unsafe = valid_unsafe_mask.float().mean().item() * 100

    print(f"[2] Unsafe Condition (V >= {UNSAFE_BARRIER}):")
    print(f"    Passed: {acc_unsafe:.2f}% ({valid_unsafe_mask.sum()}/{len(C_unsafe)})")

    # --- 3. Verify Decrease Condition ---
    # Requirement: V(next) + e <= V(curr)
    # Applied ONLY to states where V(curr) < UNSAFE_BARRIER

    # with torch.no_grad():
        # A. Get V(curr) for all decrease states
        # v_curr_all = certificate(C_decrease)

        # B. Filter: Find states that are NOT yet considered 'Unsafe' by the certificate
        # Logic: If V(x) is already > Barrier, we don't strictly need it to decrease further 
        # (or the paper says we only check valid/safe-ish states)
        # v_curr_subset = v_curr_all

        # if len(v_curr_subset) == 0:
        #     print("[3] Decrease Condition: Skipped (No states with V < Barrier)")
        #     acc_decrease = 100.0
        # else:
            # C. Get Next States (Using Env + Policy)
            # next_states = get_next_states_via_env(env, C_decrease, policy)

            # # D. Get V(next)
            # v_next_subset = certificate(next_states)

            # # --- CRITICAL: Handle Next State Collisions ---
            # # If the next state is a collision, the Certificate Net might predict garbage.
            # # We must force its value to be UNSAFE_BARRIER to reflect "Bad Outcome".
            # is_next_unsafe = check_if_unsafe(next_states)

##### expectation

    with torch.no_grad():
        v_curr = certificate(C_decrease)
        expected_V_next=[]
        if len(C_decrease) == 0:
            print("[3] Decrease Condition: Skipped (Empty Set)")
        else:
            SAMPLES = 16
            N = len(C_decrease)
            policy_input = ObsBatch(C_decrease) 
            dist, _, _ = policy(policy_input, None)
            intended_actions = dist.probs.argmax(dim=1).cpu().numpy()
            next_list = []

            for i in range(N):
                state_tensor = C_decrease[i] 
                action = intended_actions[i]

                for _ in range(SAMPLES): 
                    env.load_state_from_tensor(state_tensor)
                    obs, _, _, _, _ = env.step(action)
                    obs_img = obs['image']
                    next_list.append(obs_img)

            huge_batch_np = np.stack(next_list)
            huge_batch = torch.tensor(huge_batch_np, dtype=torch.float32, device=C_decrease.device)
            v_next_all = certificate(huge_batch).view(-1)
            v_next_reshaped = v_next_all.view(N, SAMPLES)
            expected_V_next = v_next_reshaped.mean(dim=1, keepdim=True)
#####

            # Apply the logic: If unsafe, V = Barrier. If safe, V = predicted.
            # Using torch.where(condition, if_true, if_false)
            # v_next_corrected = torch.where(
            #     is_next_unsafe.view(-1, 1), 
            #     torch.tensor(UNSAFE_BARRIER, device=v_next_subset.device), 
            #     v_next_subset
            # )

            # E. Check the Inequality: V(next) + EPSILON <= V(curr)
            decrease_gap = v_curr - (expected_V_next + EPSILON)
            valid_decrease_mask = (decrease_gap >= 0)
            failed_decrease_mask = ~valid_decrease_mask
            acc_decrease = valid_decrease_mask.float().mean().item() * 100

            print(f"[3] Decrease Condition (V_next + {EPSILON} <= V_curr):")
            print(f"    States Checked: {len(C_decrease)} (Filtered from {len(C_decrease)})")
            print(f"    Passed: {acc_decrease:.2f}%")

            # Optional: Debugging Failures
            if acc_decrease < 100:
                print(f"    Avg Violation Gap: {decrease_gap[~valid_decrease_mask].mean().item():.4f}")




    # --- Final Summary ---
    passed = (acc_init > 99.9) and (acc_unsafe > 99.9) and (acc_decrease > 99.9)
    print("="*30)
    print(f"VERIFICATION RESULT: {'PASSED ✅' if passed else 'FAILED ❌'}")
    print("="*30)

    return passed


# In[17]:


# --- CONFIGURATION ---
ENV_NAME = "MiniGrid-Dynamic-Obstacles-16x16-v0"
MODEL_PATH = "storage/randomenv_ranreward_16_v2/status.pt" 
CERTI_PATH = "certis/certificate_16x16_LV_1itrlearning.pt"

# Paper Parameters
P_SATISFACTION = 0.95
UNSAFE_BARRIER = 1.0 / (1.0 - P_SATISFACTION) # ~100
EPSILON = 0.01
MAX_ITERATIONS = 20  # Max number of CEGIS loops
EPOCHS_PER_ITER = 10000 # How long to train per loop

def main():
    # 1. Setup Universe (All possible states)
    env = make_env(ENV_NAME, seed=1)
    states, masks, agentPos, direc = generate_transitions(ENV_NAME,1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = states.to(device);
    C_init = states[masks["start"].to(device)]
    C_unsafe = states[masks["unsafe"].to(device)]
    C_decrease = states[masks["safe"].to(device) ]
    print(len(C_unsafe))
    for k in masks: masks[k] = masks[k].to(device)

    # 2. Initialize Networks
    dummy_obs_space = {"image": (16, 16, 3)}
    policy = ACModel(dummy_obs_space, env.action_space)
    checkpoint = torch.load(MODEL_PATH)
    policy.load_state_dict(checkpoint["model_state"])

    certificate = CertificateNet(dummy_obs_space)
    optimizer = optim.Adam(list(certificate.parameters()), lr=0.0003)
    # checkpoint = torch.load(CERTI_PATH)
    # certificate.load_state_dict(checkpoint["model_state"])
    # optimizer.load_state_dict(checkpoint["optimizer_state"])
    certificate.to(device); policy.to(device)




    # --- CEGIS LOOP ---
    # for i in range(MAX_ITERATIONS):
    #     print(f"\n ITERATION {i+1}")

    # Step A: Learner (Train on current set)
    policy, certificate, history = train_learner(env,
        policy, certificate, optimizer,  C_init, C_unsafe, C_decrease, 1
    )
    # plot_learning_curve(history)

    # Step B: Verifier (Check ALL states)
    passed = run_verifier(env, policy,certificate, C_init, C_unsafe, C_decrease )


    # Step D: Aggregation (Add Counterexamples to Training Set)
    # Paper: "extended by counterexamples computed by the verifier" 
    # initial_len = len(train_indices)
    # train_indices.update(counterexamples)
    # print(f"Added {len(train_indices) - initial_len} new unique states to training set.")

    # Save
    # torch.save(policy.state_dict(), "cegis_policy.pt")
    # torch.save(certificate.state_dict(), "cegis_certificate.pt")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:


#### SCRATCH PAD ################
# env = make_env(ENV_NAME,seed=2)
# states, masks,_,_ = generate_transitions(env)
# # print(f"Shape: {states.shape}")  # Expected: (N, 8, 8, 3)
# # print(f"Data Type: {states.dtype}")
# # print(f"Min Value: {states.min()}, Max Value: {states.max()}")

# # import matplotlib.pyplot as plt
# # # def show_env(env):
# # # env.reset()
# # # This returns a numpy array (pixels) instead of opening a window
# # img = env.get_obs_render(states[2].cpu().numpy())
# # plt.figure(figsize=(4, 4))
# # plt.imshow(img)
# # plt.axis('off')
# # plt.show()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# states = states.to(device);
# for k in masks: masks[k] = masks[k].to(device)

# # 2. Initialize Networks
# policy = ACModel(dummy_obs_space, env.action_space)
# checkpoint = torch.load(MODEL_PATH)
# if "model_state" in checkpoint:
#     policy.load_state_dict(checkpoint["model_state"])
# else:
#     policy.load_state_dict(checkpoint)
# CERTI_PATH = "cegis_certificate.pt"
# dummy_obs_space = {"image": (8, 8, 3)}
# certificate = CertificateNet(dummy_obs_space)
# optimizer = optim.Adam(list(certificate.parameters()), lr=0.0003)
# checkpoint = torch.load(CERTI_PATH)
# certificate.load_state_dict(checkpoint["model_state"])
# optimizer.load_state_dict(checkpoint["optimizer_state"])
# certificate.to(device)
# # policy, certificate = train_learner(env,
# #         policy, certificate, optimizer, 
# #         states, masks,1
# #     )


# with torch.no_grad():
#         all_values = certificate(states).view(-1).cpu().numpy()

# all_values = certificate(states)
# idx=0
# print(all_values[idx])
# env.load_state_from_tensor(states[idx])
# img = env.render()
# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# # height, width = 8, 8
# # grid_map = np.full((width, height), np.nan)

# # # 3. Map each value to its (x, y) coordinate
# # # We look at the 'states' tensor to find where the agent is.
# # states_np = states.cpu().numpy()

# # for i in range(len(states)):
# #     # Channel 0 contains Object IDs. Agent ID is 10.
# #     obj_layer = states_np[i, :, :, 0]

# #     # Find coordinates where value is 10
# #     # rows = x, cols = y in MiniGrid convention usually, but let's stick to array indices
# #     # np.where returns tuple of arrays (row_indices, col_indices)
# #     locs = np.where(obj_layer == 10)

# #     if len(locs[0]) > 0:
# #         x, y = locs[0][0], locs[1][0]

# #         # Handling Direction Conflicts:
# #         # If multiple states exist for the same (x,y) (different directions),
# #         # we take the MIN value (conservative estimate) or just overwrite.
# #         # Here we take min to see the safest approach.
# #         current_val = grid_map[x, y]
# #         new_val = all_values[i]

# #         if np.isnan(current_val):
# #             grid_map[x, y] = new_val
# #         else:
# #             # Optional: Average them or take Min
# #             grid_map[x, y] = min(current_val, new_val)

# # # 4. Print the Grid nicely
# # print(f"\n{'='*10} CERTIFICATE VALUE GRID (8x8) {'='*10}")
# # print("Cells with '....' means no state was found for that position.\n")

# # # Print Column Headers
# # print("      ", end="")
# # for y in range(height):
# #     print(f"   {y}    ", end="")
# # print("\n")

# # for x in range(width):
# #     print(f"{x}  |", end="") # Row Header
# #     for y in range(height):
# #         val = grid_map[x, y]
# #         if np.isnan(val):
# #             print("  ....  ", end=" ")
# #         else:
# #             # Color coding (Optional text formatting)
# #             # High = Unsafe, Low = Safe
# #             print(f" {val:6.2f} ", end=" ")
# #     print("|")
# #     print("") # Newline between rows

# # show_env(env)


# In[ ]:





# In[ ]:





# In[ ]:




