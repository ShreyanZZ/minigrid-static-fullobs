import torch
import numpy as np
from minigrid.core.world_object import Ball, Goal, Wall

def generate_transitions(env):
    """
    Generates all valid transitions using FULL OBSERVABILITY (8x8).
    Resulting states will be (N, 8, 8, 3).
    """
    print("Generating State Space (Full 8x8 Grid)...")
    
    states = []
    next_states_map = [] 
    
    # Masks lists
    is_goal = []
    is_unsafe = []
    is_start = []
    
    # Access the base environment to manipulate the grid
    base_env = env.unwrapped
    width = base_env.grid.width
    height = base_env.grid.height
    
    # Iterate over every cell and direction
    for i in range(width):
        for j in range(height):
            cell = base_env.grid.get(i, j)
            
            # Skip Walls
            if cell is not None and isinstance(cell, Wall):
                continue

            for direction in range(4):
                # 1. Force Agent State
                base_env.agent_pos = (i, j)
                base_env.agent_dir = direction
                
                # --- THE FIX: Get Full 8x8 Observation ---
                # grid.encode() returns the full (W, H, 3) array.
                # It includes walls, objects, and the agent's position/dir.
                # This matches what FullyObsWrapper provides.
                full_grid_image = base_env.grid.encode()
                
                # Ensure it matches the range [0, 255] or [0, 10] depending on normalization
                # MiniGrid usually returns integers. We stick to that.
                states.append(full_grid_image)
                
                # Identify State Type
                if cell is not None and isinstance(cell, Goal):
                    is_goal.append(True)
                    is_unsafe.append(False)
                    is_start.append(False)
                elif cell is not None and isinstance(cell, Ball):
                    is_goal.append(False)
                    is_unsafe.append(True)
                    is_start.append(False)
                elif i == 1 and j == 1: # Standard Start
                    is_goal.append(False)
                    is_unsafe.append(False)
                    is_start.append(True)
                else:
                    is_goal.append(False)
                    is_unsafe.append(False)
                    is_start.append(False)
                
                # --- 2. Simulate Next States (Actions 0, 1, 2) ---
                current_transitions = []
                for action in [0, 1, 2]: # Left, Right, Forward
                    # Reset state for simulation
                    base_env.agent_pos = (i, j)
                    base_env.agent_dir = direction
                    
                    # Step the environment
                    # We ignore the returned 'obs' because it might be wrapped/partial.
                    # Instead, we look at the grid again after the step.
                    _ = env.step(action)
                    
                    # Get the Full 8x8 Grid for the NEXT state
                    next_full_image = base_env.grid.encode()
                    current_transitions.append(next_full_image)
                
                next_states_map.append(current_transitions)

    # Convert to Tensors
    # states shape: (N, 8, 8, 3)
    states_tensor = torch.tensor(np.array(states), dtype=torch.float)
    # next_states shape: (N, 3, 8, 8, 3)
    next_states_tensor = torch.tensor(np.array(next_states_map), dtype=torch.float)
    
    masks = {
        "goal": torch.tensor(is_goal, dtype=torch.bool),
        "unsafe": torch.tensor(is_unsafe, dtype=torch.bool),
        "start": torch.tensor(is_start, dtype=torch.bool),
        "safe": ~torch.tensor(is_unsafe, dtype=torch.bool) & ~torch.tensor(is_goal, dtype=torch.bool)
    }
    
    print(f"Generated {len(states)} fully observable states.")
    print(f"Shape check: {states_tensor.shape} (Should be N, 8, 8, 3)")
    
    return states_tensor, next_states_tensor, masks