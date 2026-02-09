import gymnasium as gym
import random
import numpy as np

class StochasticTransitionWrapper(gym.Wrapper):
    def __init__(self, env, prob=0.03):
        """
        Args:
            env: The environment to wrap.
            prob: Probability (0.0 to 1.0) that the state transition will be randomized.
                  Default 0.03 (3%).
        """
        super().__init__(env)
        self.stochastic_prob = prob

    def step(self, action):
        """
        Executes the step with a chance of random state transition (slip).
        """
        # 1. Check for random slip (2% chance)
        if random.random() < self.stochastic_prob:
            # --- SLIP LOGIC ---
            # Instead of executing the action, we move the agent to a random adjacent cell.
            
            # Access the base environment to manipulate position directly
            base_env = self.unwrapped
            base_env.step_count += 1
            truncated = base_env.step_count >= base_env.max_steps
            x, y = base_env.agent_pos
            terminated = False
            width, height = base_env.width, base_env.height
            
            # Define only 4 possible adjacent moves (Up, Down, Left, Right) - NO DIAGONALS
            potential_moves = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1)
            ]
            
            # Filter for valid cells (within bounds and not walls)
            valid_neighbors = []
            for nx, ny in potential_moves:
                # Check grid bounds
                if 0 <= nx < width and 0 <= ny < height:
                    # Check for walls
                    cell = base_env.grid.get(nx, ny)
                    # We move if the cell is empty (None) or not a wall.
                    # Note: We treat balls/lava as passable for the slip movement itself.
                    if cell is None or (hasattr(cell, 'type') and cell.type != 'wall'):
                        valid_neighbors.append((nx, ny))
            
            # If valid neighbors exist, pick one randomly and move there
            if valid_neighbors:
                new_pos = valid_neighbors[np.random.choice(len(valid_neighbors))]
                base_env.agent_pos = new_pos
                cell_at_new_pos = base_env.grid.get(*new_pos)
                if cell_at_new_pos is not None and cell_at_new_pos.type == 'ball':
                    terminated = True   # <--- Terminate if we slipped onto a ball
                else:
                    terminated = False
            
            # Regenerate observation after forced movement
            # We return the raw observation here. The outer CollisionFullyObsWrapper
            # will catch this and convert it to your tensor.
            obs = base_env.gen_obs()
            
            # We return 0 reward here because the outer ShapedRewardWrapper will 
            # see the position change and calculate the correct distance reward/penalty.
            reward = 0.0 
            info = {"slip": True} 
            
            return obs, reward, terminated, truncated, info

        else:
            # 2. Normal Execution (98% chance)
            # Execute the intended best action
            base_env = self.unwrapped
            base_env.step_count += 1
            truncated = base_env.step_count >= base_env.max_steps
            terminated = False
            x, y = base_env.agent_pos
            direction = base_env.agent_dir
            width, height = base_env.width, base_env.height
            
            if action == base_env.actions.left:
                base_env.agent_dir = (direction - 1) % 4

            elif action == base_env.actions.right:
                base_env.agent_dir = (direction + 1) % 4

            elif action == base_env.actions.forward:
                # Calculate target position based on current direction
                dx, dy = base_env.dir_vec
                nx, ny = x + dx, y + dy
            

                if 0 <= nx < width and 0 <= ny < height:
                        cell = base_env.grid.get(nx, ny)

                        # Move if empty or not a wall 
                        # (Critically: We ALLOW moving onto balls/lava/goals)
                        if cell is None or (hasattr(cell, 'type') and cell.type != 'wall'):
                            base_env.agent_pos = (nx, ny)

                            # Check if we walked into something that kills or wins
                            if cell is not None:
                                if cell.type == 'ball':
                                    terminated = True
                                elif cell.type == 'lava':
                                    terminated = True
                                elif cell.type == 'goal':
                                    terminated = True
                                    # Note: 'ShapedRewardWrapper' handles the actual +10 reward
                                    # based on the position change, so we keep reward=0.0 here.

            # Regenerate observation after manual update
            obs = base_env.gen_obs()

            reward = 0.0 
            info = {"slip": True} 
            return obs, reward, terminated, truncated, info
        
            # new_pos = np.random.choice(len(valid_neighbors))
            # base_env.agent_pos = new_pos
            # cell_at_new_pos = base_env.grid.get(*new_pos)
            # if cell_at_new_pos is not None and cell_at_new_pos.type == 'ball':
            #     terminated = True   # <--- Terminate if we slipped onto a ball
            # else:
            #     terminated = False
            # return self.env.step(action)