import numpy as np
import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Wall, Goal, Ball, Lava
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

# Constants
COLLISION_ID = 16
BALL_ID = OBJECT_TO_IDX['ball']
AGENT_ID = OBJECT_TO_IDX['agent']

class CollisionFullyObsWrapper(gym.ObservationWrapper):
    """
    1. Forces Fully Observable View (Global Map).
    2. Injects ID 16 if Agent overlaps a Ball.
    3. Provides helper to generate hypothetical states.
    """
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to handle global grid
        # We assume square grid for simplicity, but works for rects too
        size = self.env.unwrapped.width
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0, high=255, shape=(size, size, 3), dtype='uint8'
            )
        })

    def observation(self, obs):
        """
        Called automatically after env.step() or env.reset()
        Returns: The Global Tensor with ID 16 logic applied.
        """
        env = self.env.unwrapped
        
        # 1. Get current physical details
        x, y = env.agent_pos
        direction = env.agent_dir
        
        grid_tensor=self._generate_custom_tensor(env, x, y, direction)
    
        new_obs = {
            'image': grid_tensor,
            'mission': obs.get('mission', 'survive') # Use existing or default
        }
        return new_obs

    def get_temp_state(self, x, y, direction):
        """
        YOUR REQUEST: Pass a hypothetical state (x,y,dir)
        and get the tensor showing if it's unsafe (16).
        Does NOT move the actual agent.
        """
        return self._generate_custom_tensor(self.env.unwrapped, x, y, direction)

    def _generate_custom_tensor(self, env, agent_x, agent_y, agent_dir):
        """
        Internal logic to build the grid tensor.
        """
        # 1. Get Static Background (Walls, Balls, Goals)
        # grid.encode() does NOT contain the agent.
        grid_tensor = env.grid.encode() 

        # 2. Check what is at the target coordinates
        # We handle out-of-bounds generally, but assuming x,y are valid here:
        target_cell = env.grid.get(agent_x, agent_y)
        
        # 3. Apply Logic
        if target_cell is not None and target_cell.type == 'ball':
            # --- COLLISION CASE (ID 16) ---
            grid_tensor[agent_x][agent_y] = np.array([
                COLLISION_ID, 
                COLOR_TO_IDX[target_cell.color], 
                agent_dir
            ])
        else:
            # --- NORMAL AGENT CASE (ID 10) ---
            grid_tensor[agent_x][agent_y] = np.array([
                AGENT_ID, 
                COLOR_TO_IDX['red'], 
                agent_dir
            ])
            
        return grid_tensor
    
    # ... inside class CollisionFullyObsWrapper ...

    def load_state_from_tensor(self, tensor_data):
        """
        Reconstructs the ENTIRE grid (Walls, Objects, Agent) from the tensor.
        """
        # 1. Prepare Data
        if hasattr(tensor_data, 'cpu'):
            grid_data = tensor_data.cpu().numpy()
        else:
            grid_data = tensor_data
            
        width, height = grid_data.shape[:2]
        
        # 2. Reset the internal Grid to Empty
        # We will refill it based on what we see in the tensor
        from minigrid.core.grid import Grid
        self.env.unwrapped.grid = Grid(width, height)
        
        # 3. Iterate over the Tensor to rebuild the world
        for i in range(width):
            for j in range(height):
                obj_id = int(grid_data[i, j, 0])
                color_idx = int(grid_data[i, j, 1])
                state = int(grid_data[i, j, 2])
                
                # Skip empty cells (ID 0 usually 'unseen' or 'empty')
                if obj_id == 0 or obj_id == 1: 
                    continue
                
                # --- A. Handle Agent ---
                if obj_id == 10: # Agent ID
                    self.env.unwrapped.agent_pos = (i, j)
                    self.env.unwrapped.agent_dir = state
                    continue
                
                # --- B. Handle Collision (Agent + Object) ---
                if obj_id == 16: # Your Custom COLLISION_ID
                    # 1. Place the Agent
                    self.env.unwrapped.agent_pos = (i, j)
                    self.env.unwrapped.agent_dir = state
                    
                    # 2. Restore the object underneath 
                    # (We assume it's a Ball based on your logic, or check color)
                    obj = Ball()
                    obj.color = IDX_TO_COLOR[color_idx]
                    self.env.unwrapped.grid.set(i, j, obj)
                    continue

                # --- C. Handle Static Objects (Walls, Goals, etc) ---
                # Use the ID to figure out what object this is
                try:
                    obj_type = IDX_TO_OBJECT[obj_id]
                    
                    if obj_type == 'wall':
                        self.env.unwrapped.grid.set(i, j, Wall())
                    elif obj_type == 'goal':
                        self.env.unwrapped.grid.set(i, j, Goal())
                    elif obj_type == 'ball':
                        ball = Ball()
                        ball.color = IDX_TO_COLOR[color_idx]
                        self.env.unwrapped.grid.set(i, j, ball)
                    elif obj_type == 'lava':
                        self.env.unwrapped.grid.set(i, j, Lava())
                        
                except KeyError:
                    pass # Unknown ID, skip