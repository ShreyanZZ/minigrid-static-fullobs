import gymnasium as gym
import numpy as np

class ShapedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_dist = None
        self.step_penalty = 0.01
        self.shaping_scale = 0.012
        self.goal_reward = 10.0
        self.fail_penalty = -15.0

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        w = self.env.unwrapped.width
        h = self.env.unwrapped.height
        self.goal_pos = (w - 2, h - 2)
        agent_pos = self.env.unwrapped.agent_pos
        self.prev_dist = self._manhattan_dist(agent_pos, self.goal_pos)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_pos = self.env.unwrapped.agent_pos
        
        
        # 1. Calculate Shaping
        curr_dist = self._manhattan_dist(agent_pos, self.goal_pos)
        dist_reward = (self.prev_dist - curr_dist) * self.shaping_scale
        
        # 2. Base Step Reward
        custom_reward = dist_reward - self.step_penalty

        # 3. Terminal States
        status = "Step"
        if terminated:
            if np.array_equal(agent_pos, self.goal_pos):
                custom_reward = self.goal_reward
                status = "WIN"
            else:
                custom_reward = self.fail_penalty
                status = "FAIL"
        
        # --- DEBUG PRINT ---
        # Only print for the first 50 steps so your console doesn't freeze
        # if self.env.unwrapped.step_count < 3 and curr_dist != self.prev_dist:
        #     print(f"[{status}] Dist: {curr_dist} | pre dist: {self.prev_dist:.2f} | New Reward: {custom_reward:.4f}")
        # -------------------

        self.prev_dist = curr_dist
        return obs, custom_reward, terminated, truncated, info