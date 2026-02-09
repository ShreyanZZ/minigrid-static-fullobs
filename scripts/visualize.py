import argparse
import numpy

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

success_count = 0
collision_count = 0
timeout_count = 0
total_episodes = 0
print(f"Starting evaluation for {args.episodes} episodes...")

for episode in range(args.episodes):
    obs, _ = env.reset()
    total_episodes += 1

    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done:
            if reward > 0:
                success_count += 1
                outcome = "SUCCESS"
            elif truncated:
                timeout_count += 1
                outcome = "TIMEOUT"
            else:
                collision_count += 1
                outcome = "CRASH"
            print(f"Ep {episode+1}: {outcome} | Reward: {reward:.2f}")
            break

print("\n" + "="*30)
print("       EVALUATION RESULTS       ")
print("="*30)
print(f"Total Episodes: {total_episodes}")
print(f"Successes:      {success_count}")
print(f"Crashes:        {collision_count}")
print(f"Timeouts:       {timeout_count}")
print("-" * 30)
if total_episodes > 0:
    print(f"Success Probability: {success_count / total_episodes:.2%}")
print("="*30)

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
