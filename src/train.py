import numpy as np
import matplotlib.pyplot as plt
import torch

from src.env import MultiReservoirEnv
from src.init import load_reservoir_arrays, N_RES, T
from src.maddpg import MADDPG


def train_maddpg(save_path="latest_model.pth"):
    # Set random seeds for reproducibility
    np.random.seed(40)
    torch.manual_seed(40)

    # Load data and initialize environment
    inflow_data, demand_data = load_reservoir_arrays()
    env = MultiReservoirEnv(inflow_data, demand_data)
    agent = MADDPG(4, 1, N_RES)
    # agent.load("models/latest_model.pth")  # Uncomment to load a pre-trained model

    # Training hyperparameters
    noise = 0.15
    noise_decay = 0.9
    min_noise = 0.0
    episodes = 50
    print_interval = 5

    # Track episode rewards
    scores = []

    print(f"Training MADDPG for {episodes} episodes")
    print("-" * 50)

    for ep in range(episodes):
        # Reset environment at the start of each episode
        states = env.reset()
        episode_reward = 0.0
        steps = 0

        # Run one episode
        for step in range(T):
            actions = agent.act(states, noise)
            next_states, reward, done, _ = env.step(actions)
            agent.remember(states, actions, reward, next_states, done)
            agent.learn()

            states = next_states
            episode_reward += reward
            steps += 1

            if done:
                break

        # Decay exploration noise
        noise = max(min_noise, noise * noise_decay)
        scores.append(episode_reward)

        # Log progress periodically
        if ep % print_interval == 0:
            print(f"Ep {ep:3d} | Reward: {episode_reward:7.2f} | Noise: {noise:.4f}")

    # Save the trained model
    agent.save(save_path)

    # Print training summary
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Avg score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print("=" * 50)

    return scores


if __name__ == "__main__":
    print("Training MADDPG...")
    scores = train_maddpg()

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title("MADDPG Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.grid(True)
    plt.show()