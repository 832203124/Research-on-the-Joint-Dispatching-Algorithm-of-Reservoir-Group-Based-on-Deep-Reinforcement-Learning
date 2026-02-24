import numpy as np
import pandas as pd
from src.env import MultiReservoirEnv
from src.init import load_reservoir_arrays, N_RES, T, PARAMS
from src.maddpg import MADDPG


def test_maddpg(load_path="models/latest_model.pth", save_csv=True, filename="result/test_results_AttentionMADDPG.csv"):
    np.random.seed(40)
    inflow_data, demand_data = load_reservoir_arrays()
    env = MultiReservoirEnv(inflow_data, demand_data)
    n_agents = N_RES

    agent = MADDPG(4, 1, n_agents)
    agent.load(load_path)

    states = env.reset()
    infos = [{
        "timestep": 0,
        "total_reward": 0,
        "states": env.get_obs(),
        "actions": [0] * n_agents,
        "done": False,
        "normalized_storage": [0.5] * n_agents,
        "storage": 0.5 * (PARAMS["capacity"] + PARAMS["dead_storage"]),
        "release": [0] * n_agents,
        "inflow": inflow_data[0],
        "demand": demand_data[0],
    }]

    for step in range(T):
        actions = agent.act(states, 0)
        next_states, reward, done, info = env.step(actions)

        infos.append({
            "timestep": step + 1,
            "total_reward": reward,
            "states": next_states.copy(),
            "actions": actions.copy(),
            "done": done,
            **info
        })

        states = next_states
        if done:
            print(f"Episode completed at step {step}")
            break

    records = []
    for info in infos:
        t = info["timestep"]
        rec = {
            "timestep": t,
            "total_reward": round(info["total_reward"], 4),
        }

        # Add per-agent metrics
        for j in range(n_agents):
            rec[f"normalized_storage_{j + 1}"] = round(info["normalized_storage"][j], 4)
            rec[f"storage_{j + 1}"] = round(info["storage"][j], 4)
            rec[f"inflow_{j + 1}"] = round(inflow_data[t][j], 4)
            rec[f"demand_{j + 1}"] = round(demand_data[t][j], 4)
            rec[f"release_reservoir_{j + 1}"] = round(info["release"][j], 4)

            norm_rel = (info["release"][j] - PARAMS["min_release"][j]) / \
                       (PARAMS["max_release"][j] - PARAMS["min_release"][j])
            rec[f"normalized_release_{j + 1}"] = np.round(norm_rel, 4)

        # Add raw states and actions
        rec.update({f"state_{j}": val for j, val in enumerate(info["states"])})
        rec.update({f"action_{j}": val for j, val in enumerate(info["actions"])})

        records.append(rec)

    df = pd.DataFrame(records)
    if save_csv:
        df.to_csv(filename, index=False)

    print("\nTest Summary:")
    print("=" * 50)
    print(f"Steps: {len(df)}")
    print(f"Total reward: {df['total_reward'].sum():.4f}")
    print(f"Mean step reward: {df['total_reward'].mean():.4f}")
    print("=" * 50)

    print("\nRelease stats:")
    for j in range(n_agents):
        col = f"release_reservoir_{j + 1}"
        print(f"  Reservoir{j + 1}: mean={df[col].mean():.3f}, "
              f"std={df[col].std():.3f}, "
              f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")

    return df, infos


if __name__ == "__main__":
    print("\nTesting trained model...")
    test_maddpg()