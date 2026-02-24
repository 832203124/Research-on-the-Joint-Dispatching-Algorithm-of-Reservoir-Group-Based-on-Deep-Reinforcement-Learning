import numpy as np

from src.init import (
    PARAMS, N_RES, T, ABBR2IDX, RESERVOIRS,
)


class MultiReservoirEnv:
    def __init__(self, inflow_data, demand_data):
        self.T = T
        self.N = N_RES
        self.inflow_data = inflow_data
        self.demand_data = demand_data
        self.reset()

    def reset(self):
        self.t = 0
        self.storage = (PARAMS["capacity"] + PARAMS["dead_storage"]) / 2
        self.overflow = np.zeros(self.N)
        self.release = np.zeros(self.N)
        self.prev_release = np.zeros(self.N)
        return self.get_obs()

    def get_obs(self):
        obs = []
        for i in range(self.N):
            # Normalized storage
            s_norm = (self.storage[i] - PARAMS["dead_storage"][i]) / (
                PARAMS["capacity"][i] - PARAMS["dead_storage"][i]
            )
            # Normalized inflow and demand
            i_norm = self.inflow_data[self.t, i] / (self.inflow_data[:, i].max() + 1e-6)
            d_norm = self.demand_data[self.t, i] / (self.demand_data[:, i].max() + 1e-6)
            # Upstream inflow from releases
            ups = RESERVOIRS[i + 1]["upstream"]
            up_flow = self.release[[ABBR2IDX[n] for n in ups]].sum() if ups and self.t > 0 else 0.0
            obs.append([
                s_norm,
                i_norm,
                up_flow / (PARAMS["max_release"][i] + 1e-6),
                d_norm
            ])
        return obs

    def step(self, action_norm):
        action_norm = np.array(action_norm).flatten()

        # Map normalized actions to physical releases
        release = np.clip(
            action_norm * (PARAMS["max_release"] - PARAMS["min_release"]) + PARAMS["min_release"],
            PARAMS["min_release"],
            PARAMS["max_release"]
        )

        # Add upstream releases to local inflow
        inflow = self.inflow_data[self.t].copy()
        for i in range(self.N):
            ups = RESERVOIRS[i + 1]["upstream"]
            if ups:
                inflow[i] += release[[ABBR2IDX[n] for n in ups]].sum()

        # Update storage
        new_sto = self.storage + inflow - release
        dead = PARAMS["dead_storage"]
        cap = PARAMS["capacity"]
        usable = cap - dead

        # Target storage (flood season assumed)
        flood = True
        target = dead + (0.5 if flood else 0.6) * usable

        # Compute reward
        reward = 0.0
        in_range = True

        for i in range(self.N):
            # Penalty for deviating from target storage
            dev = abs(new_sto[i] - target[i]) / (target[i] + 1e-6)
            if dev > 0.35:
                reward -= 5 * dev
                in_range = False

            # Penalty for large release changes
            rel_change = abs(release[i] - self.prev_release[i]) / (PARAMS["max_release"][i] + 1e-6)
            if self.t > 0 and rel_change > 0.25:
                reward -= 5 * rel_change
                in_range = False

        # Bonus for meeting demand (only if within safe operating range)
        if in_range:
            for i in range(self.N):
                reward += 10 * min(release[i] / (self.demand_data[self.t, i] + 1e-6), 1.0)

        # Enforce physical bounds
        self.storage = np.clip(new_sto, dead, cap)
        self.prev_release = release.copy()
        self.release = release
        self.t += 1

        done = self.t >= self.T - 1
        norm_sto = (self.storage - dead) / (usable + 1e-6)

        return self.get_obs(), reward, done, {
            "total_reward": float(reward),
            "normalized_storage": norm_sto.copy(),
            "release": release.copy(),
            "storage": self.storage.copy(),
        }