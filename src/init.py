import os
import numpy as np
import pandas as pd

RESERVOIRS = {
    1: {"abbr": "AS", "upstream": []},
    2: {"abbr": "CT", "upstream": []},
    3: {"abbr": "GT", "upstream": []},
    4: {"abbr": "SD", "upstream": []},
    5: {"abbr": "SK", "upstream": ["GT", "SD", "SXK"]},
    6: {"abbr": "SXK", "upstream": ["AS", "CT"]},
}

T = 90
FLOOD_START = 90
FLOOD_END = 270
N_RES = len(RESERVOIRS)
M3S_2_MCMD = 0.0864
ABBR2IDX = {v["abbr"]: k - 1 for k, v in RESERVOIRS.items()}
MEAN_REL = np.array([13.8, 24.6, 2.18, 12.4, 149.3, 53.4])

PARAMS = {
    "capacity": np.array([740, 870, 642, 108, 2600, 374]),
    "dead_storage": np.array([185.0, 217.5, 160.5, 27.0, 650.0, 93.5]),
    "min_release": np.array([2.8, 4.9, 0.4, 2.5, 29.9, 10.7]),
    "max_release": np.array([82.8, 147.6, 13.1, 74.4, 895.8, 320.4]),
    "efficiency": np.array([0.88, 0.88, 0.88, 0.88, 0.88, 0.88]),
    "rated_head": np.array([58, 53, 97, 20, 30, 18]),
    "flood_limit": np.array([629, 739, 545, 91, 2210, 317]),
}


def estimate_params():
    cap = np.array([740, 870, 642, 108, 2600, 374])
    dead = np.round(0.25 * cap, 1)
    min_r = np.round(0.2 * MEAN_REL, 1)
    max_r = np.round(6 * MEAN_REL, 1)
    eff = np.full(N_RES, 0.88)
    head = np.array([58, 53, 97, 20, 30, 18])
    flood = (0.85 * cap).astype(int)
    params = {
        "capacity": cap,
        "dead_storage": dead,
        "min_release": min_r,
        "max_release": max_r,
        "efficiency": eff,
        "rated_head": head,
        "flood_limit": flood,
    }
    print("PARAMS = {")
    for k, v in params.items():
        print(f'    "{k}": np.array({v.tolist()}),')
    print("}")
    return params


def simulate(days=90, seed=42, season="wet"):
    rng = np.random.default_rng(seed)
    os.makedirs("data", exist_ok=True)

    # Net inflow after upstream subtraction
    net = MEAN_REL.copy()
    for idx, info in RESERVOIRS.items():
        for up in info["upstream"]:
            net[idx - 1] -= MEAN_REL[ABBR2IDX[up]]
    net = np.maximum(net, 0)

    Q = np.zeros((days, N_RES))
    D = np.zeros_like(Q)

    inflow_factors = {
        1: 0.28, 2: 0.35, 3: 0.45, 4: 0.65,
        5: 1.50, 6: 1.76, 7: 1.45, 8: 1.20,
        9: 0.80, 10: 0.50, 11: 0.40, 12: 0.30
    }

    annual_factor = rng.uniform(0.9, 1.1)

    if season == "wet":
        start_date_str = "2024-05-01"
    elif season == "dry":
        start_date_str = "2024-01-01"
    else:
        raise ValueError("season must be 'wet' or 'dry'")

    burn_in = 30
    total_days = days + burn_in
    dates_full = pd.date_range(start_date_str, periods=total_days, freq="D")
    months_full = dates_full.month.values

    rho = 0.85   # Temporal persistence
    alpha = 0.2  # Smoothing factor

    first_month = months_full[0]
    prev_Q = np.maximum(net * inflow_factors[first_month] * annual_factor, 1e-3)

    for d in range(total_days):
        month = months_full[d]
        base_factor = inflow_factors[month] * annual_factor

        is_flood = 4 <= month <= 9
        is_dry = month in [1, 2, 12]

        if is_flood:
            season_scale = rng.normal(1.2, 0.12)
        elif is_dry:
            season_scale = rng.normal(0.95, 0.05)
        else:
            season_scale = rng.normal(1.0, 0.07)

        sigma = 0.35 if is_flood else 0.15

        for i in range(N_RES):
            target_flow = net[i] * base_factor * season_scale
            noise = rng.normal(0, sigma)
            log_q = rho * np.log(prev_Q[i] + 1e-6) + (1 - rho) * np.log(target_flow + 1e-6) + noise
            q_raw = np.exp(log_q)
            q_smooth = alpha * q_raw + (1 - alpha) * prev_Q[i]
            prev_Q[i] = q_smooth

            if d >= burn_in:
                Q[d - burn_in, i] = q_smooth
                D[d - burn_in, i] = 0.7 * q_smooth * rng.normal(1.0, 0.1)

    Q = np.maximum(Q, 0)
    D = np.maximum(D, 0)

    cols = [RESERVOIRS[i + 1]["abbr"] for i in range(N_RES)]
    dates = pd.date_range(start_date_str, periods=days, freq="D")

    Qdf = pd.DataFrame(np.round(Q, 2), columns=cols)
    Qdf.insert(0, "date", dates)
    Ddf = pd.DataFrame(np.round(D, 2), columns=cols)
    Ddf.insert(0, "date", dates)

    Qdf.to_csv("data/reservoir_flows.csv", index=False)
    Ddf.to_csv("data/reservoir_demand.csv", index=False)

    return Qdf, Ddf, Q, D


def load_reservoir_arrays(flow="data/reservoir_flows.csv", dem="data/reservoir_demand.csv"):
    Qdf = pd.read_csv(flow)
    Ddf = pd.read_csv(dem)
    cols = [c for c in Qdf.columns if c != "date"]
    return Qdf[cols].values, Ddf[cols].values


if __name__ == "__main__":
    Qdf_wet, _, _, _ = simulate(season="wet")
    estimate_params()