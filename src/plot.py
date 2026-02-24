import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.init import PARAMS

reservoirs = ['AS', 'CT', 'GT', 'SD', 'SK', 'SXK']
output_dir = 'picture'
os.makedirs(output_dir, exist_ok=True)

model_configs = {
    'AttentionMADDPG': {
        'label': 'Attention-MADDPG',
        'color': 'C0',
        'linestyle': '-'
    },
    'MADDPG': {
        'label': 'MADDPG',
        'color': 'C1',
        'linestyle': '--'
    },
    'DDPG': {
        'label': 'DDPG',
        'color': 'C2',
        'linestyle': ':'
    }
}

result_files = {
    'AttentionMADDPG': 'result/test_results_AttentionMADDPG.csv',
    'MADDPG': 'result/test_results_MADDPG.csv',
    'DDPG': 'result/test_results_DDPG.csv'
}


def plot_storage_comparison():
    capacity = PARAMS["capacity"]
    dead = PARAMS["dead_storage"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.axhline(dead[i], color='red', linestyle='--', linewidth=0.8, label='Dead Level' if i == 0 else "")

        effective_capacity = capacity[i] * 0.95
        ax.axhline(effective_capacity, color='green', linestyle='--', linewidth=0.8,
                   label='Effective Storage' if i == 0 else "")

        ax.axhline(capacity[i], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        for model_name, config in model_configs.items():
            if not os.path.exists(result_files[model_name]):
                print(f"Warning: {result_files[model_name]} not found. Skipping.")
                continue

            df = pd.read_csv(result_files[model_name])
            s = df[f'storage_{i + 1}']
            ax.plot(
                df['timestep'],
                s,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=1.5,
                label=config['label'] if i == 0 else ""
            )

        ax.set_title(f'{reservoirs[i]} Storage')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Storage (10⁸ m³)')
        ax.grid(True, linestyle=':', alpha=0.7)

        if i == 0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, 'storage_comparison.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_normalized_release_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in enumerate(axes.flatten()):
        for model_name, config in model_configs.items():
            if not os.path.exists(result_files[model_name]):
                continue

            df = pd.read_csv(result_files[model_name])
            ax.plot(
                df['timestep'],
                df[f'normalized_release_{i + 1}'],
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=1.5,
                label=config['label'] if i == 0 else ""
            )

        ax.set_title(f'{reservoirs[i]} Normalized Release')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Normalized Release')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.7)

        if i == 0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, 'release_comparison.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_performance_comparison():
    metrics = {
        'Storage Stability (↓)': [],
        'Release Smoothness (↓)': [],
        'Supply Satisfaction Rate (↑)': []
    }
    labels = []

    for model_name, config in model_configs.items():
        path = result_files[model_name]
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue

        df = pd.read_csv(path)
        T = len(df)

        storage_std = np.mean([df[f'storage_{i}'].std() for i in range(1, 7)])
        metrics['Storage Stability (↓)'].append(storage_std)

        smoothness = np.mean([
            np.abs(np.diff(df[f'release_reservoir_{i}'].values)).mean()
            for i in range(1, 7)
        ])
        metrics['Release Smoothness (↓)'].append(smoothness)

        satisfaction = np.mean([
            (df[f'release_reservoir_{i}'] >= df[f'demand_{i}']).sum() / T
            for i in range(1, 7)
        ])
        metrics['Supply Satisfaction Rate (↑)'].append(satisfaction)

        labels.append(config['label'])

    perf_df = pd.DataFrame(metrics, index=labels)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(
        x - width,
        perf_df.iloc[:, 0],
        width,
        label=perf_df.columns[0],
        color='skyblue'
    )
    b2 = ax.bar(
        x,
        perf_df.iloc[:, 1],
        width,
        label=perf_df.columns[1],
        color='lightgreen'
    )
    b3 = ax.bar(
        x + width,
        perf_df.iloc[:, 2],
        width,
        label=perf_df.columns[2],
        color='salmon'
    )

    for bars, vals in zip([b1, b2, b3], [perf_df.iloc[:, i] for i in range(3)]):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'performance_comparison.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_reward_distribution(csv_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['total_reward'], kde=True)
    plt.title('Reward Histogram')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['total_reward'])
    plt.title('Reward Boxplot')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'reward_distribution.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_inflow_demand_release(csv_path):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in enumerate(axes.flatten(), start=1):
        ax.plot(df['timestep'], df[f'inflow_{i}'], label='Inflow', alpha=0.8)
        ax.plot(df['timestep'], df[f'demand_{i}'], label='Demand', alpha=0.8)
        ax.plot(df['timestep'], df[f'release_reservoir_{i}'], label='Release', alpha=0.8)

        ax.set_title(f'{reservoirs[i - 1]}: Inflow/Demand/Release')
        ax.set_xlabel('Timestep')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'inflow_demand_release.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_storage_proportion(csv_path):
    df = pd.read_csv(csv_path)

    storage_cols = [f'storage_{i}' for i in range(1, 7)]
    total_storage = df[storage_cols].sum(axis=1)
    props = df[storage_cols].divide(total_storage, axis=0)

    plt.figure(figsize=(12, 6))
    plt.stackplot(df['timestep'], [props[col] for col in storage_cols], labels=reservoirs)

    plt.xlabel('Timestep')
    plt.ylabel('Proportion')
    plt.legend(loc='upper right')

    save_path = os.path.join(output_dir, 'storage_proportion.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_combined_correlation_heatmaps(csv_path='result/test_results.csv'):
    df = pd.read_csv(csv_path)

    release_cols = [f'release_reservoir_{i + 1}' for i in range(6)]
    storage_cols = [f'storage_{i + 1}' for i in range(6)]

    release_df = df[release_cols]
    storage_df = df[storage_cols]

    release_df.columns = reservoirs
    storage_df.columns = reservoirs

    corr_release = release_df.corr()
    corr_storage = storage_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        corr_release,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        ax=axes[0],
        cbar_kws={'shrink': 0.8}
    )
    axes[0].set_title('Release Correlation')

    sns.heatmap(
        corr_storage,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        ax=axes[1],
        cbar_kws={'shrink': 0.8}
    )
    axes[1].set_title('Storage Correlation')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_correlation_heatmaps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    csv = 'result/test_results_AttentionMADDPG.csv'
    plot_storage_comparison()
    plot_normalized_release_comparison()
    plot_performance_comparison()
    plot_reward_distribution(csv)
    plot_inflow_demand_release(csv)
    plot_storage_proportion(csv)
    plot_combined_correlation_heatmaps(csv)


if __name__ == '__main__':
    main()
