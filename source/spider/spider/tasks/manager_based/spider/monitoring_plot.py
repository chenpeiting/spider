"""
Filename: monitoring_plot.py
Location: Same directory as spider_env_cfg.py
"""

import matplotlib
matplotlib.use('Agg')  # 非互動式，不需要螢幕
import matplotlib.pyplot as plt
import numpy as np


def plot_rl_cpg_analysis(log_buffer, rl_weight=0.1, save_path='rl_cpg_analysis.png'):
    """
    Plot RL-CPG analysis charts
    
    Args:
        log_buffer: Monitoring data dictionary
        rl_weight: RL weight coefficient
        save_path: Save path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    steps = np.arange(len(log_buffer['prior_magnitude']))

    # (1) Magnitude Comparison
    ax = axes[0, 0]
    ax.plot(steps, log_buffer['prior_magnitude'], label='CPG', linewidth=2, color='blue')
    ax.plot(steps, log_buffer['rl_magnitude'], label='RL', linewidth=2, color='red')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Magnitude (rad)')
    ax.set_title('Action Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) Direction Alignment
    ax = axes[0, 1]
    ax.plot(steps, log_buffer['correlation'], color='green', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold 0.5')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('RL-CPG Direction Alignment')
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (3) RL Variance Contribution
    ax = axes[0, 2]
    ax.plot(steps, log_buffer['rl_contribution'], color='purple', linewidth=2)
    theoretical_contribution = (rl_weight**2) / (1 + rl_weight**2)
    ax.axhline(y=theoretical_contribution, color='orange', linestyle='--', 
               linewidth=2, label=f'Theory {theoretical_contribution:.2%}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Contribution Ratio')
    ax.set_title('RL Variance Contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (4) Magnitude Ratio Distribution
    ax = axes[1, 0]
    ratio = np.array(log_buffer['rl_magnitude']) / (np.array(log_buffer['prior_magnitude']) + 1e-8)
    ax.hist(ratio, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=rl_weight, color='red', linestyle='--', linewidth=2, 
               label=f'Expected {rl_weight:.2f}')
    ax.set_xlabel('RL / CPG Magnitude Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Magnitude Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (5) Correlation Distribution
    ax = axes[1, 1]
    ax.hist(log_buffer['correlation'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Direction Correlation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (6) RL vs CPG 貢獻百分比 - 堆疊面積圖
    ax = axes[1, 2]
    rl_mag   = np.array(log_buffer['rl_magnitude'])
    prior_mag = np.array(log_buffer['prior_magnitude'])
    total = rl_mag + prior_mag + 1e-8

    rl_pct    = rl_mag / total * 100
    prior_pct = prior_mag / total * 100

    ax.stackplot(steps, prior_pct, rl_pct,
                 labels=['CPG %', 'RL %'],
                 colors=['#4C72B0', '#DD8452'],
                 alpha=0.85)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Contribution (%)')
    ax.set_title('RL vs CPG Contribution (%)')
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='50% line')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    # 標出最後 1000 步的平均值
    recent = min(1000, len(rl_pct))
    avg_rl    = np.mean(rl_pct[-recent:])
    avg_prior = np.mean(prior_pct[-recent:])
    ax.text(0.05, 0.5,
            f'Last {recent} steps avg:\n'
            f'CPG: {avg_prior:.1f}%\n'
            f'RL:  {avg_rl:.1f}%',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 防止記憶體洩漏
    print(f"Chart saved: {save_path}")
    
    return fig