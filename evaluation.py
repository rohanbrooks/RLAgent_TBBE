import pandas as pd

def analyse_agent_performance(df):
    import pandas as pd
    import numpy as np
    from scipy.stats import shapiro, mannwhitneyu
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Summary metrics
    summary = df.groupby("agent_type").agg(
        mean_balance=("final_balance", "mean"),
        std_balance=("final_balance", "std"),
        total_transactions=("num_transactions", "sum")
    )

    summary["mean_reward"] = summary["mean_balance"] - 100000000
    summary["amount_wagered"] = 15 * summary["total_transactions"]
    summary["roi"] = ((summary["mean_balance"] - 100000000) / summary["amount_wagered"]) * 100
    summary["sharpe_ratio"] = (summary["mean_balance"] - 100000000) / summary["std_balance"]
    summary = summary.reset_index()

    # Save summary to CSV
    summary.to_csv("agent_summary_metrics.csv", index=False)

    # Statistical comparison vs RL agent
    results = []
    agent_types = df['agent_type'].unique()
    agent_sizes = df['agent_type'].value_counts()

    if 'RLBettingAgent' in agent_types and agent_sizes['RLBettingAgent'] >= 3:
        rl_data = df[df['agent_type'] == 'RLBettingAgent']['final_balance']
        valid_agents = [a for a in agent_types if a != 'RLBettingAgent' and agent_sizes[a] >= 3]

        for agent in valid_agents:
            agent_data = df[df['agent_type'] == agent]['final_balance']
            shapiro_p = shapiro(agent_data)[1]
            mwu_p = mannwhitneyu(agent_data, rl_data, alternative='two-sided')[1]

            results.append({
                "comparison_agent": agent,
                "shapiro_p": shapiro_p,
                "normal": shapiro_p > 0.1,
                "mannwhitney_p": mwu_p,
                "significant": mwu_p < 0.1,
                "rl_median": rl_data.median(),
                "agent_median": agent_data.median(),
                "direction": (
                    "Up" if rl_data.median() > agent_data.median()
                    else "Down" if rl_data.median() < agent_data.median()
                    else "Equal"
                )
            })

        pd.DataFrame(results).to_csv("rl_agent_comparison.csv", index=False)

    # Boxplot for final balances
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="agent_type", y="final_balance")
    plt.xticks(rotation=45, ha="right")
    plt.title("Agent Final Balance Distribution")
    plt.tight_layout()
    plt.savefig("agent_balance_boxplot.png")
    plt.close()

    # Histogram of number of transactions
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="agent_type", y="num_transactions")
    plt.xticks(rotation=45, ha="right")
    plt.title("Agent Number of Transactions")
    plt.tight_layout()
    plt.savefig("agent_transactions_boxplot.png")
    plt.close()

    return summary

# Load again to enable local testing
df = pd.read_csv("agent_summary_statistics.csv")
l = analyse_agent_performance(df)
print(l)