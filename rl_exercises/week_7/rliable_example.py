import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import get_interval_estimates, metrics
from rliable.plot_utils import plot_sample_efficiency_curve

# Read data from different runs
df_s0 = pd.read_csv("training_data_seed_0.csv")
df_s1 = pd.read_csv("training_data_seed_1.csv")

# Add a column to distinguish between seeds
df_s0["seed"] = 0
df_s1["seed"] = 1

# Combine the dataframes and convert to numpy array
df = pd.concat([df_s0, df_s1], ignore_index=True)
steps = df["steps"].to_numpy()
score_dict = {"dqn": df["rewards"].to_numpy()}

iqm = lambda scores: np.array(  # noqa: E731
    [
        metrics.aggregate_iqm(scores[..., eval_idx])
        for eval_idx in range(scores.shape[-1])
    ]
)
iqm_scores, iqm_cis = get_interval_estimates(
    score_dict,
    iqm,
    reps=2000,
)
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["dqn"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)
plt.legend()
plt.tight_layout()
plt.show()
