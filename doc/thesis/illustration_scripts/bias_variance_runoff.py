from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../../figstyle.mplstyle")

fig, ax = plt.subplots(1, 1)
ax.set_title("The bias-variance tradeoff")
x = np.linspace(-2, 2, 1000)

bias = np.exp(-x)
variance = np.exp(x)
total = bias + variance


ax.plot(x, bias)
ax.plot(x, variance)
ax.plot(x, total)
ax.legend(["Bias", "Variance", "Total error"])
ax.tick_params(
    axis="both",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    left=False,  # ticks along the top edge are off
    labelbottom=False,
    labelleft=False,
)  # labels along the bottom edge are off
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black")
bbox_props2 = dict(boxstyle="larrow,pad=0.3", fc="white", ec="black")
t = ax.text(
    1.5,
    1,
    "Higher complexity",
    ha="center",
    va="center",
    rotation=0,
    size=10,
    bbox=bbox_props,
)
t2 = ax.text(
    -1.5,
    1,
    "Lower complexity",
    ha="center",
    va="center",
    rotation=0,
    size=10,
    bbox=bbox_props2,
)
ax.set_xlabel("Model complexity")
ax.set_ylabel("Error")
fig.tight_layout()
path = Path("../figures/examples")
fig.savefig(path / "bias_variance_runoff.pdf")
