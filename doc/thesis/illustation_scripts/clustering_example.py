from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster as spc
import pandas as pd

fig, (ax1, ax2) = plt.subplots(2,1, figsize=[4.7747, 1.618*4.7747])
x = np.arange(0, 100)
X = np.array([x, -x, 2*x, x**2, -x**2, 2*x**2, x**3, -x**3, 2*x**3, x**4])
df = pd.DataFrame(X.T, columns=[r"$x$", r"$-x$", r"$2x$", r"$x^2$", r"$-x^2$", r"$2x^2$", r"$x^3$", r"$-x^3$", r"$2x^3$", r"$x^4$"])
corr = df.corr()
# Abs corr or not?
corr_linkage = spc.hierarchy.ward(np.abs(corr))
dendro = spc.hierarchy.dendrogram(corr_linkage, labels=df.columns, ax=ax1)
ax1.set_title("Pearson correlation")
# ax.set_yscale("log")
#fig.savefig("../figures/examples/cluster_example.pdf")


ax2.set_title("Spearman correlation")
corr_spearman = df.corr(method="spearman")
corr_linkage = spc.hierarchy.ward(corr_spearman)
dendro = spc.hierarchy.dendrogram(corr_linkage, labels=df.columns, ax=ax2)
fig.tight_layout()
path = Path("../figures/examples")
path.mkdir(exist_ok=True)
fig.savefig(path  / "cluster_example.pdf")
