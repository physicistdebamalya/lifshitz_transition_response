"""APS/PRB figure style, lifted verbatim from the notebook that produced
BLG-Ratios_vs_Ef.pdf so every figure in the project comes out identical.

    from plotstyle import use_prb, PRB_FIGSIZE, COLOR_X, COLOR_Y
    use_prb()
    fig, ax = plt.subplots(figsize=PRB_FIGSIZE, constrained_layout=True)
"""

import matplotlib as mpl

PRB_FIGSIZE = (3.4, 2.8)     # single PRB column
COLOR_X = "#1f77b4"
COLOR_Y = "#d62728"

PRB_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix",

    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    "savefig.dpi": 600,
    "pdf.fonttype": 42,      # embed as TrueType, not Type-3: required by APS
    "ps.fonttype": 42,
}


def use_prb():
    mpl.rcParams.update(PRB_RC)
