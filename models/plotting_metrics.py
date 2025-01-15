import matplotlib.pyplot as plt


def plot_metric(metric, ax: plt.Axes, **kwargs):
    ax.plot(range(len(metric)), metric, **kwargs)
    ax.set_xlabel("Epochs")
    ax.grid(True)
    ax.legend()
    
def plot_metrics(df, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for col in df.columns:
        if "loss" in col:
            plot_metric(df[col], ax1, label=col)
            ax1.set_ylabel("loss")
        else:
            plot_metric(df[col], ax2, label=col)
            ax2.set_ylabel("accuracy")

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    return fig, (ax1, ax2)