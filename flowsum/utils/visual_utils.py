import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from typing import List, Any, Tuple
import warnings

warnings.filterwarnings("ignore")


def visualize_latent_and_prior(
    Zs: List[Any],
    labels: List[str],
    title: str = "",
    figsize: Tuple[float, float] = (6, 10),
) -> Figure:
    """
    Args:
        Zs: a list of tensors or arrays
        labels: the legend labels

    [TODO] add save_path
    [TODO] add pairplot in seaborn

    References:
        (1) Make one subplot occupy multiple subplots: https://stackoverflow.com/a/2265506/13448382.
    """
    assert len(Zs) == len(labels)

    fig = plt.figure(figsize=figsize)

    # Add the main title
    if title:
        fig.suptitle(title.capitalize(), fontsize=15)

    # plot scatterplot of the 1st and the 2nd dimension
    plt.subplot(2, 1, 1)
    plt.title(r"Joint Distribution")
    plt.xlabel(r"$z_1$")
    plt.ylabel(r"$z_2$")
    colors = [None, "firebrick", "forestgreen"]  # [TODO] to add dynamically
    for i in range(len(Zs)):
        plt.scatter(
            Zs[i][:, 0], Zs[i][:, 1], color=colors[i], label=labels[i], alpha=0.5
        )
    plt.legend()

    # plot kernel density of the first two dimensions
    for j in range(2):  # j refers to the dimension index
        plt.subplot(2, 2, j + 3)
        for i in range(len(Zs)):  # i refers to the array index
            sns.distplot(
                Zs[i][:, j],
                hist=False,
                kde=True,
                bins=None,
                color=colors[i],
                hist_kws={"edgecolor": "black"},
                kde_kws={"linewidth": 2},
                label=labels[i],
            )
        plt.title(rf"$p(z_{j+1})$")
        # plt.legend()  # uncomment if want legend in the KDE plot

    plt.close()  # to avoid displaying

    return fig
