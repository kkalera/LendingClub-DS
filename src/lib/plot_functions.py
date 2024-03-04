import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.sparse import issparse
import shap
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

COLOR_MAIN = "#69b3a2"
COLOR_CONTRAST = "#B3697A"
PALETTE = [COLOR_MAIN, COLOR_CONTRAST]


def get_cmap():
    """
    Returns a matplotlib colormap with a main color and a contrast color.

    Returns:
    matplotlib.colors.LinearSegmentedColormap: The matplotlib colormap.
    """
    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [
        [norm(-1.0), COLOR_CONTRAST],
        [norm(0.0), "#ffffff"],
        [norm(1.0), COLOR_MAIN],
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


def color_palette_husl(n_colors):
    return sns.color_palette("husl", n_colors=n_colors)


def countplot(
    data,
    column_name: str,
    title: str = "Countplot",
    hue: str = None,
    ax=None,
    figsize=(10, 5),
    bar_labels: bool = False,
    use_percentage: bool = True,
    horizontal: bool = False,
    palette=PALETTE,
):
    """
    Generate a countplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the countplot. Defaults to "Countplot".
        hue (str, optional): The column name to use for grouping the countplot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        bar_labels (bool, optional): Whether to add labels to the bars. Defaults to False.
        bar_label_kind (str, optional): The kind of labels to add to the bars. Can be "percentage" or "count". Defaults to "percentage".

    Returns:
        matplotlib.axes.Axes: The axis object containing the countplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    if hue:
        if horizontal:
            sns.countplot(
                data=data,
                y=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
        else:
            sns.countplot(
                data=data,
                x=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
    else:
        if horizontal:
            sns.countplot(data=data, y=column_name, ax=ax, color=COLOR_MAIN)
        else:
            sns.countplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN)

    if horizontal:
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.1)
    else:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.1)

    ## Add bar labels
    if bar_labels:
        for container in ax.containers:
            if use_percentage:
                ax.bar_label(container, fmt=lambda x: f" {x / len(data):.1%}")
            else:
                ax.bar_label(container, fmt=lambda x: f" {x}")

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def barplot(
    data,
    column_name: str,
    title: str = "Barplot",
    hue: str = None,
    ax=None,
    figsize=(10, 5),
    bar_labels: bool = False,
    horizontal: bool = False,
    palette=PALETTE,
    convert_amount=False,
):
    """
    Generate a barplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the barplot. Defaults to "barplot".
        hue (str, optional): The column name to use for grouping the barplot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        bar_labels (bool, optional): Whether to add labels to the bars. Defaults to False.
        bar_label_kind (str, optional): The kind of labels to add to the bars. Can be "percentage" or "bar". Defaults to "percentage".

    Returns:
        matplotlib.axes.Axes: The axis object containing the barplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    if hue:
        if horizontal:
            sns.barplot(
                data=data,
                x=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
        else:
            sns.barplot(
                data=data,
                y=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=palette,
                hue=hue,
            )
    else:
        if horizontal:
            sns.barplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN)
        else:
            sns.barplot(data=data, y=column_name, ax=ax, color=COLOR_MAIN)

    if horizontal:
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.1)
    else:
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.1)

    ## Add bar labels
    if bar_labels:
        for container in ax.containers:
            if convert_amount:
                ax.bar_label(container, fmt=lambda x: f" {convert_size(x)}")
            else:
                ax.bar_label(container, fmt=lambda x: f" {x}")

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def boxplot(
    data,
    column_name: str,
    title: str = "Boxplot",
    ax=None,
    figsize=(10, 5),
    y_lim: tuple = None,
    hue: str = None,
    palette=PALETTE,
):
    """
    Create a boxplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to create the boxplot for.
        title (str, optional): The title of the boxplot. Defaults to "Boxplot".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the boxplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue is None:
        sns.boxplot(
            data=data,
            y=column_name,
            ax=ax,
            color=COLOR_MAIN,
        )
    else:
        sns.boxplot(
            data=data,
            y=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
        )
    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def histplot(
    data,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 5),
    kde: bool = False,
    y_lim: tuple = None,
    bins="auto",
    palette=PALETTE,
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.histplot(
            data=data,
            x=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
            kde=kde,
            bins=bins,
        )
    else:
        sns.histplot(
            data=data, x=column_name, ax=ax, color=COLOR_MAIN, kde=kde, bins=bins
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def kdeplot(
    data,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 4),
    y_lim: tuple = None,
    bw_adjust: float = 1,
    palette=PALETTE,
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.kdeplot(
            data=data,
            x=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
            warn_singular=False,
            bw_adjust=bw_adjust,
        )
    else:
        sns.kdeplot(
            data=data,
            x=column_name,
            ax=ax,
            color=COLOR_MAIN,
            bw_adjust=bw_adjust,
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def scatterplot(
    data,
    x,
    y,
    title="Scatterplot",
    ax=None,
    figsize=(5, 10),
    hue=None,
    size=None,
    sizes=None,
    palette=PALETTE,
):
    """
    Plot a scatterplot of two numerical columns in a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    x (str): The name of the column to plot on the x-axis.
    y (str): The name of the column to plot on the y-axis.
    title (str): The title of the plot. Default is "Scatterplot".
    ax (matplotlib.axes.Axes): The axis to plot on. If not provided, a new axis will be created.
    figsize (tuple): The size of the figure. Default is (10, 5).

    Returns:
    matplotlib.axes.Axes: The axis object containing the scatterplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    if hue is None:
        sns.scatterplot(data=data, x=x, y=y, ax=ax, color=COLOR_MAIN)
    else:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            hue=hue,
            palette=palette,
            size=size,
            sizes=sizes,
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    return ax


def plot_distribution_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    bins="auto",
    hue=None,
    kde=False,
    palette=PALETTE,
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    if hue is None:
        histplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bins=bins,
            kde=kde,
            color=COLOR_MAIN,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            color=COLOR_MAIN,
        )
    else:
        histplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bins=bins,
            hue=hue,
            kde=kde,
            palette=palette,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            hue=hue,
            palette=palette,
        )
    fig.suptitle(title, fontsize=16)
    return fig


def plot_kde_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    hue=None,
    palette=PALETTE,
    bw_adjust: float = 1,
    x_lim_kde: tuple = None,
    y_lim_kde: tuple = None,
    x_lim_box: tuple = None,
    y_lim_box: tuple = None,
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    # assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    if hue is None:
        kdeplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bw_adjust=bw_adjust,
            color=COLOR_MAIN,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            color=COLOR_MAIN,
        )
    else:
        kdeplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            hue=hue,
            palette=palette,
            bw_adjust=bw_adjust,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            hue=hue,
            palette=palette,
        )

    if x_lim_kde:
        ax[0].set_xlim(x_lim_kde)
    if y_lim_kde:
        ax[0].set_ylim(y_lim_kde)
    if x_lim_box:
        ax[1].set_xlim(x_lim_box)
    if y_lim_box:
        ax[1].set_ylim(y_lim_box)

    fig.suptitle(title, fontsize=16)
    return fig


def plot_distribution_and_ratio(
    data,
    ratio: pd.Series,
    column_name: str,
    hue: str,
    title: str = "Distribution and Ratio",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    horizontal: bool = False,
    label_rotation: int = 0,
    use_percentage: bool = True,
    palette=PALETTE,
):
    """
    Plot the distribution and ratio of a categorical variable.

    Parameters:
    - data: The DataFrame containing the data.
    - ratio: The ratio of the categories.
    - column_name: The name of the categorical column.
    - hue: The column to use for grouping the data.
    - title: The title of the plot (default: "Distribution and Ratio").
    - ax: The matplotlib axes object to plot on (default: None).
    - figsize: The figure size (default: (10, 5)).
    - width_ratios: The width ratios of the subplots (default: [3, 1.25]).
    - horizontal: Whether to plot the bars horizontally (default: False).
    - label_rotation: The rotation angle of the tick labels (default: 0).
    """
    fig, ax = plt.subplots(
        figsize=figsize, nrows=1, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    countplot(
        data=data,
        column_name=column_name,
        hue=hue,
        title="Distribution",
        bar_labels=True,
        ax=ax.flatten()[0],
        horizontal=horizontal,
        use_percentage=use_percentage,
        palette=palette,
    )
    if horizontal:
        sns.barplot(
            y=ratio.index,
            x=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
        fig.subplots_adjust(wspace=0.7)
    else:
        sns.barplot(
            x=ratio.index,
            y=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    ax[1].set_title("Ratio")

    if label_rotation:
        if horizontal:
            for t1, t2 in zip(ax[0].get_yticklabels(), ax[1].get_yticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)
        else:
            for t1, t2 in zip(ax[0].get_xticklabels(), ax[1].get_xticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)

    fig.suptitle(title, fontsize=16)
    return fig


def correlation_matrix(corr, title="Correlation Matrix"):
    """
    Plot a correlation matrix heatmap.

    Parameters:
    corr (numpy.ndarray): The correlation matrix.
    title (str): The title of the plot. Default is "Correlation Matrix".

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = get_cmap()

    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1, fmt=".2f")
    fig.suptitle(title, fontsize=16)


def plot_roc_curve(model, x_test, y_test, ax=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    ax = plt.figure(figsize=(8, 6)) if ax is None else plt.sca(ax)
    plt.plot(fpr, tpr, color=COLOR_MAIN, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color=COLOR_CONTRAST, lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")


def plot_roc_and_confusion_matrix(model, X_test, y_test, normalize=True):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and the confusion matrix for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        figsize=(10, 5),
        nrows=1,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.3},
    )
    plt.grid(False)
    plot_roc_curve(model, X_test, y_test, ax=ax[0])
    cm = confusion_matrix(
        y_test, model.predict(X_test), normalize="all" if normalize else None
    )
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No issues", "Issues"]
    ).plot(include_values=True, cmap="Blues", ax=ax.flatten()[1])


def convert_size(amount):
    """
    Converts a given amount to a human-readable size representation.

    Args:
        amount (int): The amount to be converted.

    Returns:
        str: The human-readable size representation.

    Example:
        >>> convert_size(1024)
        '1.0 thousand'
    """
    if amount == 0:
        return "0"

    size_name = (
        " ",
        "thousand",
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
    )
    i = int(math.floor(math.log(amount, 1000)))
    p = math.pow(1024, i)
    s = round(amount / p, 2)
    return f"{s} {size_name[i]}"


def plot_xgb_shap(
    model, x, feature_names, figsize=(10, 5), multi_class=False, classes_to_plot=None
):
    if issparse(x):
        x = pd.DataFrame.sparse.from_spmatrix(x, columns=feature_names)
    else:
        x = pd.DataFrame(x, columns=feature_names)

    explainer = shap.Explainer(model)
    shap_values = explainer(x)

    if not multi_class:
        shap.plots.beeswarm(shap_values, plot_size=figsize, max_display=20)
    else:
        for c in classes_to_plot:
            print(f"Plotting SHAP values for class {c}")
            shap.summary_plot(shap_values[:, :, c], x, plot_size=figsize)
