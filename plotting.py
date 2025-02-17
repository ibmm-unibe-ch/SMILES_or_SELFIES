import itertools
import logging
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import umap
from constants import SEED
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

matplotlib.use("Agg")
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-colorblind')
#sns.set_style("whitegrid", {'axes.grid' : False})
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
default_colours = ['#882e72','#1965B0', '#7bafde', '#4eb265', '#cae0ab', '#f7f056', '#f1932d', '#dc050c']
default_markers = ["d", "^", "s", "X", "P"]
hatches = ["", "/", "\\", "|", "-", "x", "o", "+", "O", ".", "*"]
default_hatches = [
    "".join(combi) for combi in list(itertools.product(hatches, hatches))[1:]
]

full_page_width_in_inches = 17/2.54
half_page_width_in_inches = 8.5/2.54

def get_colours_markers(colour_offset:int, marker_offset:int):
    colours = default_colours[colour_offset%len(default_colours):] + default_colours[:colour_offset%len(default_colours)]
    markers = default_markers[marker_offset%len(default_markers):] + default_markers[:marker_offset%len(default_markers)]
    return colours, markers

def plot_correlation(
    fingerprint_distances: List[float],
    embedding_distances: List[Tuple[str, List[float]]],
    save_path: Path,
    alpha: float = 0.2,
):
    os.makedirs(save_path.parent, exist_ok=True)
    for number, (label, embedding_distance) in enumerate(embedding_distances):
        plt.scatter(
            fingerprint_distances,
            embedding_distance,
            s=20,
            alpha=alpha,
            c=default_colours[number % len(default_colours)],
            marker=markers[number % len(markers)],
            label=label,
        )
    plt.legend()
    plt.ylabel("Embedding distance")
    plt.xlabel("Fingerprint distance")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=600, transparent=True)
    plt.clf()


def plot_umap(embeddings, colours, save_path, min_dist=0.1, n_neighbors=15, alpha=0.6, markers_offset=0):
    os.makedirs(save_path.parent, exist_ok=True)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED + 6539
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(half_page_width_in_inches, half_page_width_in_inches)
    draw_colours, markers = get_colours_markers(0, markers_offset)
    dataframe = pd.DataFrame()
    dataframe["UMAP 1"] = umap_embeddings[:,0]
    dataframe["UMAP 2"] = umap_embeddings[:,1]
    dataframe["Molecule type"] = colours
    sns.scatterplot(data=dataframe, x="UMAP 1", y="UMAP 2",hue="Molecule type", style="Molecule type", size=20, palette=draw_colours, markers=markers, legend=False, alpha=alpha)
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines["left"].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines["bottom"].set_color('black')
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    # Set tick parameters
    ax.tick_params(length=8, width=1, labelsize=12)
    # Set major tick locators
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')
    # Calculate the limits for the axes
    x_min, x_max = dataframe[f"UMAP 1"].min(), dataframe[f"UMAP 1"].max()
    y_min, y_max = dataframe[f"UMAP 2"].min(), dataframe[f"UMAP 2"].max()
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)
    margin = 0.08 * (axis_max - axis_min) + 0.1 # 5% margin
    
    # Set the same limits for both axes with margins
    ax.set_xlim(axis_min - margin, axis_max + margin)
    ax.set_ylim(axis_min - margin, axis_max + margin)

    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=600, transparent=True)
    plt.savefig(str(save_path)[:-3]+"png", format="png", dpi=600, transparent=True)
    plt.clf()
    markers, labels, draw_colours = markers, dataframe["Molecule type"].unique(), draw_colours
    plot_legend(markers, labels, draw_colours, str(save_path)[:-4], alpha)

def plot_legend(markers, labels, draw_colours, save_path, alpha):
    cm = 1/2.54 
    fig, ax = plt.subplots(figsize=(8.3*cm, 2*cm))  # Adjust the size as needed
    legend_elements=[]
    for it in range(len(labels)):
        legend_elements.append(Line2D([0], [0], marker=markers[it], color='w', label=labels[it], markerfacecolor=draw_colours[it], markersize=9))
    ax.legend(handles=legend_elements, loc='center', fontsize=12, ncol=len(legend_elements), markerscale=2)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f"{save_path}_legend.svg", format="svg", dpi=600, bbox_inches='tight', transparent=True)
    fig.savefig(f"{save_path}_legend.png", format="png", dpi=600, bbox_inches="tight", transparent=True)
    fig.clf()

def plot_pca(embeddings, colours, save_path, alpha=0.6, markers_offset=0):
    print(f"markers_offset={markers_offset}")
    print(f"alpha={alpha}")
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    variances = pca.explained_variance_ratio_
    logging.info(
        f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    )
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(half_page_width_in_inches, half_page_width_in_inches)
    draw_colours, markers = get_colours_markers(0, markers_offset)
    dataframe = pd.DataFrame()
    dataframe[f"PC 1 ({variances[0]*100:.2f}%)"] = pca_embeddings[:,0]
    dataframe[f"PC 2 ({variances[1]*100:.2f}%)"] = pca_embeddings[:,1]
    dataframe["Molecule type"] = colours
    sns.scatterplot(data=dataframe, x=f"PC 1 ({variances[0]*100:.2f}%)", y=f"PC 2 ({variances[1]*100:.2f}%)",hue="Molecule type", style="Molecule type", size=20, palette=draw_colours, markers=markers, legend=False, alpha=alpha)
    ax = fig.gca()
    ax.set_xlabel(f"PC 1 ({variances[0]*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"PC 2 ({variances[1]*100:.2f}%)", fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines["left"].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines["bottom"].set_color('black')
    # Set tick parameters
    ax.tick_params(length=8, width=1, labelsize=12)
    # Set major tick locators
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')
    # Calculate the limits for the axes
    x_min, x_max = dataframe[f"PC 1 ({variances[0]*100:.2f}%)"].min(), dataframe[f"PC 1 ({variances[0]*100:.2f}%)"].max()
    y_min, y_max = dataframe[f"PC 2 ({variances[1]*100:.2f}%)"].min(), dataframe[f"PC 2 ({variances[1]*100:.2f}%)"].max()
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)
    margin = 0.08 * (axis_max - axis_min) + 0.1 # 5% margin
    
    # Set the same limits for both axes with margins
    ax.set_xlim(axis_min - margin, axis_max + margin)
    ax.set_ylim(axis_min - margin, axis_max + margin)

    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=600, transparent=True)
    plt.savefig(str(save_path)[:-3]+"png", format="png", dpi=600, transparent=True)
    plt.clf()
    markers, labels, draw_colours = markers, dataframe["Molecule type"].unique(), draw_colours
    plot_legend(markers, labels, draw_colours, str(save_path)[:-4], alpha)
    

def plot_representations(
    embeddings, colours, save_path_prefix, min_dist=0.1, n_neighbors=15, alpha=0.6, offset=0,
):
    logging.info("Started plotting PCA")
    plot_pca(embeddings, colours, Path(str(save_path_prefix) + "_pca.svg"), alpha, offset)
    logging.info("Started plotting UMAP")
    plot_umap(
        embeddings,
        colours,
        Path(str(save_path_prefix) + f"{min_dist}_{n_neighbors}_umap.svg"),
        min_dist,
        n_neighbors,
        alpha,
        offset,
    )


def plot_scores(scores: dict, tests, y_label, save_path, bar_width=None):
    os.makedirs(save_path.parent, exist_ok=True)
    plt.figure(figsize=(10, len(tests)))
    if bar_width is None:
        bar_width = 0.9 / len(scores)
    for it, (key, value) in enumerate(scores.items()):
        bar_position = [
            raw_position + it * bar_width for raw_position in range(len(tests))
        ]
        plt.bar(
            bar_position,
            value,
            width=bar_width,
            color=default_colours[it % len(default_colours)],
            label=key.replace("_", " ")
            .replace("selfies", "SELFIES")
            .replace("smiles", "SMILES"),
            #hatch=default_hatches[it % len(default_hatches)],
        )
    plt.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
        handleheight=2,
        handlelength=2,
    )
    plt.xticks([x_tick + 0.5 for x_tick in range(len(tests))], tests, rotation=45)
    plt.xlabel("Tasks")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", transparent=True)
    plt.clf()