import itertools
import logging
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import umap
from constants import SEED
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

matplotlib.use("Agg")
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-colorblind')
sns.set_style("whitegrid", {'axes.grid' : False})
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
default_colours = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']
default_markers = ["d", "^", "s", "X", "+"]
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


def plot_umap(embeddings, colours, save_path, min_dist=0.1, n_neighbors=15, alpha=0.5, markers_offset=0):
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
    sns.scatterplot(data=dataframe, x="UMAP 1", y="UMAP 2",hue="Molecule type", style="Molecule type", size=20, palette=draw_colours, markers=markers, legend=False)
#    if isinstance(colours[0], str):
#        for counter, label in enumerate(colours.unique()):
#            plt.scatter(
#                umap_embeddings[colours == label, 0],
#                umap_embeddings[colours == label, 1],
#                s=20,
#                alpha=alpha,
#                c=default_colours[counter % len(draw_colours)],
#                marker=markers[counter % len(markers)],
#                label=label,
#            )
#        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, fontsize=10, markerscale=2)
#    else:
#        plt.scatter(
#            umap_embeddings[:, 0],
#            umap_embeddings[:, 1],
#            s=20,
#            c=colours,
#        )
#        plt.colorbar()
#   plt.ylabel("UMAP 2")
#   plt.xlabel("UMAP 1")
    fig.gca().spines['bottom'].set_color('black')
    fig.gca().spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=600, transparent=True)
    plt.savefig(str(save_path)[:-3]+"png", format="png", dpi=600, transparent=True)
    plt.clf()


def plot_pca(embeddings, colours, save_path, alpha=0.5, markers_offset=0):
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
    sns.scatterplot(data=dataframe, x=f"PC 1 ({variances[0]*100:.2f}%)", y=f"PC 2 ({variances[1]*100:.2f}%)",hue="Molecule type", style="Molecule type", size=20, palette=draw_colours, markers=markers, legend=False)
#    if isinstance(colours[0], str):
#        for counter, label in enumerate(colours.unique()):
#            plt.scatter(
#                pca_embeddings[colours == label, 0],
#                pca_embeddings[colours == label, 1],
#                s=20,
#                alpha=alpha,
#                c=default_colours[counter % len(draw_colours)],
#                marker=markers[counter % len(markers)],
#                label=label,
#            )
#        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=4, markerscale=2)
#    else:
#        plt.scatter(
#            pca_embeddings[:, 0],
#            pca_embeddings[:, 1],
#            s=20,
#            alpha=alpha,
#            c=colours,
#        )
#        plt.colorbar()  
#    plt.ylabel(f"PC 2 ({variances[1]*100:.2f}%)")
#    plt.xlabel(f"PC 1 ({variances[0]*100:.2f}%)")
    fig.gca().spines['bottom'].set_color('black')
    fig.gca().spines['left'].set_color('black')
    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=600, transparent=True)
    plt.savefig(str(save_path)[:-3]+"png", format="png", dpi=600, transparent=True)
    plt.clf()


def plot_representations(
    embeddings, colours, save_path_prefix, min_dist=0.1, n_neighbors=15, alpha=0.5
):
    logging.info("Started plotting PCA")
    plot_pca(embeddings, colours, Path(str(save_path_prefix) + "_pca.svg"), alpha)
    logging.info("Started plotting UMAP")
    plot_umap(
        embeddings,
        colours,
        Path(str(save_path_prefix) + f"{min_dist}_{n_neighbors}_umap.svg"),
        min_dist,
        n_neighbors,
        alpha,
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
