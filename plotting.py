import itertools
import logging
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from constants import SEED

matplotlib.use("Agg")
plt.style.use("seaborn-colorblind")
markers = list(Line2D.markers.keys())
prop_cycle = plt.rcParams["axes.prop_cycle"]
default_colours = prop_cycle.by_key()["color"]
hatches = ["", "/", "\\", "|", "-", "x", "o", "+", "O", ".", "*"]
default_hatches = [
    "".join(combi) for combi in list(itertools.product(hatches, hatches))[1:]
]


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
            s=plt.rcParams["lines.markersize"] ** 2 / 3,
            alpha=alpha,
            c=default_colours[number % len(default_colours)],
            marker=markers[number % len(markers)],
            label=label,
        )
    plt.legend()
    plt.ylabel("Embedding distance")
    plt.xlabel("Fingerprint distance")
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.clf()


def plot_umap(embeddings, colours, save_path, min_dist=0.1, n_neighbors=15, alpha=0.2):
    os.makedirs(save_path.parent, exist_ok=True)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED + 6539
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    if isinstance(colours[0], str):
        for counter, label in enumerate(colours.unique()):
            plt.scatter(
                umap_embeddings[colours == label, 0],
                umap_embeddings[colours == label, 1],
                s=plt.rcParams["lines.markersize"] ** 2 / 3,
                alpha=alpha,
                c=default_colours[counter % len(default_colours)],
                marker=markers[counter % len(markers)],
                label=label,
            )
        plt.legend()
    else:
        plt.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            s=plt.rcParams["lines.markersize"] ** 2 / 3,
            c=colours,
        )
        plt.colorbar()
    plt.ylabel("UMAP 2")
    plt.xlabel("UMAP 1")
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.clf()


def plot_pca(embeddings, colours, save_path, alpha=0.2):
    os.makedirs(save_path.parent, exist_ok=True)
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    logging.info(
        f"{save_path} has the explained variance of {pca.explained_variance_ratio_}"
    )
    if isinstance(colours[0], str):
        for counter, label in enumerate(colours.unique()):
            plt.scatter(
                pca_embeddings[colours == label, 0],
                pca_embeddings[colours == label, 1],
                s=plt.rcParams["lines.markersize"] ** 2 / 3,
                alpha=alpha,
                c=default_colours[counter % len(default_colours)],
                marker=markers[counter % len(markers)],
                label=label,
            )
        plt.legend()
    else:
        plt.scatter(
            pca_embeddings[:, 0],
            pca_embeddings[:, 1],
            s=plt.rcParams["lines.markersize"] ** 2 / 3,
            alpha=alpha,
            c=colours,
        )
        plt.colorbar()
    plt.ylabel("PCA 2")
    plt.xlabel("PCA 1")
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.clf()


def plot_representations(
    embeddings, colours, save_path_prefix, min_dist=0.1, n_neighbors=15, alpha=0.2
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
            hatch=default_hatches[it % len(default_hatches)],
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
    plt.savefig(save_path, format="svg")
    plt.clf()
