"""Visualization utilities for molecular embeddings and model performance metrics.

This module provides functions for creating various plots including:
- UMAP and PCA visualizations of embeddings
- Correlation plots between fingerprint and embedding distances
- Performance metric bar plots
"""

import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from constants import SEED

# Configure matplotlib backend and styles
matplotlib.use("Agg")
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 2,
    'axes.edgecolor': 'black',
})

# Constants for visualization
DEFAULT_COLORS = ['#332288', '#117733', '#44AA99', '#88CCEE', 
                 '#DDCC77', '#CC6677', '#AA4499', '#882255']
DEFAULT_MARKERS = ["d", "^", "s", "X", "P"]
HATCHES = ["", "/", "\\", "|", "-", "x", "o", "+", "O", ".", "*"]
DEFAULT_HATCHES = ["".join(c) for c in list(itertools.product(HATCHES, HATCHES))[1:]]

# Page dimensions in inches
FULL_PAGE_WIDTH = 17 / 2.54
HALF_PAGE_WIDTH = 8.5 / 2.54
CM = 1 / 2.54  # Centimeters to inches
        
def _setup_figure(figsize: Tuple[float, float] = (HALF_PAGE_WIDTH, HALF_PAGE_WIDTH)) -> plt.Figure:
    """Create and configure a matplotlib figure."""
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    return fig

def _configure_axes(ax: plt.Axes) -> None:
    """Configure common axes properties."""
    ax.tick_params(length=8, width=1, labelsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_aspect('equal', adjustable='box')

def _calculate_axis_limits(x_values: np.ndarray, y_values: np.ndarray, margin_factor: float = 0.08) -> Tuple[float, float]:
    """Calculate symmetric axis limits with margins."""
    axis_min = min(x_values.min(), y_values.min())
    axis_max = max(x_values.max(), y_values.max())
    margin = margin_factor * (axis_max - axis_min) + 0.1
    return axis_min - margin, axis_max + margin

def _save_plot(save_path: Union[str, Path], formats: List[str] = ["svg", "png"], dpi: int = 600) -> None:
    """Save plot in multiple formats."""
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    
    for fmt in formats:
        output_path = save_path.with_suffix(f".{fmt}")
        plt.savefig(output_path, format=fmt, dpi=dpi, transparent=True)
    plt.clf()

def get_colors_markers(color_offset: int = 0, marker_offset: int = 0) -> Tuple[List[str], List[str]]:
    """Get color and marker sequences with optional offsets."""
    colors = (DEFAULT_COLORS[color_offset % len(DEFAULT_COLORS):] + 
                DEFAULT_COLORS[:color_offset % len(DEFAULT_COLORS)])
    markers = (DEFAULT_MARKERS[marker_offset % len(DEFAULT_MARKERS):] + 
                DEFAULT_MARKERS[:marker_offset % len(DEFAULT_MARKERS)])
    return colors, markers

def plot_legend(markers: List[str], labels: List[str], colors: List[str], 
                save_prefix: str, alpha: float = 0.6) -> None:
    """Create and save a standalone legend."""
    fig, ax = plt.subplots(figsize=(8.3 * CM, 2 * CM))
    legend_elements = [
        Line2D([0], [0], marker=markers[i], color='w', 
        label=labels[i], markerfacecolor=colors[i], 
        markersize=9) 
        for i in range(len(labels))
    ]
    
    ax.legend(
        handles=legend_elements, 
        loc='center', 
        fontsize=12, 
        ncol=len(legend_elements), 
        markerscale=2
    )
    ax.axis('off')
    fig.tight_layout()
    
    for fmt in ["svg", "png"]:
        fig.savefig(
            f"{save_prefix}_legend.{fmt}",
            format=fmt,
            dpi=600,
            bbox_inches='tight',
            transparent=True
        )
    plt.close(fig)

def plot_correlation(
    fingerprint_distances: List[float],
    embedding_distances: List[Tuple[str, List[float]]],
    save_path: Path,
    alpha: float = 0.2,
) -> None:
    """Plot correlation between fingerprint and embedding distances."""
    _setup_figure()
    
    for i, (label, distances) in enumerate(embedding_distances):
        plt.scatter(
            fingerprint_distances,
            distances,
            s=20,
            alpha=alpha,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            marker=DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)],
            label=label,
        )
    
    plt.legend()
    plt.ylabel("Embedding distance")
    plt.xlabel("Fingerprint distance")
    plt.tight_layout()
    _save_plot(save_path, ["pdf"])

def plot_umap(
    embeddings: np.ndarray,
    colors: List[str],
    save_path: Path,
    min_dist: float = 0.1,
    n_neighbors: int = 15,
    alpha: float = 0.6,
    markers_offset: int = 0,
) -> None:
    """Create and save UMAP visualization of embeddings."""
    logging.info("Creating UMAP visualization")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=SEED + 6539
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    
    fig = _setup_figure()
    draw_colors, markers = get_colors_markers(0, markers_offset)
    
    df = pd.DataFrame({
        "UMAP 1": umap_embeddings[:, 0],
        "UMAP 2": umap_embeddings[:, 1],
        "Molecule type": colors
    })
    
    sns.scatterplot(
        data=df,
        x="UMAP 1",
        y="UMAP 2",
        hue="Molecule type",
        style="Molecule type",
        size=20,
        palette=draw_colors,
        markers=markers,
        legend=False,
        alpha=alpha
    )
    
    ax = fig.gca()
    _configure_axes(ax)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    
    x_min, x_max = _calculate_axis_limits(df["UMAP 1"], df["UMAP 2"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    
    _save_plot(save_path)
    plot_legend(markers, df["Molecule type"].unique(), draw_colors, str(save_path)[:-4], alpha)

def plot_pca(
    embeddings: np.ndarray,
    colors: List[str],
    save_path: Path,
    alpha: float = 0.6,
    markers_offset: int = 0,
) -> None:
    """Create and save PCA visualization of embeddings."""
    logging.info("Creating PCA visualization")
    
    pca = PCA(n_components=2, random_state=SEED + 6541)
    pca_embeddings = pca.fit_transform(embeddings)
    variances = pca.explained_variance_ratio_
    
    logging.info(f"Explained variance: {variances}")
    
    fig = _setup_figure()
    draw_colors, markers = get_colors_markers(0, markers_offset)
    
    df = pd.DataFrame({
        f"PC 1 ({variances[0]*100:.2f}%)": pca_embeddings[:, 0],
        f"PC 2 ({variances[1]*100:.2f}%)": pca_embeddings[:, 1],
        "Molecule type": colors
    })
    
    sns.scatterplot(
        data=df,
        x=f"PC 1 ({variances[0]*100:.2f}%)",
        y=f"PC 2 ({variances[1]*100:.2f}%)",
        hue="Molecule type",
        style="Molecule type",
        size=20,
        palette=draw_colors,
        markers=markers,
        legend=False,
        alpha=alpha
    )
    
    ax = fig.gca()
    _configure_axes(ax)
    ax.set_xlabel(f"PC 1 ({variances[0]*100:.2f}%)", fontsize=12)
    ax.set_ylabel(f"PC 2 ({variances[1]*100:.2f}%)", fontsize=12)
    
    x_min, x_max = _calculate_axis_limits(
        df[f"PC 1 ({variances[0]*100:.2f}%)"], 
        df[f"PC 2 ({variances[1]*100:.2f}%)"]
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    
    _save_plot(save_path)
    plot_legend(markers, df["Molecule type"].unique(), draw_colors, str(save_path)[:-4], alpha)

def plot_representations(
    embeddings: np.ndarray,
    colors: List[str],
    save_path_prefix: Union[str, Path],
    min_dist: float = 0.1,
    n_neighbors: int = 15,
    alpha: float = 0.6,
    offset: int = 0,
) -> None:
    """Create both PCA and UMAP visualizations of embeddings."""
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    logging.info(f"Standardization means: {sorted(scaler.mean_)[:10]}")
    logging.info(f"Standardization variances: {sorted(scaler.var_)[:10]}")
    
    # Create visualizations
    plot_pca(
        embeddings, 
        colors, 
        Path(f"{save_path_prefix}_pca.svg"), 
        alpha, 
        offset
    )
    plot_umap(
        embeddings,
        colors,
        Path(f"{save_path_prefix}_{min_dist}_{n_neighbors}_umap.svg"),
        min_dist,
        n_neighbors,
        alpha,
        offset,
    )

def plot_scores(
    scores: Dict[str, List[float]],
    tests: List[str],
    y_label: str,
    save_path: Path,
    bar_width: Optional[float] = None,
) -> None:
    """Create bar plot of performance metrics."""
    plt.figure(figsize=(10, len(tests)))
    bar_width = bar_width or 0.9 / len(scores)
    
    for i, (key, values) in enumerate(scores.items()):
        positions = [p + i * bar_width for p in range(len(tests))]
        
        # Clean up labels for display
        display_key = (
            key.replace("_", " ")
            .replace("selfies", "SELFIES")
            .replace("smiles", "SMILES")
        )
        
        plt.bar(
            positions,
            values,
            width=bar_width,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            label=display_key,
        )
    
    plt.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
        handleheight=2,
        handlelength=2,
    )
    plt.xticks(
        [x + 0.5 for x in range(len(tests))], 
        tests, 
        rotation=45
    )
    plt.xlabel("Tasks")
    plt.ylabel(y_label)
    plt.tight_layout()
    _save_plot(save_path, ["pdf"])