#!/usr/bin/env python3
"""
t-SNE / UMAP parameter grid search for the (ICA-weighted) red sector.

Reproduces the data pipeline from notebooks/02_clustering_sector.ipynb (the
same dereplicated representative set + Ch/Tr/El reference sequences, the same
column-weighted one-hot encoding using the red sector's ICA loadings), then
sweeps a grid of t-SNE and UMAP parameters, saving one scatter plot per
parameter combination plus a combined grid-of-thumbnails overview per method,
so a "no clustering" result can be checked across the parameter space rather
than on a single, possibly unlucky, choice.

Usage:
    python scripts/tsne_umap_gridsearch.py

Run from the notebooks/ directory or anywhere with ../data/ reachable
(it resolves data paths relative to this file, so it can be run from
anywhere in the repo).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

from alntk.alignment import Alignment

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "figures" / "tsne_umap_gridsearch"

CH_ACCS = ["P07338", "P17538", "P80646", "P00767", "Q7M3E1", "Q61759", "Q6GPI", "P04813", "P00766"]
TR_ACCS = ["P00760", "P07477", "P00762", "P07478", "P35030", "Q9R0T7", "P07146", "P00763", "P35031", "P35033"]
EL_ACCS = ["P08217", "P08218", "P08419", "Q7SIG3", "P05208", "P00773", "Q91X79", "P00772", "Q29461", "P09093", "Q28153", "P00774"]

TSNE_PERPLEXITIES = [5, 15, 30, 50, 75, 100]
TSNE_LEARNING_RATES = ["auto", 1000]

UMAP_N_NEIGHBORS = [5, 15, 30, 50, 100]
UMAP_MIN_DISTS = [0.0, 0.1, 0.25, 0.5, 0.8]

SEED = 42


def find_representative(aln, identity_threshold, seed=42):
    """Greedy dereplication: repeatedly keep a random sequence and drop
    everything within `identity_threshold` matches of it."""
    N, L = aln.shape
    rng = np.random.default_rng(seed)

    remaining = set(range(N))
    representatives = []

    while remaining:
        curr_idx = rng.choice(list(remaining))
        curr_seq = aln[curr_idx]

        identity_values = np.sum(curr_seq == aln, axis=1)
        too_similar = set(np.where(identity_values >= identity_threshold)[0])
        remaining -= too_similar
        representatives.append(curr_idx)

    return representatives


def load_weighted_red_sector_matrix():
    """Reproduce the weighted one-hot matrix from 02_clustering_sector.ipynb,
    restricted to the dereplicated representatives + Ch/Tr/El references."""
    num = np.loadtxt(DATA_DIR / "iter_aln_v2_numbering.txt", dtype=str)
    red_sector = np.loadtxt(DATA_DIR / "iter_aln_v2_redsector.txt").astype(int)
    icweights = np.loadtxt(DATA_DIR / "iter_aln_v2_redsector_icweights.txt")

    red_sector_sorted = np.sort(red_sector)
    weights_sorted = icweights[np.argsort(red_sector)]

    aln = Alignment()
    aln.import_from_fasta(DATA_DIR / "iter_aln_v2.faa")
    seqs = aln.get_seqs()
    descs = aln.get_descs()

    seqs_red = seqs[:, red_sector_sorted]

    ch_idxs, tr_idxs, el_idxs = [], [], []
    for desc_idx, desc in enumerate(descs):
        for accession in desc.split("_"):
            if accession in CH_ACCS:
                ch_idxs.append(desc_idx)
            elif accession in TR_ACCS:
                tr_idxs.append(desc_idx)
            elif accession in EL_ACCS:
                el_idxs.append(desc_idx)

    representatives = find_representative(seqs_red, identity_threshold=7, seed=SEED)
    tsne_idxs = np.array(sorted(set(representatives) | set(ch_idxs) | set(tr_idxs) | set(el_idxs)))

    ohe = OneHotEncoder(handle_unknown="ignore")
    seqs_red_ohe = ohe.fit_transform(seqs_red)
    col_weights = np.concatenate([
        np.full(len(cats), w) for cats, w in zip(ohe.categories_, weights_sorted)
    ])
    seqs_red_ohe_weighted = seqs_red_ohe.multiply(col_weights).tocsr()

    X = seqs_red_ohe_weighted[tsne_idxs].toarray()

    idx_to_pos = {idx: pos for pos, idx in enumerate(tsne_idxs)}
    ch_pos = [idx_to_pos[i] for i in ch_idxs if i in idx_to_pos]
    tr_pos = [idx_to_pos[i] for i in tr_idxs if i in idx_to_pos]
    el_pos = [idx_to_pos[i] for i in el_idxs if i in idx_to_pos]
    rep_pos = [idx_to_pos[i] for i in representatives if i in idx_to_pos]

    print(f"Embedding {X.shape[0]} points (representatives + Ch/Tr/El), "
          f"{X.shape[1]} weighted one-hot features.")

    return X, {"ch": ch_pos, "tr": tr_pos, "el": el_pos, "rep": rep_pos}


def plot_embedding(ax, embedding, groups, title):
    pos_a, pos_b = 0, 1
    ax.scatter(embedding[groups["rep"], pos_a], embedding[groups["rep"], pos_b],
               s=3, alpha=0.3, color="tab:purple", label="Representatives")
    ax.scatter(embedding[groups["ch"], pos_a], embedding[groups["ch"], pos_b],
               s=40, alpha=1, marker="x", label="Ch")
    ax.scatter(embedding[groups["tr"], pos_a], embedding[groups["tr"], pos_b],
               s=40, alpha=1, marker="x", label="Tr")
    ax.scatter(embedding[groups["el"], pos_a], embedding[groups["el"], pos_b],
               s=40, alpha=1, marker="x", label="El")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def run_tsne_grid(X, groups, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = len(TSNE_PERPLEXITIES), len(TSNE_LEARNING_RATES)
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for i, perplexity in enumerate(TSNE_PERPLEXITIES):
        for j, learning_rate in enumerate(TSNE_LEARNING_RATES):
            print(f"t-SNE: perplexity={perplexity}, learning_rate={learning_rate}")
            tsne = TSNE(
                n_components=2, init="pca", random_state=SEED,
                perplexity=perplexity, learning_rate=learning_rate,
            )
            embedding = tsne.fit_transform(X)

            title = f"perplexity={perplexity}, lr={learning_rate}"

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_embedding(ax, embedding, groups, title)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"tsne_perp{perplexity}_lr{learning_rate}.png", dpi=150)
            plt.close(fig)

            ax_grid = axes_grid[i, j] if n_rows > 1 else axes_grid[j]
            plot_embedding(ax_grid, embedding, groups, title)

    axes_grid.flat[0].legend(fontsize=8)
    fig_grid.suptitle("t-SNE grid search: weighted red sector (representatives + Ch/Tr/El)")
    fig_grid.tight_layout()
    fig_grid.savefig(out_dir.parent / "tsne_grid.png", dpi=150)
    plt.close(fig_grid)
    print(f"Saved t-SNE grid overview to {out_dir.parent / 'tsne_grid.png'}")


def run_umap_grid(X, groups, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = len(UMAP_N_NEIGHBORS), len(UMAP_MIN_DISTS)
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for i, n_neighbors in enumerate(UMAP_N_NEIGHBORS):
        for j, min_dist in enumerate(UMAP_MIN_DISTS):
            print(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")
            reducer = UMAP(
                n_components=2, random_state=SEED,
                n_neighbors=n_neighbors, min_dist=min_dist,
            )
            embedding = reducer.fit_transform(X)

            title = f"n_neighbors={n_neighbors}, min_dist={min_dist}"

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_embedding(ax, embedding, groups, title)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"umap_nn{n_neighbors}_mindist{min_dist}.png", dpi=150)
            plt.close(fig)

            ax_grid = axes_grid[i, j] if n_rows > 1 else axes_grid[j]
            plot_embedding(ax_grid, embedding, groups, title)

    axes_grid.flat[0].legend(fontsize=8)
    fig_grid.suptitle("UMAP grid search: weighted red sector (representatives + Ch/Tr/El)")
    fig_grid.tight_layout()
    fig_grid.savefig(out_dir.parent / "umap_grid.png", dpi=150)
    plt.close(fig_grid)
    print(f"Saved UMAP grid overview to {out_dir.parent / 'umap_grid.png'}")


def main():
    X, groups = load_weighted_red_sector_matrix()
    run_tsne_grid(X, groups, FIGURES_DIR / "tsne")
    run_umap_grid(X, groups, FIGURES_DIR / "umap")


if __name__ == "__main__":
    main()
