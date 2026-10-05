import numpy as np

GAP_AND_20_AA = list("-ACDEFGHIKLMNPQRSTVWY")


def compute_aa_freqs(seqs, seq_weights=None, alphabet=GAP_AND_20_AA):
    """
    Per-position amino acid frequencies over an alignment.

    seqs: (N_seqs, N_pos) array of single-character residues.
    seq_weights: (N_seqs,) per-sequence (row) weights, or None for
        unweighted counts.
    alphabet: symbols to count frequencies for (default: gap + 20 aa, q=21).

    Returns (N_pos, len(alphabet)) array, each row summing to ~1.
    """
    seqs = np.asarray(seqs)
    n_seqs = seqs.shape[0]
    if seq_weights is None:
        seq_weights = np.ones(n_seqs)
    seq_weights = np.asarray(seq_weights, dtype=float)
    m_eff = seq_weights.sum()

    freqs = np.zeros((seqs.shape[1], len(alphabet)))
    for a_idx, aa in enumerate(alphabet):
        is_aa = (seqs == aa)
        freqs[:, a_idx] = (is_aa * seq_weights[:, None]).sum(axis=0) / m_eff
    return freqs


def normalized_entropy(freqs):
    """
    Shannon entropy per position, normalized by log(q) so values fall in
    [0, 1] (0 = fully conserved, 1 = uniform over all q symbols).

    freqs: (N_pos, q) frequency array (rows summing to ~1).
    """
    freqs = np.asarray(freqs)
    q = freqs.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(freqs > 0, freqs * np.log(freqs), 0.0)
    h = -terms.sum(axis=1)
    return h / np.log(q)
