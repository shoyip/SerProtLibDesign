#!/usr/bin/env python3
"""
Clustering Compatibility Metric (FIXED)

Computes how compatible two clustering assignments are compared to a null model
of random assignment. Uses mutual information-based metrics and a permutation test.

This version fixes the AMI calculation to use the correct formula.

Usage:
    python clustering_compatibility_fixed.py labels1.txt labels2.txt

Each file should contain one integer label per line, with the same number of lines.
"""

import numpy as np
from scipy.special import gammaln
from collections import Counter
import sys


def load_labels(filepath):
    """Load clustering labels from a text file (one integer per line)."""
    with open(filepath, 'r') as f:
        labels = np.array([int(line.strip()) for line in f if line.strip()])
    return labels


def compute_contingency_matrix(labels1, labels2):
    """
    Compute the contingency matrix between two clustering assignments.
    """
    unique1 = np.unique(labels1)
    unique2 = np.unique(labels2)
    
    map1 = {label: idx for idx, label in enumerate(unique1)}
    map2 = {label: idx for idx, label in enumerate(unique2)}
    
    contingency = np.zeros((len(unique1), len(unique2)), dtype=int)
    
    for l1, l2 in zip(labels1, labels2):
        contingency[map1[l1], map2[l2]] += 1
    
    return contingency, unique1, unique2


def mutual_information(labels1, labels2):
    """Compute mutual information between two clustering assignments."""
    contingency, _, _ = compute_contingency_matrix(labels1, labels2)
    n = contingency.sum()
    
    if n == 0:
        return 0.0
    
    p_xy = contingency / n
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def entropy_of_labels(labels):
    """Compute Shannon entropy of label distribution (natural log)."""
    counts = np.array(list(Counter(labels).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))


def normalized_mutual_information(labels1, labels2):
    """
    Compute Normalized Mutual Information (NMI).
    
    NMI = 2 * MI(X;Y) / (H(X) + H(Y))
    
    Range: [0, 1] where 1 means perfect agreement.
    """
    mi = mutual_information(labels1, labels2)
    h1 = entropy_of_labels(labels1)
    h2 = entropy_of_labels(labels2)
    
    if h1 + h2 == 0:
        return 0.0
    
    return 2 * mi / (h1 + h2)


def expected_mutual_information_hypergeometric(contingency):
    """
    Compute expected mutual information under the hypergeometric null model.
    
    This is the correct way to compute E[MI] for AMI, accounting for the
    combinatorial structure of contingency tables with fixed marginals.
    
    Based on the formula from:
    "Information Theoretic Measures for Clusterings Comparison" (Vinh et al., 2010)
    """
    n = contingency.sum()
    a = contingency.sum(axis=1)  # row sums
    b = contingency.sum(axis=0)  # column sums
    
    # Precompute log factorials for efficiency
    def log_comb(n, k):
        """Log of binomial coefficient C(n, k)."""
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    
    expected_mi = 0.0
    
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            # Sum over all possible values of n_ij
            # n_ij ranges from max(0, a_i + b_j - n) to min(a_i, b_j)
            min_nij = max(0, a[i] + b[j] - n)
            max_nij = min(a[i], b[j])
            
            for nij in range(min_nij, max_nij + 1):
                if nij == 0:
                    continue
                
                # Hypergeometric probability:
                # P(n_ij) = C(a_i, n_ij) * C(n - a_i, b_j - n_ij) / C(n, b_j)
                log_prob = (log_comb(a[i], nij) + 
                           log_comb(n - a[i], b[j] - nij) - 
                           log_comb(n, b[j]))
                prob = np.exp(log_prob)
                
                # Contribution to E[MI]: P(n_ij) * (nij/n) * log((nij/n) / ((a_i/n) * (b_j/n)))
                # = P(n_ij) * (nij/n) * log(nij * n / (a_i * b_j))
                if prob > 0:
                    term = prob * (nij / n) * np.log(nij * n / (a[i] * b[j]))
                    expected_mi += term
    
    return expected_mi


def adjusted_mutual_information(labels1, labels2):
    """
    Compute Adjusted Mutual Information (AMI).
    
    AMI = (MI - E[MI]) / (avg(H(X), H(Y)) - E[MI])
    
    where avg(H(X), H(Y)) = (H(X) + H(Y)) / 2
    
    Adjusts for chance: AMI ≈ 0 for random assignments, AMI = 1 for perfect agreement.
    Can be slightly negative for worse-than-random agreement.
    """
    contingency, _, _ = compute_contingency_matrix(labels1, labels2)
    n = contingency.sum()
    
    if n == 0:
        return 0.0
    
    mi = mutual_information(labels1, labels2)
    h1 = entropy_of_labels(labels1)
    h2 = entropy_of_labels(labels2)
    
    # Expected MI under null model
    expected_mi = expected_mutual_information_hypergeometric(contingency)
    
    # Average entropy
    avg_h = (h1 + h2) / 2
    
    # AMI
    if avg_h - expected_mi == 0:
        return 0.0
    
    ami = (mi - expected_mi) / (avg_h - expected_mi)
    
    return ami


def permutation_test(labels1, labels2, n_permutations=1000, random_state=42):
    """
    Perform a permutation test to assess significance of clustering agreement.
    """
    np.random.seed(random_state)
    
    observed_nmi = normalized_mutual_information(labels1, labels2)
    
    null_nmis = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted_labels2 = np.random.permutation(labels2)
        null_nmis[i] = normalized_mutual_information(labels1, permuted_labels2)
    
    p_value = (np.sum(null_nmis >= observed_nmi) + 1) / (n_permutations + 1)
    
    return observed_nmi, null_nmis.mean(), null_nmis.std(), p_value, null_nmis


def main():
    if len(sys.argv) != 3:
        print("Usage: python clustering_compatibility_fixed.py <labels1.txt> <labels2.txt>")
        sys.exit(1)
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    labels1 = load_labels(file1)
    labels2 = load_labels(file2)
    
    if len(labels1) != len(labels2):
        print(f"Error: Files have different numbers of labels ({len(labels1)} vs {len(labels2)})")
        sys.exit(1)
    
    n_samples = len(labels1)
    n_clusters1 = len(np.unique(labels1))
    n_clusters2 = len(np.unique(labels2))
    
    print("=" * 60)
    print("Clustering Compatibility Analysis (FIXED)")
    print("=" * 60)
    print(f"\nFile 1: {file1}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Clusters: {n_clusters1}")
    print(f"\nFile 2: {file2}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Clusters: {n_clusters2}")
    
    mi = mutual_information(labels1, labels2)
    nmi = normalized_mutual_information(labels1, labels2)
    ami = adjusted_mutual_information(labels1, labels2)
    
    print("\n" + "-" * 60)
    print("Agreement Metrics")
    print("-" * 60)
    print(f"Mutual Information (MI):     {mi:.6f}")
    print(f"Normalized MI (NMI):         {nmi:.6f}  [0 = independent, 1 = perfect]")
    print(f"Adjusted MI (AMI):           {ami:.6f}  [~0 = random, 1 = perfect]")
    
    print("\n" + "-" * 60)
    print("Permutation Test (null model: random assignment)")
    print("-" * 60)
    
    n_permutations = 1000
    obs_nmi, null_mean, null_std, p_val, null_nmis = permutation_test(
        labels1, labels2, n_permutations=n_permutations
    )
    
    print(f"Observed NMI:                {obs_nmi:.6f}")
    print(f"Null distribution (NMI):     {null_mean:.6f} ± {null_std:.6f}")
    print(f"Z-score:                     {(obs_nmi - null_mean) / null_std:.2f}")
    print(f"Empirical p-value:           {p_val:.4f}")
    
    print("\n" + "-" * 60)
    print("Interpretation")
    print("-" * 60)
    
    if ami < 0.05:
        print("→ Clustering assignments are nearly independent (random-like)")
    elif ami < 0.2:
        print("→ Weak agreement between clustering methods")
    elif ami < 0.5:
        print("→ Moderate agreement between clustering methods")
    elif ami < 0.7:
        print("→ Strong agreement between clustering methods")
    else:
        print("→ Very strong agreement between clustering methods")
    
    if p_val < 0.001:
        print("→ Agreement is highly significant (p < 0.001)")
    elif p_val < 0.01:
        print("→ Agreement is significant (p < 0.01)")
    elif p_val < 0.05:
        print("→ Agreement is marginally significant (p < 0.05)")
    else:
        print("→ Agreement is not statistically significant (p ≥ 0.05)")
    
    print("\n" + "=" * 60)
    
    np.save('null_distribution.npy', null_nmis)
    print(f"\nNull distribution saved to: null_distribution.npy")


if __name__ == "__main__":
    main()