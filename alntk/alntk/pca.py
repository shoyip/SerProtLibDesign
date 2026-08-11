from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# def get_pca(reference_alignment, other_alignments, other_alignment_labels, pca_title):
#     ohe = OneHotEncoder(handle_unknown='ignore')
#     reference_seqs_ohe = ohe.fit_transform(reference_alignment)
    
#     pca = PCA(n_components=2, svd_solver='arpack')
#     pca.fit(reference_seqs_ohe)
#     reference_pca = pca.transform(reference_seqs_ohe)
    
#     alignments_pcas = [reference_pca]
#     for alignment, label in zip(other_alignments, other_alignment_labels):
#         alignment_ohe = ohe.transform(alignment)
#         alignment_pca = pca.transform(alignment_ohe)
#         alignments_pcas.append(alignment_pca, label=label)

#     fig, ax = plt.subplots()
#     ax.set_title(pca_title)
#     for alignment_pca in alignments_pcas:
#         ax.scatter(alignment_pca[:, 0], alignment_pca[:, 1], s=1)
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     return fig, ax

def get_pca(reference_alignment, pca_title):
    reference_seqs = reference_alignment.get_seqs()
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    reference_seqs_ohe = ohe.fit_transform(reference_seqs)
    
    pca = PCA(n_components=2, svd_solver='arpack')
    pca.fit(reference_seqs_ohe)
    reference_pca = pca.transform(reference_seqs_ohe)

    fig, ax = plt.subplots()
    ax.set_title(pca_title)
    ax.scatter(reference_pca[:, 0], reference_pca[:, 1], s=1, alpha=.1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    return fig, ax
    

# def ohe_seqs(seqs):
#     ohe = OneHotEncoder(handle_unknown='ignore')
#     seqs_ohe = ohe.fit_transform(seqs)

#     return seqs_ohe

# def pca_ohe_seqs(seqs_ohe):
#     pca = PCA(n_components=2, svd_solver='arpack')
#     pca.fit(seqs_ohe)
#     return pca.transform(seqs_ohe)

# def plot_pca(pca, pca_title, alpha=1):
#     fig, ax = plt.subplots()
#     ax.set_title(pca_title)
#     ax.scatter(pca[:, 0], pca[:, 1], s=1, alpha=alpha)
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     return fig, ax