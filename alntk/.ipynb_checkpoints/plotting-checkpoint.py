import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def default_plot_style():
    """
    Call function at the beginning of the file in order to set the default style.
    """
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['font.family'] = 'sans'
    mpl.rcParams['font.size'] = 12
    
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['axes.titlepad'] = 10.
    mpl.rcParams['axes.labelpad'] = 8.
    
    mpl.rcParams['patch.force_edgecolor'] = True
    mpl.rcParams['patch.edgecolor'] = 'black'
    mpl.rcParams['patch.linewidth'] = 0.4
    mpl.rcParams['patch.facecolor'] = 'white'
    
    color_cycle = ['377eb8', 'e41a1c', '4daf4a', '984ea3', 'ff7f00', 'ffff33', 'a65628', 'f781bf', '999999']
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler("color", color_cycle)
    return color_cycle

def plot_length_distrib(aln, aln_title, bins=50):
    """
    """
    aln_seqs = aln.get_seqs()
    lengths = np.sum(aln_seqs != '-', axis=1)
    
    fig, ax = plt.subplots()
    ax.set_title(aln_title)
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    ax.hist(lengths, bins=bins)

    return fig

def plot_gap_distrib(fig_data):
    fig = plt.figure()
    # plt.title('Gaps per sequence')
    for dataset in fig_data:
        plt.hist(dataset['data'], label=dataset['label'], facecolor='none', edgecolor='#'+dataset['color'], linewidth=1)
    plt.xlabel('Ratio of gaps')
    plt.ylabel('Occurences')
    plt.legend()
plt.show()