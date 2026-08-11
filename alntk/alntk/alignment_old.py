import numpy as np
from Bio import SeqIO

def import_from_fasta(fasta_file):
    """
    Import an array of descriptions and an array of sequences from a FASTA file.

    Parameters
    ----------
    fasta_file: str
        filename of FASTA file, typically `.fasta`, `.faa` or `.fa`.

    Returns
    -------
    descs_arr: ndarray
        1D array containing the descriptions from the FASTA file
    seqs_arr: ndarray
        2D array containing the sequence alignment, with one sequence
        per row and one position per column
    """
    seqs = []
    descs = []
    
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seqs.append(str(record.seq))
        descs.append(record.description.split('/')[0])

    descs_arr = np.array(descs)
    seqs_arr = np.array([[residue for residue in seq] for seq in seqs])
    return descs_arr, seqs_arr

def get_unaligned_seqs(aln_seqs):
    """
    Get a list of unaligned sequences from the alignment sequences array.

    Parameters
    ----------
    aln_seqs: ndarray
        2D array containing the sequence alignment

    Returns
    -------
    unaln_seqs: list
        List of strings, one for each sequence of the alignment
    """
    unaln_seqs = []
    for seq in aln_seqs:
        unaln_seq = (''.join(seq)).replace('-', '')
        unaln_seqs.append(unaln_seq)
    return unaln_seqs

def get_compact_alignment(aln_descs, aln_seqs, gap_threshold_ratio, return_gappy_pos=False, tol_uaa=0.):
    """
    Compactify the alignment by deleting sequences and consequently
    deleting entirely gapped columns.
    """

    # Get the gap threshold in terms of integer number of gaps from the ratio
    gap_threshold = int(gap_threshold_ratio*aln_seqs.shape[0])

    # Get the gap / no gap boolean matrix
    aln_seqs_isgap = (aln_seqs == '-')
    
    # Count the number of gaps in each column
    gaps_per_column = np.sum(aln_seqs_isgap, axis=0)

    # Get the index of all the columns that have more gaps than the threshold
    # We call these "gappy positions"
    gappy_positions = np.where(gaps_per_column > gap_threshold)[0]

    # Count how many gaps do sequences have in "gappy positions"
    # We call these counts of "unusual amino acids"
    unusual_aa_per_sequence = np.sum(aln_seqs[:, gappy_positions] != '-', axis=1)

    # Get the index of sequences that have "unusual amino acids" less than tol_uaa
    int_tol_uaa = int(aln_seqs.shape[1] * tol_uaa)
    sequences_mask = np.where(unusual_aa_per_sequence <= int_tol_uaa)[0]

    # Mask the previous alignment and choose only the sequences that
    # DO NOT have "unusual amino acids"
    new_aln_seqs = aln_seqs[sequences_mask, :]

    # Get the amino acid / gap boolean matrix
    new_aln_seqs_isaa = (new_aln_seqs != '-')

    # Count the number of amino acidic residues for each column
    new_aa_per_column = np.sum(new_aln_seqs_isaa, axis=0)

    # Get the index of positions where there are no amino acids
    # (i.e. the column is entirely made of gaps)
    position_mask = np.where(new_aa_per_column > 0)[0]

    # Mask the alignment by deleting fully gapped columns
    final_aln_seqs = new_aln_seqs[:, position_mask]

    # Mask the descriptions
    final_aln_descs = aln_descs[sequences_mask]
    
    if return_gappy_pos:
        return final_aln_descs, final_aln_seqs, gappy_positions
    else:
        return final_aln_descs, final_aln_seqs

def write_to_fasta(aln_descs, aln_seqs, fasta_file):
    """
    Write alignment to a FASTA file.

    Parameters
    ----------
    aln_descs: ndarray
        1D array containing the description of each sequence
    aln_seqs: ndarray
        2D array containing the sequence alignment
    fasta_file: str
        Filename of the FASTA file that we want to write
    """
    with open(fasta_file, 'w') as f:
        for d, s in zip(aln_descs, aln_seqs):
            s = ''.join(s)
            f.write(f'>{d}\n')
            f.write(s)
            f.write('\n\n')