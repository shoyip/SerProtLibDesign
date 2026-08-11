import numpy as np
from numpy.lib.stride_tricks import as_strided
from Bio import SeqIO
import json

class Alignment:
    def __init__(self):
        np.random.seed(42)

    def import_from_fasta(self, fasta_file):
        """
        Import an array of descriptions and an array of sequences from a FASTA file.
        Save them as attributes of the Alignment instance.
    
        Parameters
        ----------
        fasta_file: str
            filename of FASTA file, typically `.fasta`, `.faa` or `.fa`.
        """
        seqs = []
        descs = []
        
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seqs.append(str(record.seq))
            descs.append(record.description.split('/')[0])
    
        self.descs_arr = np.array(descs)
        self.seqs_arr = np.array([[residue for residue in seq] for seq in seqs])
        self.seq_idxs0 = np.arange(0, self.seqs_arr.shape[0])
        self.pos_idxs0 = np.arange(0, self.seqs_arr.shape[1])
        self.seq_idxs = self.seq_idxs0
        self.pos_idxs = self.pos_idxs0
        self.seq_untrim_idxs = self.seq_idxs0

    def subsample(self, n_subsample):
        """
        Randomly choose a subsample for faster computation of statistics.
        Sequences are not deleted, just the list of indices `seq_idxs` is updated.
        Generator is seeded at 42 for reproducibility.

        Parameters
        ----------
        n_subsample: int
            number of sequences that we want to subsample from the original alignment
        """
        self.seq_idxs = np.random.randint(0, self.seqs_arr.shape[0], size=n_subsample)

    def drop(self, del_idxs):
        aln_seqs = self.get_seqs()
        drop_seq_trim_idxs = np.where(np.sum(aln_seqs[:, del_idxs[1]] != '-', axis=1) > 0)[0]
        self.seq_untrim_idxs = np.delete(self.seq_untrim_idxs, drop_seq_trim_idxs)
        self.seq_idxs = np.delete(self.seq_idxs, del_idxs[0])
        self.pos_idxs = np.delete(self.pos_idxs, del_idxs[1])

    def reset_seq(self):
        self.seq_idxs = self.seq_idxs0

    def reset_pos(self):
        self.pos_idxs = self.pos_idxs0
        self.seq_untrim_idxs = self.seq_idxs0

    def get_seqs(self):
        return self.seqs_arr[self.seq_idxs, :][:, self.pos_idxs]

    def get_descs(self):
        return self.descs_arr[self.seq_idxs]

    def print_report(self, trimmed=False):
        print(f"Number of sequences: {len(self.seq_idxs)}")
        print(f"Number of positions: {len(self.pos_idxs)}")
        if trimmed:
            print(f"Number of untrimmed sequences: {len(self.seq_untrim_idxs)}")

    def get_seq_gap(self):
        """
        """
        aln_seqs = self.get_seqs()
        gaps_per_seq = np.sum(aln_seqs == '-', axis=1) / aln_seqs.shape[1]
    
        return gaps_per_seq
    
    def get_pos_gap(self):
        """
        """
        aln_seqs = self.get_seqs()
        gaps_per_pos = np.sum(aln_seqs == '-', axis=0) / aln_seqs.shape[0]
    
        return gaps_per_pos

    def update_descs(self, descs_tbu):
        """
        Update descriptions according to a deduplication map.
    
        Parameters
        ----------
        descs_tbu : dict
            Mapping: representative_description -> [other_description, ...]
            where the representative is the "first" sequence in a group of
            identical sequences, and the others are the ones that will be dropped.
    
        This method:
          - For each (rep_desc, others) in descs_tbu:
              - Finds the sequence whose description == rep_desc
              - Sets its description to "_".join([rep_desc] + others)
          - Does NOT delete any sequences or modify seq_idxs / pos_idxs.
            Use your existing drop() method for that.
    
        Assumes the class has:
          - self.descs : list of description strings (mutable)
          - self.get_descs() returning that list (or something compatible)
        """
        aln_descs = self.descs_arr  # direct reference; we will modify in place
    
        # Build a map description -> index (assume unique at call time)
        desc_to_idx = {d: i for i, d in enumerate(aln_descs)}
    
        for rep_desc, other_descs in descs_tbu.items():
            if rep_desc not in desc_to_idx:
                # Representative not found; skip or raise, depending on your needs
                continue
    
            idx = desc_to_idx[rep_desc]
            merged_desc = "_".join([rep_desc] + other_descs)
            aln_descs[idx] = merged_desc

    def annotate_duplicate_descs(self):
        """
        For duplicate descriptions, append /0, /1, /2, ... in order.
        Modifies self.descs_arr in place.
        """
        import numpy as np
    
        descs = list(self.descs_arr)
        n = len(descs)
        if n == 0:
            return
    
        order = np.argsort(descs)
        sorted_descs = [descs[i] for i in order]
    
        new_descs = [None] * n
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_descs[end] == sorted_descs[start]:
                end += 1
            # indices in original order for this group
            group_idx = [order[i] for i in range(start, end)]
            base = sorted_descs[start]
            for i, orig in enumerate(group_idx):
                new_descs[orig] = f"{base}/{i}"
            start = end
    
        if isinstance(self.descs_arr, np.ndarray):
            self.descs_arr[:] = np.array(new_descs, dtype=object)
        else:
            self.descs_arr[:] = new_descs

    def strip_uniref90_prefix(self):
        """
        For each description, split by '_' and keep only the first element.
        This removes prefixes like 'Uniref90_' from descriptions.
        Modifies self.descs_arr in place.
        """
        descs = list(self.descs_arr)
        new_descs = [d.split("_")[1] for d in descs]
    
        if isinstance(self.descs_arr, np.ndarray):
            self.descs_arr[:] = np.array(new_descs, dtype=object)
        else:
            self.descs_arr[:] = new_descs

def filter_gappy(aln, threshold=0.8):
    """
    """
    aln_seqs = aln.get_seqs()

    col_tbd = np.where(np.sum(aln_seqs == '-', axis=0) / aln_seqs.shape[0] > threshold)
    return [], col_tbd

def filter_length(aln, min_len, max_len):
    """
    """
    aln_seqs = aln.get_seqs()
    
    cond_min = np.sum(aln_seqs != '-', axis=1) > min_len
    cond_max = np.sum(aln_seqs != '-', axis=1) < max_len
    seq_tbd = np.where(cond_min & cond_max == False)[0]
    return seq_tbd, []

def find_pattern(text, pattern):
    """
    """
    pattern = np.asarray(list(pattern))
    
    if len(pattern) > len(text):
        return np.array([], dtype=int)
    
    # Create sliding window view of text
    windows = np.lib.stride_tricks.as_strided(
        text, 
        shape=(len(text) - len(pattern) + 1, len(pattern)),
        strides=(text.strides[0], text.strides[0])
    )
    
    # Find indices where windows match the pattern
    match_indices = np.where(np.all(windows == pattern, axis=1))[0]
    
    return match_indices

def find_sequence(aln, accession):
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    mask = np.array([accession in desc for desc in aln_descs])
    #ref_idx = np.where(aln_descs == accession)[0][0]
    #return aln_seqs[ref_idx]
    return aln_seqs[np.where(mask)[0][0]]

def filter_residue(aln, ref_seq_acc, ref_pattern, residue):
    """
    """
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    ref_idx = np.where(aln_descs == ref_seq_acc)[0][0]
    ref_seq = aln_seqs[ref_idx]
    ref_pattern_idxs = find_pattern(ref_seq, ref_pattern)

    # delete sequences that do not have a specific residue in the first position of the pattern
    seq_tbd = np.where(aln_seqs[:, ref_pattern_idxs[0]] != residue)[0]

    return seq_tbd, []

def filter_residues_any(aln, ref_seq_acc, ref_pattern):
    """
    """
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    ref_idx = np.where(aln_descs == ref_seq_acc)[0][0]
    ref_seq = aln_seqs[ref_idx]
    ref_pattern_idxs = find_pattern(ref_seq, ref_pattern)

    # delete sequences that have a gap in the columns corresponding
    # to the first occurence of the reference pattern in the reference sequence
    seq_tbd = np.where(np.sum(aln_seqs[:, ref_pattern_idxs[0]:ref_pattern_idxs[0]+len(ref_pattern)] == '-', axis=1) > 0)[0]
    pos_tbd = np.arange(0, ref_pattern_idxs[0])

    return seq_tbd, pos_tbd

def filter_idseqs(aln, export_map=None):
    """
    Identify identical sequences and prepare:
      - seq_tbd: 1D array of sequence indices to drop
      - pos_tbd: empty array (for compatibility)
      - descs_tbu: dict mapping representative_description -> [other_description, ...]
    """
    import numpy as np
    import json

    aln_seqs = aln.get_seqs()
    aln_descs = np.array(aln.get_descs())
    n_seq = aln_seqs.shape[0]

    if n_seq == 0:
        if export_map is not None:
            with open(export_map, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        return [np.array([], dtype=int), np.array([], dtype=int)], {}

    # Lexicographic sort of rows
    order = np.lexsort(aln_seqs.T[::-1])
    sorted_seqs = aln_seqs[order]
    sorted_descs = aln_descs[order]

    # Identify where adjacent rows differ
    diff = ~np.all(sorted_seqs[:-1] == sorted_seqs[1:], axis=1)

    group_starts = np.concatenate(([0], np.where(diff)[0] + 1))
    group_ends = np.concatenate((group_starts[1:], [n_seq]))

    drop_list = []
    descs_tbu = {}

    for start, end in zip(group_starts, group_ends):
        group_idx = order[start:end]
        group_descs = sorted_descs[start:end]

        if end - start > 1:
            drop_list.extend(group_idx[1:].tolist())

            rep_desc = str(group_descs[0])
            other_descs = [str(d) for d in group_descs[1:]]
            descs_tbu[rep_desc] = other_descs

    seq_tbd = np.array(drop_list, dtype=int) if drop_list else np.array([], dtype=int)
    pos_tbd = np.array([], dtype=int)

    # Do NOT do: seq_tbd = np.concatenate(seq_tbd)

    if export_map is not None:
        with open(export_map, "w", encoding="utf-8") as f:
            json.dump(descs_tbu, f, ensure_ascii=False, indent=2)

    return [seq_tbd, pos_tbd], descs_tbu


def filter_iddescs(aln, export_map=None):
    """
    Group sequences by identical description strings.

    Parameters
    ----------
    aln : alignment object
        Must implement .get_seqs() and .get_descs().
    export_map : str or None
        If not None, path to a JSON file where a mapping from the
        representative sequence (as a string) to the list of other
        sequences (as strings) in the same description group will be written.

    Returns
    -------
    seq_tbd : list of np.ndarray
        For each group of sequences with the same description, an array of
        indices of all duplicates except the first one.
    pos_tbd : np.ndarray
        Empty array, kept for compatibility with filter_residues_any signature.
    """
    aln_seqs = aln.get_seqs()
    aln_descs = np.array(aln.get_descs())
    n_seq = aln_descs.shape[0]

    if n_seq == 0:
        if export_map is not None:
            with open(export_map, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        return [], np.array([], dtype=int)

    # Lexicographic sort of descriptions
    order = np.argsort(aln_descs)
    sorted_descs = aln_descs[order]
    sorted_seqs = aln_seqs[order]

    # Identify where adjacent descriptions differ
    diff = sorted_descs[:-1] != sorted_descs[1:]

    group_starts = np.concatenate(([0], np.where(diff)[0] + 1))
    group_ends = np.concatenate((group_starts[1:], [n_seq]))

    seq_tbd = []
    seq_map = {}

    for start, end in zip(group_starts, group_ends):
        group_idx = order[start:end]
        if end - start > 1:
            seq_tbd.append(group_idx[1:])

        # Representative sequence and others in this description group
        rep_seq = "".join(sorted_seqs[start])
        other_seqs = ["".join(s) for s in sorted_seqs[start + 1:end]]
        if other_seqs:
            seq_map[rep_seq] = other_seqs

    if export_map is not None:
        with open(export_map, "w", encoding="utf-8") as f:
            json.dump(seq_map, f, ensure_ascii=False, indent=2)

    seq_tbd = np.concatenate(seq_tbd) if seq_tbd else np.array([], dtype=int)

    pos_tbd = np.array([], dtype=int)
    return seq_tbd, pos_tbd

def find_sequence_old(aln, accession):
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    ref_idx = np.where(np.array([e.split('|')[2] for e in aln_descs]) == accession)[0][0]

    return aln_seqs[ref_idx]

def filter_residue_old(aln, ref_seq_acc, ref_pattern, residue):
    """
    """
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    ref_idx = np.where(np.array([e.split('|')[2] for e in aln_descs]) == ref_seq_acc)[0][0]
    ref_seq = aln_seqs[ref_idx]
    ref_pattern_idxs = find_pattern(ref_seq, ref_pattern)

    # delete sequences that do not have a specific residue in the first position of the pattern
    seq_tbd = np.where(aln_seqs[:, ref_pattern_idxs[0]] != residue)[0]

    return seq_tbd, []

def filter_residues_any_old(aln, ref_seq_acc, ref_pattern):
    """
    """
    aln_seqs = aln.get_seqs()
    aln_descs = aln.get_descs()

    ref_idx = np.where(np.array([e.split('|')[2] for e in aln_descs]) == ref_seq_acc)[0][0]
    ref_seq = aln_seqs[ref_idx]
    ref_pattern_idxs = find_pattern(ref_seq, ref_pattern)

    # delete sequences that have a gap in the columns corresponding
    # to the first occurence of the reference pattern in the reference sequence
    seq_tbd = np.where(np.sum(aln_seqs[:, ref_pattern_idxs[0]:ref_pattern_idxs[0]+len(ref_pattern)] == '-', axis=1) > 0)[0]
    pos_tbd = np.arange(0, ref_pattern_idxs[0])

    return seq_tbd, pos_tbd

def filter_ambiguous(aln):
    """
    """
    aln_seqs = aln.get_seqs()
    seq_tbd = np.where(np.sum((aln_seqs == 'B') + (aln_seqs == 'J') + (aln_seqs == 'X') + (aln_seqs == 'Z'), axis=1) > 0)[0]
    return seq_tbd, []

def filter_compact(aln, gap_threshold_ratio=0.95):
    """
    Compactify the alignment by deleting sequences and consequently
    deleting entirely gapped columns.
    """
    aln_seqs = aln.get_seqs()

    # Get the gap threshold in terms of integer number of gaps from the ratio
    gap_threshold = int(gap_threshold_ratio*aln_seqs.shape[0])

    # Get the gap / no gap boolean matrix
    aln_seqs_isgap = (aln_seqs == '-')
    
    # Count the number of gaps in each column
    gaps_per_column = np.sum(aln_seqs_isgap, axis=0)

    # Get the index of all the columns that have more gaps than the threshold
    # We call these "positions to be deleted"
    pos_tbd = np.where(gaps_per_column > gap_threshold)[0]

    # Count how many gaps do sequences have in "gappy positions"
    # We call these counts of "unusual amino acids"
    # The sequences that have these amino acids are "sequences to be deleted"
    unusual_aa_per_sequence = np.sum(aln_seqs[:, pos_tbd] != '-', axis=1)
    seq_tbd = np.where(unusual_aa_per_sequence > 0)[0]

    return seq_tbd, pos_tbd 

def write_to_fasta(aln, fasta_file):
    """
    Write alignment to a FASTA file.
    """
    aln_descs = aln.get_descs()
    aln_seqs = aln.get_seqs()
    with open(fasta_file, 'w') as f:
        for d, s in zip(aln_descs, aln_seqs):
            s = ''.join(s)
            f.write(f'>{d}\n')
            f.write(s)
            f.write('\n\n')