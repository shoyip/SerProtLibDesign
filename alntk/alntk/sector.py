import numpy as np
from .alignment import Alignment, find_sequence, write_to_fasta

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def highlight_sector(aln, accession, sector_indices, labels,
                     width=100, label_every=10, highlight_color="red"):
    """
    labels must contain one label for each alignment column.

    width:
        Number of sequence columns printed per block.

    label_every:
        Print a label every N alignment columns.
    """
    if highlight_color=="red":
        color_statement = color.RED
    elif highlight_color=="green":
        color_statement = color.GREEN
    elif highlight_color=="blue":
        color_statement = color.BLUE
    sequence = find_sequence(aln, accession)
    sector_indices = set(sector_indices.tolist())

    if len(labels) != len(sequence):
        raise ValueError(
            f"Expected {len(sequence)} labels, got {len(labels)}"
        )

    for block_start in range(0, len(sequence), width):
        block_end = min(block_start + width, len(sequence))

        sequence_block = sequence[block_start:block_end]
        labels_block = labels[block_start:block_end]

        # Label line has the same visual width as the sequence line.
        label_line = [" "] * len(sequence_block)

        for position in range(0, len(sequence_block), label_every):
            label = str(labels_block[position])

            # Write the label at this position without exceeding the block.
            for offset, character in enumerate(label):
                if position + offset < len(label_line):
                    label_line[position + offset] = character

        print("".join(label_line))

        for position, residue in enumerate(sequence_block):
            global_index = block_start + position

            if global_index in sector_indices:
                print(
                    color_statement + color.BOLD + residue + color.END,
                    end=""
                )
            else:
                print(residue, end="")

        print("\n")