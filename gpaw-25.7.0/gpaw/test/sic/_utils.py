import re

import numpy as np


class MockWorld:
    def __init__(self, rank=0, size=1):
        self.rank = rank
        self.size = size


def extract_lagrange_section(log_output: str) -> str:
    """
    Finds all blocks starting with a line containing the word 'L_ii',
    and captures that line plus all immediately subsequent lines that
    consist only of space-separated floats.

    Args:
        log_output: The full string content.

    Returns:
        A Lagrange elements block.
    """
    float_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    data_line_pattern = r"^\s*" + float_pattern +\
        r"(?:\s+" + float_pattern + r")*\s*\n"

    # Pattern to find the blocks:
    # - Group 1 captures the entire block (trigger + data lines)
    pattern = re.compile(
        # Start Group 1
        r"("
        # 1. Trigger Line: Must contain 'L_ii' as a whole word (\b)
        r"^(.*\bL_ii\b.*?\n)"  # Capture the trigger line itself in Group 2
        # 2. Data Lines: Zero or more (*) lines immediately following
        #    that match the data_line_pattern.
        r"(?:" + data_line_pattern + r")*"
        # End Group 1
        r")",
        re.MULTILINE,
    )

    # Find all non-overlapping matches
    matches = pattern.finditer(log_output)

    # Extract the full captured block (Group 1) from each match
    extracted_blocks = [match.group(1) for match in matches]

    # Generate single string
    ext_log = "".join(extracted_blocks)

    return ext_log


def mk_arr_from_str(log_out: str, row_elems: int = 3,
                    skip_rows: int = 0) -> str:
    if skip_rows > 0:
        ldat = log_out.split("\n")[skip_rows:]
        ldat = "\n".join(ldat)
    else:
        ldat = log_out
    ldat = ldat.split()[row_elems:]
    return np.fromiter(ldat, dtype=float).reshape(-1, row_elems)
