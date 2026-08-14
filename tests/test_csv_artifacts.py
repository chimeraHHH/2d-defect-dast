import numpy as np
import pytest

from scripts.prm_build_protocol import write_sample_table
from scripts.prm_collect_factorial import write_csv as write_factorial_csv
from scripts.prm_materials_analysis import write_csv as write_materials_csv
from scripts.prm_uq_analysis import write_csv as write_uq_csv


CSV_WRITERS = (
    lambda path, rows: write_factorial_csv(path, rows, ["split", "mae"]),
    write_materials_csv,
    write_uq_csv,
)


@pytest.mark.parametrize("writer", CSV_WRITERS)
def test_result_csv_writers_use_lf_line_endings(tmp_path, writer):
    output = tmp_path / "result.csv"

    writer(output, [{"split": "id_cv5_f0", "mae": 0.5}])

    assert output.read_bytes() == b"split,mae\nid_cv5_f0,0.5\n"


def test_protocol_sample_table_uses_lf_line_endings(tmp_path):
    output = tmp_path / "samples.csv"
    sample = {
        "id": 1,
        "numbers": np.asarray([6, 8]),
        "target": 0.5,
        "metadata": {"host": "C", "dopant": "O"},
    }

    write_sample_table(output, [sample], {0}, {})

    contents = output.read_bytes()
    assert b"\r\n" not in contents
    assert contents.endswith(b"\n")
