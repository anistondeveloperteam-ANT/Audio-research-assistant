"""Chunk-type classification must require STRUCTURAL evidence (a real grid / equation / pseudocode),
not topic keywords. Previously ~1/3 of all chunks were mis-tagged 'table_or_metrics' just for
containing words like 'snr' or 'dataset', which made the retrieval chunk-type boost misfire."""
from backend.ingestion.document_chunker import (
    has_table, has_equation, has_algorithm, classify_chunk,
)

_PROSE = ("We evaluate speech enhancement using PESQ, STOI and SDR on the VCTK dataset, and compare "
          "against the SNR baseline during training and inference; the procedure is described above.")


# ---- tables: a grid or numeric block, NOT a metric word in prose ----
def test_prose_with_metric_words_is_not_a_table():
    assert has_table(_PROSE) == 0
    assert classify_chunk(_PROSE) == "text"


def test_markdown_grid_is_a_table():
    grid = "|Model|PESQ|STOI|\n|---|---|---|\n|A|2.9|0.93|\n|B|3.1|0.95|"
    assert has_table(grid) == 1
    assert classify_chunk(grid) == "table_or_metrics"


def test_numeric_block_is_a_table():
    block = "Model A 2.85 0.91 12.3 45\nModel B 2.91 0.93 11.8 50\nModel C 3.02 0.95 10.5 55"
    assert has_table(block) == 1


# ---- equations: numbered tag or LaTeX, NOT a bare '=' ----
def test_bare_equals_in_prose_is_not_an_equation():
    assert has_equation("The total loss L = a + b balances the two terms.") == 0


def test_numbered_equation_tag_is_an_equation():
    assert has_equation("y = Wx + b   (3)") == 1


def test_latex_markers_are_an_equation():
    assert has_equation("the objective \\sum_{i} \\frac{1}{n} x_i") == 1
    assert classify_chunk("minimise \\sum_i (y_i - \\hat{y}_i)^2  argmin") == "equation"


# ---- algorithms: a header / pseudocode, NOT 'training'/'inference' prose ----
def test_training_prose_is_not_an_algorithm():
    assert has_algorithm("During training and inference we follow the standard procedure.") == 0


def test_algorithm_header_is_an_algorithm():
    assert has_algorithm("Algorithm 1 Beam search decoding") == 1
    assert classify_chunk("Algorithm 2: iterate until convergence") == "algorithm"


def test_input_output_pseudocode_is_an_algorithm():
    assert has_algorithm("Input: audio x\nOutput: mask m\ncompute STFT then apply mask") == 1


# ---- priority + plain text ----
def test_plain_prose_is_text():
    assert classify_chunk("This paragraph simply explains the motivation behind the approach.") == "text"
