# tests/test_pigt_logic.py
from scripts.test_pigt_sanity import test_counterfactual_safety_retreats

def test_counterfactual_logic():
    # this will run on CPU in CI; keep num_plays small
    test_counterfactual_safety_retreats(
        num_plays=4,
        retreat_threshold=2.0,
        min_pass_ratio=0.5,
        device="cpu",
    )
