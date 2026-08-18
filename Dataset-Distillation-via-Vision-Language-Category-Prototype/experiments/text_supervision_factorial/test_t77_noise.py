import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_t77_noise import (  # noqa: E402
    EXPECTED_BUDGETS,
    fixed_contrasts,
    load_fixed,
    load_sparse,
    nested_moments,
    sparse_array,
)


def write_log(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Best, last acc:----{values}\n", encoding="utf-8")


def test_nested_moments_separates_fixed_budget_and_three_noise_levels():
    values = []
    for budget in range(8):
        budget_values = []
        for bank_effect in (-4.0, 4.0):
            bank_values = []
            for generation_effect in (-2.0, 2.0):
                bank_values.append([
                    50 + budget + bank_effect + generation_effect + classifier_effect
                    for classifier_effect in (-1.0, 1.0)
                ])
            budget_values.append(bank_values)
        values.append(budget_values)
    result = nested_moments(values)
    assert result["variances"]["bank_checkpoint"] == 28.0
    assert result["variances"]["generation"] == 7.0
    assert result["variances"]["classifier"] == 2.0


def test_sparse_and_dense_indexes_preserve_pairing():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sparse_rows = []
        for budget in EXPECTED_BUDGETS:
            for bank_seed in (0, 1):
                for generation_seed in (0, 1):
                    for prompt, gain in (("label", 0), ("bank_t77", 1)):
                        log = root / f"m{budget}_b{bank_seed}_g{generation_seed}_{prompt}.log"
                        write_log(log, [70 + gain, 72 + gain])
                        sparse_rows.append({
                            "budget": budget, "bank_seed": bank_seed,
                            "generation_seed": generation_seed, "prompt": prompt,
                            "evaluation_log": str(log),
                        })
        sparse_index = root / "sparse.json"
        sparse_index.write_text(json.dumps(sparse_rows), encoding="utf-8")
        records, budgets, banks, generations, _ = load_sparse([sparse_index])
        paired = sparse_array(records, budgets, banks, generations, "bank_t77_minus_label")
        assert all(value == 1 for budget in paired for bank in budget for gen in bank for value in gen)

        fixed_rows = []
        for training_seed in (0, 1):
            for generation_seed in (0, 1):
                for family, family_gain in (("matched_ft", 2), ("unpaired_ft", 0)):
                    for prompt, prompt_gain in (("correct_t77", 1), ("shuffled_t77", 0)):
                        log = root / f"s{training_seed}_g{generation_seed}_{family}_{prompt}.log"
                        write_log(log, [60 + family_gain + prompt_gain, 62 + family_gain + prompt_gain])
                        fixed_rows.append({
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "checkpoint_family": family, "prompt": prompt,
                            "evaluation_log": str(log),
                        })
        fixed_index = root / "fixed.json"
        fixed_index.write_text(json.dumps(fixed_rows), encoding="utf-8")
        fixed, training, generations, _ = load_fixed(fixed_index)
        rows = fixed_contrasts(fixed, training, generations, samples=50, seed=7)
        matched_unpaired = [row for row in rows if row["contrast"] == "matched_minus_unpaired"]
        within_family = [row for row in rows if row["contrast"] == "correct_minus_shuffled"]
        interaction = next(
            row for row in rows if row["contrast"] == "matching_specific_interaction"
        )
        assert all(row["mean_difference"] == 2 for row in matched_unpaired)
        assert all(row["mean_difference"] == 1 for row in within_family)
        assert interaction["mean_difference"] == 0
