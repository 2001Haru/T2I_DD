"""Compare baseline and caption-conditioned CoDA timing records."""

import argparse
import json

from experiment_timing import read_timing_record


def _seconds(record, stage):
    return float(record.get("stages_seconds", {}).get(stage, 0.0))


def _percentage(numerator, denominator):
    return None if denominator <= 0 else round(100.0 * numerator / denominator, 2)


def main():
    parser = argparse.ArgumentParser(description="Summarize CoDA timing overhead.")
    parser.add_argument("--baseline", required=True, help="Timing JSON for coda_baseline.")
    parser.add_argument("--caption", required=True, help="Timing JSON for vlm_caption.")
    args = parser.parse_args()

    baseline = read_timing_record(args.baseline)
    caption = read_timing_record(args.caption)
    baseline_generation = _seconds(baseline, "synthetic_generation")
    caption_generation = _seconds(caption, "synthetic_generation")
    captioning = _seconds(caption, "caption_generation")
    baseline_discovery = _seconds(baseline, "feature_extraction") + _seconds(baseline, "clustering")
    caption_discovery = _seconds(caption, "feature_extraction") + _seconds(caption, "clustering")
    caption_reuses_discovery = caption_discovery == 0.0 and baseline_discovery > 0.0
    caption_full_discovery = baseline_discovery if caption_reuses_discovery else caption_discovery
    caption_full_pipeline = (
        caption_full_discovery
        + captioning
        + caption_generation
        + _seconds(caption, "downstream_training")
    )

    summary = {
        "baseline": {
            "distribution_discovery_seconds": round(baseline_discovery, 3),
            "synthetic_generation_seconds": baseline_generation,
            "downstream_training_seconds": _seconds(baseline, "downstream_training"),
            "total_recorded_seconds": baseline.get("total_recorded_seconds", 0.0),
        },
        "vlm_caption": {
            "measured_distribution_discovery_seconds": round(caption_discovery, 3),
            "distribution_discovery_reused_from_baseline": caption_reuses_discovery,
            "full_pipeline_distribution_discovery_seconds": round(caption_full_discovery, 3),
            "caption_generation_seconds": captioning,
            "synthetic_generation_seconds": caption_generation,
            "downstream_training_seconds": _seconds(caption, "downstream_training"),
            "measured_run_seconds": caption.get("total_recorded_seconds", 0.0),
            "full_pipeline_estimate_seconds": round(caption_full_pipeline, 3),
        },
        "cost_metrics": {
            "caption_as_percent_of_baseline_generation": _percentage(captioning, baseline_generation),
            "caption_plus_generation_overhead_percent_vs_baseline_generation": _percentage(
                captioning + caption_generation - baseline_generation, baseline_generation
            ),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
