# Visual x Text Factorial on ImageNette

This experiment asks whether cluster-specific text contributes information
independently of a VLCP visual prototype, or only through an interaction with
prototype initialization. It uses the frozen SD 1.5 checkpoint and reuses the
fixed IPC-10 prototype and DCS JSON files from the prior-alignment pilot. It
does not run LLaVA, rebuild captions, or perform diffusion fine-tuning.

## Conditions

The six cells are:

| Visual condition | Text condition | Output name |
| --- | --- | --- |
| no visual information | class label | `no_visual_label` |
| no visual information | correct cluster DCS | `no_visual_dcs` |
| no visual information | same-class shuffled DCS | `no_visual_dcs_shuffled` |
| cluster prototype | class label | `prototype_label` |
| cluster prototype | correct cluster DCS | `prototype_dcs` |
| cluster prototype | same-class shuffled DCS | `prototype_dcs_shuffled` |

The shuffled control is a fixed cyclic derangement within each class:
prototype `i` receives DCS caption `(i + 1) mod 10`. It preserves class,
caption style, and caption length while breaking cluster correspondence.

## Matched visual intervention

The no-visual cell is not a separate full-schedule text-to-image pipeline.
Both visual cells use the same `StableDiffusionImg2ImgPipeline`, strength,
timestep schedule, and image-level random seed:

```text
no_visual: q(z_t | z_0 = 0)
prototype: q(z_t | z_0 = cluster_prototype)
```

Resetting the generator to the same image seed gives the two cells the same
sampled epsilon. This isolates the information in the prototype without
confounding it with a different number of denoising steps.

## Primary estimands

Within each visual condition:

```text
cluster correspondence effect = accuracy(correct DCS) - accuracy(shuffled DCS)
descriptive prompt effect = accuracy(shuffled DCS) - accuracy(label)
```

The primary interaction is:

```text
(correct DCS - shuffled DCS)_prototype
-
(correct DCS - shuffled DCS)_no_visual
```

`correct DCS > shuffled DCS` is evidence that cluster-caption correspondence,
not merely descriptive language, contributes useful information.

## Run

The default source directory reuses artifacts from
`../vlcp_ablation_runs/author_checkpoint_pilot_v0`. Override
`SOURCE_RUN_ROOT`, `PROTOTYPE_PATH`, or `DCS_PATH` when needed.

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
SOURCE_RUN_ROOT=/linxi/T2I_DD/vlcp_ablation_runs/author_checkpoint_pilot_v0 \
RUN_ID=visual_text_factorial_v0 \
GENERATION_SEEDS="0 1" \
bash experiments/visual_text_factorial/run_experiment.sh
```

Resume the exact configuration with:

```bash
source ../vlcp_factorial_runs/visual_text_factorial_v0/resume.env
bash experiments/visual_text_factorial/run_experiment.sh
```

Generation, evaluation, and summary can be controlled independently:

```bash
GENERATE=false EVALUATE=true SUMMARIZE=true \
bash experiments/visual_text_factorial/run_experiment.sh
```

Outputs are isolated under:

```text
/linxi/T2I_DD/vlcp_factorial_runs/<RUN_ID>/
  run_config.txt
  synthetic/seed_<N>/<condition>/
  evaluation/seed_<N>/<condition>.log
  summary/conditions.csv
  summary/contrasts.csv
  summary/summary.json
```

Every synthetic condition stores a manifest containing model and artifact
identities, exact prompts, prompt-source cluster indices, initialization
definition, and image seeds. Resume refuses configuration mismatches.

Audit a completed run before interpreting an unexpected result:

```bash
python experiments/visual_text_factorial/audit_run.py \
  --run-root /linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0
```

The audit verifies all six manifests, paired image seeds, correct and shuffled
prompt-source indices, DCS text hashes, image counts, completed classifier
logs, and the synthetic directory reported by each evaluation log. It writes
`audit.json` in the run root.
