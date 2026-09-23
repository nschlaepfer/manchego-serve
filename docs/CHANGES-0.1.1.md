# manchego-serve 0.1.1: one temperature per answer type

**Summary.** 0.1.1 changes one thing in the served policy. The option probabilities are now read at a fixed temperature
per question type, fitted for Manchego v2.1 on our own held-out development data:

| type | T |
|---|---:|
| choice | 1.791 |
| noul | 1.73 |
| score | 1.0 (not fitted: no held-out score rows) |

The weights (`oraculumai/Manchego` @ `77403228`, tag `v2.1`), the prompts, the contract (auto), one pass, the option
order, the confidence definition and the limits are all unchanged.

## What changes, what does not

- `p = softmax(z / T[type])` over the offered option-code logits `z`.
- **Changed:** `noul`, `probabilities`, `confidence` and a score's expected level. They are less extreme, so v2.1's
  probabilities are less over-confident on hard, unfamiliar decisions.
- **Unchanged:** the chosen option, so every answer and the accuracy stay the same. `choice` is now read directly from
  the logits (largest logit, first in the client's order on an exact tie). That is the same option v0.1.0 returned, for
  every T.
- **v0.1.0 exactly:** `--temperature-map off`, `--no-temperature-map` or `MANCHEGO_TEMPERATURE_MAP=off`. This gives
  T = 1.0 for every type, the v0.1.0 readout byte for byte. A test checks it against the v0.1.0 golden fixtures with
  `json.dumps` equality.
- **Reported:**
  - `/healthz` and every response carry `temperature` (the map, or 1.0 when off), `temperature_map` (source, SHA-256
    of the file, how it binds to the loaded weights) and `temperature_by_question`.
  - `/healthz` also carries the file's provenance.
  - The map file is `manchego_serve/temperature_map_v2.1.json`, sha256 `2458d21771bbddc9c23ae0e63c880f2ebe5a6aa997e8c9de6b5e67b7210cd07e`, pinned in `temperature.py`.
- **Builds:**
  - The map applies to the published v2.1 bf16 weights, where it was fitted (CUDA bf16), and to the MLX 8-bit build
    (checked on 321 held-out rows: the temperature fitted on its logits is within 2% of the bf16 one).
  - Any other weights get T = 1.0 with a warning, unless a map file is passed explicitly.

## How it was fitted

- **Data.** Development records of v2.1 only, none of which v2.1 was trained on:
  - the Stage-0 calibration draws of our hard-gap screen: generated long policies, multi-hop lookups, judging of worked
    responses, and fresh draws of v2.1's own families;
  - 22 Natural Instructions tasks that v2.1 never saw.
- **Split and objective.** 7,019 rows split 50/50 by instance (by task for the NI rows). One temperature per type was
  chosen to minimise NLL on the first half and evaluated on the second half. The protocol, including the decision rule,
  was registered before the fit.
- **Never used:** any JevBench item, per-item result or record; Public8; our transfer holdout; test and sealed sets;
  model-generated labels.

## Evidence (held-out half, 3,524 rows)

| | T = 1.0 | 0.1.1 map | change [95% bootstrap interval] |
|---|---:|---:|---:|
| accuracy | 0.639 | 0.639 | 0 (T never changes the choice) |
| NLL | 0.832 | 0.767 | −0.065 [−0.082, −0.048] |
| ECE (15 bins) | 0.093 | 0.018 | −0.075 [−0.081, −0.046] |
| Brier | 0.465 | 0.445 | −0.019 [−0.025, −0.013] |
| distance to a one-hot target (TVD) | 0.407 | 0.443 | +0.036 [+0.032, +0.040] |

**What else the data shows:**
- **Where the gain is.** It sits on the hard, unfamiliar components. ECE goes from 0.15 to 0.07 on long policies
  (resolved), 0.12 to 0.08 on judging (not resolved on its own) and 0.15 to 0.08 on multi-hop (resolved).
- **About the intervals.** They are percentile bootstraps over instances (tasks for NI). Because ECE is biased upward
  in resamples, its intervals are skewed around the point estimates. The decision rule used their upper ends.
- **Unseen NI tasks.** On the 22 unseen NI tasks the change is unresolved (NLL 0.540 → 0.516, ECE 0.075 → 0.060).
- **Per type vs one T.** One temperature shared by every type (1.778) does as well as the per-type map; the map adds
  nothing measurable.

**Costs:**
- **Familiar task types.** v2.1 is already well calibrated at T = 1 on the task types it was trained on, and the map
  over-flattens them. On its own development groups (2,920 rows, accuracy 0.885, reported only, never fitted), ECE goes
  from 0.023 to 0.092 and NLL rises by +0.055 [+0.049, +0.061]; under the map every one of them is under-confident. The same holds for
  fresh draws of its own families (choice ECE 0.046 → 0.092).
- **TVD.** Flatter probabilities put less mass on the right answer, so the distance to one-hot targets rises everywhere
  (+0.036 above).

## Disclosure

- The hard-gap components' families were designed from the published JevBench hard-tier specification (families only;
  no JevBench item text). The map is therefore JevBench-hard-informed, in the same sense as the model's training
  families.
- No JevBench item, per-item result or record was used to fit or check it.
- The model card's numbers are at temperature 1.0.

## Draft follow-up for JevBench #56 (for the account holder to post)

> **manchego-serve v0.1.1 (served-policy change, same weights).** One temperature per answer type (choice 1.791,
> noul 1.73, score 1.0), applied to the option logits before the softmax.
> - **How it was fitted.** Only on our own held-out development data: generated hard-decision components and 22 Natural
>   Instructions tasks the model never saw. The fit protocol was registered in advance. No JevBench item, per-item
>   result or record, and no other benchmark's test item, was used to fit or check it.
> - **What it changes.** A temperature never changes which option is chosen, so answers and accuracy are identical to
>   v0.1.0; only the probabilities and `confidence` change. On our held-out half, ECE goes from 0.093 to 0.018 and NLL
>   from 0.832 to 0.767. On task types the model was trained on, it becomes under-confident (ECE 0.023 → 0.092 there).
> - **Reproducing it.** `/healthz` reports the map and its SHA-256. `--temperature-map off` reproduces v0.1.0 byte for
>   byte. Weights, prompts, one pass and the run recipe are unchanged.
> - **Disclosure.** Our hard components' families were designed from the published hard-tier specification (families
>   only, no item text), so the map is JevBench-hard-informed in the same sense as the model.
