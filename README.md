# train_deploy.py: MLP / MoE commands

Replace `<subtype>` with `ER`, `PR` or `Ki67`. For one pooled model use `ER PR Ki67` in `--subtypes` and `ER-PR-Ki67` in the file names. Run from the folder with `train_deploy.py` and `config.py`. Each command uses a fixed 200 epochs, the cosine learning-rate schedule and no early stopping, and writes its own log. External-test numbers are the last `acc=` and `slide-mean acc by subtype` lines of the log.

Needs this line at the end of `config.py` (for the `_no_ctx` feature set):

```python
FEATURE_SETS["all_derived_no_dab_od_no_ctx"] = [f for f in FEATURE_SETS["all_derived_no_dab_od"] if not f.startswith("Smoothed:")]
```

## What each part of a command does

| part | meaning |
|---|---|
| `--arch mlp` / `--arch moe` | model type (see headings below) |
| `--feature_set X` | which input features the model may use (table below) |
| `--seed 42/43/44` | repeats with a different seed; the seed also changes which 15% of training slides are held out for validation, so average the three |
| `--epochs 200 --lr_schedule cosine` | fixed training length; learning rate follows a cosine curve and does not depend on the validation slides (the plateau schedule starved training in one run) |
| no `--early_stop` | the final epoch is saved, not the best-validation epoch (the 2-slide Ki67 holdout is too noisy to select on) |
| `--out ....pt` / `tee log_....txt` | model checkpoint and full training log |

## Feature sets

| feature set | features | what it tests |
|---|---|---|
| `all_derived` | 108 | Baseline: every feature, including DAB, OD Sum, the 50/75 um neighbourhood (Smoothed) features and Aspect ratio. |
| `all_derived_no_dab` | 79 | Baseline minus the 29 DAB features (nucleus and neighbourhood). Tests whether DAB helps or confuses the model. |
| `all_derived_no_dab_od` | 50 | Also removes the 29 OD Sum features, which still contain the DAB signal. Cleanest 'no brown information' set. |
| `all_derived_no_dab_od_no_ctx` | 26 | As above, also without the neighbourhood (Smoothed) features: nucleus-level shape, size and hematoxylin only. |
| `no_context_derived` | 64 | Keeps DAB and OD Sum but drops the neighbourhood (Smoothed) features. Tests the context-feature shortcut on its own. |

## MLP

MLP: one network on all selected features (hidden layers 96-48-24-12-6, dropout 0.5).

```bash
# all_derived (108 features) - seeds 42, 43, 44
python train_deploy.py --out mlp_<subtype>_all_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_s42.txt
python train_deploy.py --out mlp_<subtype>_all_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_s43.txt
python train_deploy.py --out mlp_<subtype>_all_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_s44.txt
# all_derived_no_dab (79 features) - seeds 42, 43, 44
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_s42.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_s43.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_s44.txt
# all_derived_no_dab_od (50 features) - seeds 42, 43, 44
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_s42.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_s43.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_s44.txt
# all_derived_no_dab_od_no_ctx (26 features) - seeds 42, 43, 44
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_no_ctx_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od_no_ctx --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_no_ctx_s42.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_no_ctx_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od_no_ctx --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_no_ctx_s43.txt
python train_deploy.py --out mlp_<subtype>_all_derived_no_dab_od_no_ctx_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set all_derived_no_dab_od_no_ctx --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_all_derived_no_dab_od_no_ctx_s44.txt
# no_context_derived (64 features) - seeds 42, 43, 44
python train_deploy.py --out mlp_<subtype>_no_context_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set no_context_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_no_context_derived_s42.txt
python train_deploy.py --out mlp_<subtype>_no_context_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set no_context_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_no_context_derived_s43.txt
python train_deploy.py --out mlp_<subtype>_no_context_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set no_context_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_no_context_derived_s44.txt
```

## MOE

MoE: mixture of experts, one expert per feature group (see moe.py for the exact groups, e.g. shape, size, intensity, texture), combined by a gate.

```bash
# all_derived (108 features) - seeds 42, 43, 44
python train_deploy.py --out moe_<subtype>_all_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_s44.txt
# all_derived_no_dab (79 features) - seeds 42, 43, 44
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_s44.txt
# all_derived_no_dab_od (50 features) - seeds 42, 43, 44
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_s44.txt
# all_derived_no_dab_od_no_ctx (26 features) - seeds 42, 43, 44
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_no_ctx_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od_no_ctx --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_no_ctx_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_no_ctx_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od_no_ctx --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_no_ctx_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_no_dab_od_no_ctx_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived_no_dab_od_no_ctx --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_no_dab_od_no_ctx_s44.txt
# no_context_derived (64 features) - seeds 42, 43, 44
python train_deploy.py --out moe_<subtype>_no_context_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set no_context_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_no_context_derived_s42.txt
python train_deploy.py --out moe_<subtype>_no_context_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set no_context_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_no_context_derived_s43.txt
python train_deploy.py --out moe_<subtype>_no_context_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set no_context_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_no_context_derived_s44.txt
```

## Extra experiments (not yet in the MoE doc, or only run pooled there)

The MoE doc already covers: all 108, no context, context only (MLP), no DAB, no DAB+OD (MLP and MoE), the gate variants, and removing one expert at a time. These commands add what it is missing, per stain (`<subtype>`).

### 1. Context only (MoE; the doc has the MLP only). Also repeats MLP context-only with this protocol

```bash
# context_only_derived - mlp
python train_deploy.py --out mlp_<subtype>_context_only_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set context_only_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_context_only_derived_s42.txt
python train_deploy.py --out mlp_<subtype>_context_only_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set context_only_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_context_only_derived_s43.txt
python train_deploy.py --out mlp_<subtype>_context_only_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch mlp --feature_set context_only_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_mlp_context_only_derived_s44.txt
# context_only_derived - moe
python train_deploy.py --out moe_<subtype>_context_only_derived_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set context_only_derived --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_context_only_derived_s42.txt
python train_deploy.py --out moe_<subtype>_context_only_derived_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set context_only_derived --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_context_only_derived_s43.txt
python train_deploy.py --out moe_<subtype>_context_only_derived_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set context_only_derived --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_context_only_derived_s44.txt
```

### 2. MoE gate variants (doc ran them pooled; this runs them for one stain)

```bash
# gate learned
python train_deploy.py --out moe_<subtype>_all_derived_learned_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate learned --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_learned_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_learned_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate learned --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_learned_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_learned_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate learned --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_learned_s44.txt
# gate global
python train_deploy.py --out moe_<subtype>_all_derived_global_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate global --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_global_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_global_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate global --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_global_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_global_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate global --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_global_s44.txt
# gate per_stain
python train_deploy.py --out moe_<subtype>_all_derived_per_stain_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate per_stain --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_per_stain_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_per_stain_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate per_stain --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_per_stain_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_per_stain_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate per_stain --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_per_stain_s44.txt
# gate vote
python train_deploy.py --out moe_<subtype>_all_derived_vote_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate vote --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_vote_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_vote_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate vote --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_vote_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_vote_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate vote --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_vote_s44.txt
# gate prob_mix
python train_deploy.py --out moe_<subtype>_all_derived_prob_mix_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate prob_mix --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_prob_mix_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_prob_mix_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate prob_mix --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_prob_mix_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_prob_mix_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_gate prob_mix --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_prob_mix_s44.txt
```
(`uniform` is the default MoE command above.)

### 3. Remove hematoxylin and nuclear morphology experts together (the doc lists this as untested)

Group names are printed at the top of an MoE log; replace `<hema group>` and `<morphology group>` with the exact names shown there.

```bash
python train_deploy.py --out moe_<subtype>_all_derived_drop_hema_morph_s42.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_drop_groups <hema group> <morphology group> --seed 42 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_drop_hema_morph_s42.txt
python train_deploy.py --out moe_<subtype>_all_derived_drop_hema_morph_s43.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_drop_groups <hema group> <morphology group> --seed 43 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_drop_hema_morph_s43.txt
python train_deploy.py --out moe_<subtype>_all_derived_drop_hema_morph_s44.pt --cache_dir ./cache --subtypes <subtype> --arch moe --feature_set all_derived --moe_drop_groups <hema group> <morphology group> --seed 44 --epochs 200 --lr_schedule cosine 2>&1 | tee log_<subtype>_moe_all_derived_drop_hema_morph_s44.txt
```

### 4. Not runnable yet (needs a code change)

- Per-slide DAB/OD normalisation (standardise each slide's DAB and OD columns by that slide's own mean and SD instead of the global training mean). `train_deploy.py` has no flag for this; ask me to add `--slide_norm dab` when you want it. Test it on Ki67 first, using the all_derived and no_dab results above as the reference.
- Pooled vs per-stain Ki67: pooled needs `--subtypes ER PR Ki67` and `ER-PR-Ki67` in the file names; then compare the Ki67 slide-mean line against the per-stain Ki67 runs.

## Summary line per log

Prints, for every log, the external-test accuracy and the slide-mean accuracy per stain (the last matches in a log are the external test; the validation block comes first).

```bash
for f in log_*.txt; do echo "$f $(grep -E '^acc=' $f | tail -1) $(grep 'slide-mean acc by subtype' $f | tail -1)"; done
```

## Results so far (Ki67 MLP, external test, slide-mean accuracy, mean of 3 seeds)

| feature set | slide-mean acc |
|---|---|
| all_derived (108) | 0.832 |
| all_derived_no_dab (79) | 0.826 |
| all_derived_no_dab_od (50) | 0.822 |

The differences are about the size of the seed-to-seed spread (0.4-0.6 points). The no_dab_od_no_ctx and no_context_derived sets have not been run yet.

## Optional extra feature sets

`all`, `shape_derived`, `shape_ctx_derived`, `intensity`, `intensity_ctx`, `context_only_derived`: replace the value after `--feature_set`.

MoE-only options: `--moe_gate learned|uniform|global|per_stain|vote|prob_mix` and `--moe_drop_groups <group names>` (group names are printed at the top of an MoE log).
