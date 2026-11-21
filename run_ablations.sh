#!/usr/bin/env bash
# MoE ablation plan. Every run uses the SAME protocol:
#   * epochs/checkpoint chosen on held-out TRAINING slides (--val_source holdout),
#     never on the reference slides (Windeatt 2008: don't tune on the set you report);
#   * the reference slides are scored once afterwards, as "EXTERNAL TEST";
#   * 3 seeds per configuration (the seed also changes which training slides are
#     held out, so the spread covers initialization AND split).
#
# Ghost class: by default (GHOST=1) every model is trained WITH Ghost as a third class
# (--classes Tumor "Non Tumor" Ghost) and logs are named g3_<config>_s<seed>.log, so they sit
# next to the earlier two-class logs. A nucleus predicted Ghost is treated as rejected; the
# 'forced' accuracy (Tumor vs Non Tumor, Ghost output ignored) is the one comparable with the
# two-class runs. GHOST=0 reproduces the original two-class runs and log names.
#
# Usage:
#   ./run_ablations.sh            # all phases
#   ./run_ablations.sh 1          # phase 1 only (combiner ladder)
#   ./run_ablations.sh 2 3        # phases 2 and 3
#   ./run_ablations.sh 4          # phase 4 only: the MLP variants
#   SEEDS="0 1 2 3 4" OUT=runs_v2 ./run_ablations.sh 1
#   EXTRA="--device cuda:1" ./run_ablations.sh
#   ONLY="mlp" SEEDS="0 1 2 3 4" bash run_ablations.sh 1   # just the plain MLP, 5 seeds
#
# Finished runs (log ends with "Saved ...") are skipped, so re-running resumes.
# Then:  python rank_results.py runs      (default metric = forced)
set -u
PY=${PY:-python}
OUT=${OUT:-runs}
SEEDS=${SEEDS:-"0 1 2"}
EXTRA=${EXTRA:-}
ONLY=${ONLY:-}   # space-separated config names to run, e.g. ONLY="mlp"; empty = all
PHASES=${*:-"1 2 3 4"}
COMMON="--feature_set all_derived --cache_dir ./cache --val_source holdout --holdout_frac 0.15 --early_stop --no_tb"
GHOST=${GHOST:-1}
if [ "$GHOST" = "1" ]; then CLASSES=(--classes Tumor "Non Tumor" Ghost); PREFIX=g3_; else CLASSES=(); PREFIX=; fi
MOE="--arch moe"
mkdir -p "$OUT"

run() {  # run <name> <args...>
  local base=$1
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $base "* ]]; then shift; return; fi
  local name=${PREFIX}$1; shift
  for s in $SEEDS; do
    local log="$OUT/${name}_s${s}.log"
    if [ -f "$log" ] && grep -q "^Saved " "$log"; then
      echo "skip  ${name}_s${s} (done)"; continue
    fi
    echo "run   ${name}_s${s}"
    $PY train_deploy.py $COMMON ${CLASSES[@]+"${CLASSES[@]}"} "$@" --seed "$s" --out "$OUT/${name}_s${s}.pt" $EXTRA > "$log" 2>&1 \
      || echo "FAILED ${name}_s${s} - see $log"
  done
}

for p in $PHASES; do case $p in
1)  # ---- Phase 1: combiner ladder (all with the default training extras)
    run mlp            --arch mlp
    run moe_vote       $MOE --moe_gate vote        # fixed combiner over expert decisions (Windeatt Fig. 1)
    run moe_uniform    $MOE --moe_gate uniform     # equal weights on expert embeddings
    run moe_global     $MOE --moe_gate global      # one learned weight per expert
    run moe_per_stain  $MOE --moe_gate per_stain   # one learned weight per expert per stain
    run moe_learned    $MOE --moe_gate learned     # per-nucleus gate (current MoE)
    run moe_nostain    $MOE --moe_gate learned --moe_no_subtype_gate
    run moe_prob_mix   $MOE --moe_gate prob_mix    # learned per-nucleus weights on expert probabilities
    ;;
2)  # ---- Phase 2: training extras, one at a time, on the learned gate.
    # Reference = phase-1 moe_learned (aux on, balance on, expert dropout on,
    # stain dropout on, bagging off, domain-adversarial off).
    run moe_learned_no_aux       $MOE --moe_aux_weight 0
    run moe_learned_no_balance   $MOE --moe_balance_weight 0
    run moe_learned_no_expdrop   $MOE --moe_expert_dropout 0
    run moe_learned_no_staindrop $MOE --moe_subtype_drop 0
    run moe_learned_bag          $MOE --moe_bag
    run moe_learned_dann         $MOE --domain_adversarial
    run moe_learned_all_off      $MOE --moe_aux_weight 0 --moe_balance_weight 0 \
                                      --moe_expert_dropout 0 --moe_subtype_drop 0
    ;;
3)  # ---- Phase 3: domain-adversarial on the plain MLP (site shift, not architecture)
    run mlp_dann --arch mlp --domain_adversarial
    ;;
4)  # ---- Phase 4: the MLP variants (feature ablations, DANN strength, one MLP per stain).
    # Same flags as the earlier two-class runs; --feature_set later on the line overrides COMMON's.
    run mlp_dann_l3      --arch mlp --domain_adversarial --domain_lambda_max 3
    run mlp_no_dab_any   --arch mlp --feature_set all_derived_no_dab
    run mlp_no_dab_od    --arch mlp --feature_set all_derived_no_dab_od
    run mlp_no_context   --arch mlp --feature_set no_context_derived
    run mlp_context_only --arch mlp --feature_set context_only_derived
    run mlp_only_ER      --arch mlp --subtypes ER
    run mlp_only_PR      --arch mlp --subtypes PR
    run mlp_only_Ki67    --arch mlp --subtypes Ki67
    ;;
*)  echo "unknown phase $p (use 1, 2, 3, 4)"; exit 1 ;;
esac; done
echo "done - summarize with:  $PY rank_results.py $OUT"