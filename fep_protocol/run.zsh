#!/usr/bin/env zsh

: ${FEPSUITE_ROOT:?Set FEPSUITE_ROOT to the FEPsuite installation}
: ${GROMACS_DIR:?Set GROMACS_DIR to the patched GROMACS installation}
export FEPSUITE_ROOT GROMACS_DIR
export JOBTYPE=feprest
export JOBSYSTEM=${JOBSYSTEM:-none}

# The controller invokes this wrapper again with RUN and stage variables set.
if [[ -n ${RUN:-} ]]; then
    source "$FEPSUITE_ROOT/controller.zsh" "$0"
    exit $?
fi

if [[ $# -lt 1 || ! -d $1 ]]; then
    print -u2 "Usage: $0 CALCULATION_DIR [JOB_SYSTEM=none] [FINAL_STAGE=8]"
    exit 1
fi
workdir=${1:A}
export JOBSYSTEM=${2:-none}
FINAL_STEP=${3:-8}
if [[ $FINAL_STEP != <1-> ]]; then
    print -u2 "FINAL_STAGE must be a positive integer"
    exit 1
fi

last_step=0
if [[ -f "$workdir/done_step.txt" ]]; then
    last_step=$(awk -F, '$1 ~ /^[0-9]+$/ {step=$1} END {print step+0}' "$workdir/done_step.txt")
fi
if (( last_step >= FINAL_STEP )); then
    print "All requested stages are complete"
    exit 0
fi
remaining_steps=($(seq $((last_step + 1)) $FINAL_STEP))
source "$FEPSUITE_ROOT/controller.zsh" "$0" "$workdir" "${remaining_steps[@]}"
