# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Single-GPU training
#
# bash train.sh ${CONFIG_FILE} [optional arguments]
#
# Optional arguments:
# --no-validate (not suggested): By default, the codebase will perform
# evaluation at every k iterations during the training. To disable this
# behavior, use --no-validate.
# --work-dir ${WORK_DIR}: Override the working directory specified in the
# config file.
# --resume-from ${CHECKPOINT_FILE}: Resume from a previous checkpoint file (to
# continue the training process).
# --load-from ${CHECKPOINT_FILE}: Load weights from a checkpoint file (to start
# finetuning for another task).
# --seed: Random seed.
# --deterministic: Switch on “deterministic” mode which slows down training but
# the results are reproducible.
#
# Difference between resume-from and load-from:
# resume-from loads both the model weights and optimizer state including the
# iteration number.
# load-from loads only the model weights, starts the training from iteration 0.


CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH mim train mmsegmentation \
    $CONFIG \
    ${@:2}