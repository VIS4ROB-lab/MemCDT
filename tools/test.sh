# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Single-GPU testing
#
# bash test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
#
# Optional arguments:
# --work-dir ${WORK_DIR}: Override the working directory specified in the
# config file. the evaluation metric results will be dumped into the directory
# as json.
# --out ${RESULT_FILE}: Filename of the output results in pickle format. If not
# specified, the results will not be saved to a file. (After mmseg v0.17,
# the output results become pre-evaluation results or format result paths).
# --eval ${EVAL_METRICS}: Items to be evaluated on the results. Allowed values 
# depend on the dataset, e.g., mIoU is available for all dataset. Cityscapes
# could be evaluated by cityscapes as well as standard mIoU metrics.
# --show: If specified, segmentation results will be plotted on the images and
# shown in a new window. It is only applicable to single GPU testing and used
# for debugging and visualization. Please make sure that GUI is available in
# your environment, otherwise you may encounter the error like cannot connect
# to X server.
# --show-dir: If specified, segmentation results will be plotted on the images
# and saved to the specified directory. It is only applicable to single GPU
# testing and used for debugging and visualization. You do NOT need a GUI
# available in your environment for using this option.
# --eval-options: Optional parameters for dataset.format_results and
# dataset.evaluate during evaluation. When efficient_test=True, it will save
# intermediate results to local files to save CPU memory. Make sure that you
# have enough local storage space (more than 20GB). (efficient_test argument
# does not have effect after mmseg v0.17, we use a progressive mode to
# evaluation and format results which can largely save memory cost and
# evaluation time.)
# --opacity: Opacity of painted segmentation map in (0, 1] range. If not
# specified, it is set to 0.5 by default.


CONFIG=$1
CHECKPOINTS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH mim test mmsegmentation \
    $CONFIG \
    --checkpoint $CHECKPOINTS \
    ${@:3}