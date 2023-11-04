#!/bin/bash
# to organise results by date and folders

DATE="04-11-23"
SLUG="data_experiments_nn_algorithm_bl_non_tuned_041123"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_nn_algorithm_bl"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}


# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# NN model
# Annotation: Parents
# Dataset balanced: false

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_late_fusion_${DATE}.txt
