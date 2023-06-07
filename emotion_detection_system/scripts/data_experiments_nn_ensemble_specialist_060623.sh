#!/bin/bash
# to organise results by date and folders

DATE="06-06-23"
SLUG="data_experiments_nn_ensemble_060623"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_nn_ensemble"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}


# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# NN model
# Annotation: Parents
# Dataset balanced: over-sampling:random
# 'NN': MLPClassifier(verbose=True, random_state=1, max_iter=500, alpha=1e-9),
# PARAMETER_GRID_SEARCH = {
 # 'model__random_state': [0, 1, 2, 3, 4, 5]


#MULTIMODAL

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_late_fusion_${DATE}.txt
#
