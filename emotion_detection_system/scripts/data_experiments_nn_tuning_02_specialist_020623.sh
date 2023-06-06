#!/bin/bash
# to organise results by date and folders

DATE="02-06-23"
SLUG="data_experiments_nn_tuning_020623"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_nn_tuning"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}


# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# NN model
# Annotation: Parents
# Dataset balanced: over-sampling:random
# MLP(verbose=True, random_state=1)
#'model__activation': ['relu', 'logistic'],
    # 'model__solver': ['adam', 'lbfgs'],
    # 'model__learning_rate': ['adaptive', 'constant'],
 #   'model__max_iter': 500
    # 'random_state': [0, 1, 2, 3]
    # 'model__hidden_layer_sizes': [(100,), (50,), (25,)]


#MULTIMODAL

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion_02.json"
python entry_point.py ${INPUT_FOLDER}/all_data_va_late_fusion_02.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_late_fusion_02_${DATE}.txt
#
