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

#MULTIMODAL

echo "Starting new Data Experiment Configuration: participant_03_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_03_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_03_va_late_fusion_${DATE}.txt


echo "Starting new Data Experiment Configuration: session_03_01_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_03_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_01_va_late_fusion_${DATE}.txt

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_late_fusion_${DATE}.txt
#
