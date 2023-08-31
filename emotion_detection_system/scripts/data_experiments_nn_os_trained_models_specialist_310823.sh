#!/bin/bash
# to organise results by date and folders

DATE="31-08-23"
SLUG="data_experiments_nn_os_trained_models_310823"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_nn_os_trained_models"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}

####### WORKING WITH JUST MULTIMODAL MODELS ##############
# I want to look at how the best high specific and highly-specific models perform when applied to all data.
# Focusing on parents annotation.

# FOR SPECIALIST ANNOTATION
# Best SM: participant_03_va_late_fusion_specialist (best acc and balanced acc)
# Best HSM: session_04_01_va_late_fusion_specialist (best acc)

# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# NN model
# Annotation: Specialist
# Dataset balanced: over-sampling:random


# ALL DATA
echo "Starting new Data Experiment Configuration: SM_all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/SM_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_SM_all_data_va_late_fusion_${DATE}.txt


echo "Starting new Data Experiment Configuration: HSM_all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/HSM_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_HSM_all_data_va_late_fusion_${DATE}.txt
