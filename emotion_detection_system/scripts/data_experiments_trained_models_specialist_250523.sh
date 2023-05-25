#!/bin/bash
# to organise results by date and folders

DATE="25-05-23"
SLUG="data_experiments_trained_models_250523"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_trained_models"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}

####### WORKING WITH JUST MULTIMODAL MODELS ##############
# I want to look at how the best high specific and highly-specific models perform when applied to all data.
# Focusing on parents annotation.

# FOR SPECIALIST ANNOTATION
# Best SM: participant_03_va_late_fusion_specialist
# Best HSM: session_03_01_va_late_fusion_specialist and session_02_01_va_late_fusion_specialist

# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# SVM model
# Annotation: Parents
# Dataset balanced: over-sampling:random


# ALL DATA
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/SM_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_SM_all_data_va_late_fusion_${DATE}.txt


echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/HSM_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_HSM_all_data_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/HSM_02_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_HSM_02_all_data_va_late_fusion_${DATE}.txt
