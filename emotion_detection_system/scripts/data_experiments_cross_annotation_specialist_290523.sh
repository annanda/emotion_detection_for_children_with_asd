#!/bin/bash
# to organise results by date and folders

DATE="29-05-23"
SLUG="data_experiments_cross_annotation_290523"
ANNOTATION="specialist"
INPUT_FOLDER="specialist_cross_annotation"

OUTPUT_FOLDER="results/${SLUG}/specialist"
mkdir -p ${OUTPUT_FOLDER}

####### WORKING WITH JUST MULTIMODAL MODELS ##############
# I want to look at how the best general, high specific and highly-specific models perform when applied to the same
# models annotated by parents.

# FOR SPECIALIST ANNOTATION
# Best GM: all_data_va_late_fusion_specialist
# Best SM: participant_03_va_late_fusion_specialist
# Best HSM: session_03_01_va_late_fusion_specialist and session_02_01_va_late_fusion_specialist

# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# SVM model
# Annotation: Parents
# Dataset balanced: over-sampling:random


echo "Starting new Data Experiment Configuration: GM_all_data_va_late_fusion_specialist"
python entry_point.py ${INPUT_FOLDER}/GM_all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_GM_all_data_va_late_fusion_${DATE}.txt


echo "Starting new Data Experiment Configuration: SM_participant_03_va_late_fusion_specialist"
python entry_point.py ${INPUT_FOLDER}/SM_participant_03_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_SM_participant_03_va_late_fusion_${DATE}.txt


echo "Starting new Data Experiment Configuration: HSM_session_03_01_va_late_fusion_specialist"
python entry_point.py ${INPUT_FOLDER}/HSM_session_03_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_HSM_session_03_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: HSM_session_02_01_va_late_fusion_specialist"
python entry_point.py ${INPUT_FOLDER}/HSM_session_02_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_HSM_session_02_01_va_late_fusion_${DATE}.txt
