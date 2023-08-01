#!/bin/bash
# to organise results by date and folders

DATE="01-08-23"
SLUG="data_experiments_svm_undersampling_010823"
ANNOTATION="parents"
INPUT_FOLDER="parents_svm_undersampling"

OUTPUT_FOLDER="results/${SLUG}/parents"
mkdir -p ${OUTPUT_FOLDER}

# Normalised data using MinMAxScaler()
# Features included:
# audio: all
# video: AU, gaze, head_movement
# SVM model
# Annotation: parents


#MULTIMODAL
# Late Fusion
# All participants separately - Video&Audio: features (above) - Late fusion
echo "Starting new Data Experiment Configuration: participant_01_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/participant_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_01_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_02_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/participant_02_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_02_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_03_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_03_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_03_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_04_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_04_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_04_va_late_fusion_${DATE}.txt
#
#
## Each sessions separately - Video&Audio: features (above) - Late fusion
echo "Starting new Data Experiment Configuration: session_01_01_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/session_01_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_01_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/session_02_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/session_02_02_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_02_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_03_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_03_02_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_02_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_04_01_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_va_late_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_04_02_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_02_va_late_fusion_${DATE}.txt
#
#
#
## Early Fusion
## All participants separately - Video&Audio: features (above) - Early fusion
echo "Starting new Data Experiment Configuration: participant_01_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_01_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_02_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_03_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_03_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/participant_04_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_04_va_early_fusion_${DATE}.txt

# Each sessions separately - Video&Audio: all features - Early fusion
echo "Starting new Data Experiment Configuration: session_01_01_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/session_01_01_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_01_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/session_02_01_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/session_02_02_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_va_early_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_03_01_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_va_early_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_03_02_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_va_early_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_04_01_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_va_early_fusion.json"
python entry_point.py ${INPUT_FOLDER}/session_04_02_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_02_va_early_fusion_${DATE}.txt
#
#

## UNIMODAL
## Unimodality
## VIDEO
## Each participants separately - Video:  features (above) - SVM
echo "Starting new Data Experiment Configuration: participant_01_v.json"
python entry_point.py ${INPUT_FOLDER}/participant_01_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_v.json"
python entry_point.py ${INPUT_FOLDER}/participant_02_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_v.json"
python entry_point.py ${INPUT_FOLDER}/participant_03_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_03_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_v.json"
python entry_point.py ${INPUT_FOLDER}/participant_04_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_04_v_${DATE}.txt

# Each session separately - Video:  features (above) - SVM
echo "Starting new Data Experiment Configuration: session_01_01_v.json"
python entry_point.py ${INPUT_FOLDER}/session_01_01_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_01_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_v.json"
python entry_point.py ${INPUT_FOLDER}/session_02_01_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_v.json"
python entry_point.py ${INPUT_FOLDER}/session_02_02_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_v.json"
python entry_point.py ${INPUT_FOLDER}/session_03_01_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_v.json"
python entry_point.py ${INPUT_FOLDER}/session_03_02_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_v.json"
python entry_point.py ${INPUT_FOLDER}/session_04_01_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_v.json"
python entry_point.py ${INPUT_FOLDER}/session_04_02_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_02_v_${DATE}.txt
#
## AUDIO
## Each participants separately - Audio:  all - SVM
echo "Starting new Data Experiment Configuration: participant_01_a.json"
python entry_point.py ${INPUT_FOLDER}/participant_01_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_a.json"
python entry_point.py ${INPUT_FOLDER}/participant_02_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_a.json"
python entry_point.py ${INPUT_FOLDER}/participant_03_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_03_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_a.json"
python entry_point.py ${INPUT_FOLDER}/participant_04_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_participant_04_a_${DATE}.txt

# Each session separately - Audio:  features all - SVM
echo "Starting new Data Experiment Configuration: session_01_01_a.json"
python entry_point.py ${INPUT_FOLDER}/session_01_01_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_01_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_a.json"
python entry_point.py ${INPUT_FOLDER}/session_02_01_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_a.json"
python entry_point.py ${INPUT_FOLDER}/session_02_02_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_02_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_a.json"
python entry_point.py ${INPUT_FOLDER}/session_03_01_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_a.json"
python entry_point.py ${INPUT_FOLDER}/session_03_02_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_03_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_a.json"
python entry_point.py ${INPUT_FOLDER}/session_04_01_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_a.json"
python entry_point.py ${INPUT_FOLDER}/session_04_02_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_session_04_02_a_${DATE}.txt


# All data VIDEO
echo "Starting new Data Experiment Configuration: all_data_v.json"
python entry_point.py ${INPUT_FOLDER}/all_data_v.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_v_${DATE}.txt
#
## All data AUDIO
echo "Starting new Data Experiment Configuration: all_data_a.json"
python entry_point.py ${INPUT_FOLDER}/all_data_a.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_a_${DATE}.txt

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py ${INPUT_FOLDER}/all_data_va_late_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_late_fusion_${DATE}.txt
#
## All participants together - Video&Audio:  features (above) - Early fusion
echo "Starting new Data Experiment Configuration: all_data_va_early_fusion"
python entry_point.py ${INPUT_FOLDER}/all_data_va_early_fusion.json> ${OUTPUT_FOLDER}/${SLUG}_${ANNOTATION}_all_data_va_early_fusion_${DATE}.txt
#
