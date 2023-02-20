#!/bin/bash
# to organise results by date and folders

DATE="20-02-23"
mkdir -p "results/${DATE}"

# Normalised data using linear normalisation (values vary from 0 to 1)
# Features included:
# audio: all
# video: AU, gaze, head_movement
# SVM model & Late Fusion
# Annotation: parents


#MULTIMODAL
# Late Fusion
# All participants separately - Video&Audio: features (above) - Late fusion
echo "Starting new Data Experiment Configuration: participant_01_va_late_fusion.json"
python entry_point.py participant_01_va_late_fusion.json> results/${DATE}/participant_01_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_02_va_late_fusion.json"
 python entry_point.py participant_02_va_late_fusion.json> results/${DATE}/participant_02_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_03_va_late_fusion"
python entry_point.py participant_03_va_late_fusion.json> results/${DATE}/participant_03_va_late_fusion_${DATE}.txt
echo "Starting new Data Experiment Configuration: participant_04_va_late_fusion"
python entry_point.py participant_04_va_late_fusion.json> results/${DATE}/participant_04_va_late_fusion_${DATE}.txt
#
#
## Each sessions separately - Video&Audio: features (above) - Late fusion
echo "Starting new Data Experiment Configuration: session_01_01_va_late_fusion"
python entry_point.py session_01_01_va_late_fusion.json> results/${DATE}/session_01_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_va_late_fusion"
python entry_point.py session_02_01_va_late_fusion.json> results/${DATE}/session_02_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_va_late_fusion"
python entry_point.py session_02_02_va_late_fusion.json> results/${DATE}/session_02_02_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_va_late_fusion.json"
python entry_point.py session_03_01_va_late_fusion.json> results/${DATE}/session_03_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_va_late_fusion.json"
python entry_point.py session_03_02_va_late_fusion.json> results/${DATE}/session_03_02_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_va_late_fusion.json"
python entry_point.py session_04_01_va_late_fusion.json> results/${DATE}/session_04_01_va_late_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_va_late_fusion.json"
python entry_point.py session_04_02_va_late_fusion.json> results/${DATE}/session_04_02_va_late_fusion_${DATE}.txt
#
#
#
## Early Fusion
## All participants separately - Video&Audio: features (above) - Early fusion
echo "Starting new Data Experiment Configuration: participant_01_va_early_fusion"
python entry_point.py participant_01_va_early_fusion.json> results/${DATE}/participant_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_va_early_fusion"
python entry_point.py participant_02_va_early_fusion.json> results/${DATE}/participant_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_va_early_fusion"
python entry_point.py participant_03_va_early_fusion.json> results/${DATE}/participant_03_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_va_early_fusion"
python entry_point.py participant_04_va_early_fusion.json> results/${DATE}/participant_04_va_early_fusion_${DATE}.txt

# Each sessions separately - Video&Audio: all features - Early fusion
echo "Starting new Data Experiment Configuration: session_01_01_va_early_fusion"
python entry_point.py session_01_01_va_early_fusion.json> results/${DATE}/session_01_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_va_early_fusion"
python entry_point.py session_02_01_va_early_fusion.json> results/${DATE}/session_02_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_va_early_fusion"
python entry_point.py session_02_02_va_early_fusion.json> results/${DATE}/session_02_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_va_early_fusion.json"
python entry_point.py session_03_01_va_early_fusion.json> results/${DATE}/session_03_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_va_early_fusion.json"
python entry_point.py session_03_02_va_early_fusion.json> results/${DATE}/session_03_02_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_va_early_fusion.json"
python entry_point.py session_04_01_va_early_fusion.json> results/${DATE}/session_04_01_va_early_fusion_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_va_early_fusion.json"
python entry_point.py session_04_02_va_early_fusion.json> results/${DATE}/session_04_02_va_early_fusion_${DATE}.txt
#
#

## UNIMODAL
## Unimodality
## VIDEO
## Each participants separately - Video:  features (above) - SVM
echo "Starting new Data Experiment Configuration: participant_01_v.json"
python entry_point.py participant_01_v.json> results/${DATE}/participant_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_v.json"
python entry_point.py participant_02_v.json> results/${DATE}/participant_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_v.json"
python entry_point.py participant_03_v.json> results/${DATE}/participant_03_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_v.json"
python entry_point.py participant_04_v.json> results/${DATE}/participant_04_v_${DATE}.txt

# Each session separately - Video:  features (above) - SVM
echo "Starting new Data Experiment Configuration: session_01_01_v.json"
python entry_point.py session_01_01_v.json> results/${DATE}/session_01_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_v.json"
python entry_point.py session_02_01_v.json> results/${DATE}/session_02_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_v.json"
python entry_point.py session_02_02_v.json> results/${DATE}/session_02_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_v.json"
python entry_point.py session_03_01_v.json> results/${DATE}/session_03_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_v.json"
python entry_point.py session_03_02_v.json> results/${DATE}/session_03_02_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_v.json"
python entry_point.py session_04_01_v.json> results/${DATE}/session_04_01_v_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_v.json"
python entry_point.py session_04_02_v.json> results/${DATE}/session_04_02_v_${DATE}.txt
#
## AUDIO
## Each participants separately - Audio:  all - SVM
echo "Starting new Data Experiment Configuration: participant_01_a.json"
python entry_point.py participant_01_a.json> results/${DATE}/participant_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_02_a.json"
python entry_point.py participant_02_a.json> results/${DATE}/participant_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_03_a.json"
python entry_point.py participant_03_a.json> results/${DATE}/participant_03_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: participant_04_a.json"
python entry_point.py participant_04_a.json> results/${DATE}/participant_04_a_${DATE}.txt

# Each session separately - Audio:  features all - SVM
echo "Starting new Data Experiment Configuration: session_01_01_a.json"
python entry_point.py session_01_01_a.json> results/${DATE}/session_01_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_01_a.json"
python entry_point.py session_02_01_a.json> results/${DATE}/session_02_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_02_02_a.json"
python entry_point.py session_02_02_a.json> results/${DATE}/session_02_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_01_a.json"
python entry_point.py session_03_01_a.json> results/${DATE}/session_03_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_03_02_a.json"
python entry_point.py session_03_02_a.json> results/${DATE}/session_03_02_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_01_a.json"
python entry_point.py session_04_01_a.json> results/${DATE}/session_04_01_a_${DATE}.txt

echo "Starting new Data Experiment Configuration: session_04_02_a.json"
python entry_point.py session_04_02_a.json> results/${DATE}/session_04_02_a_${DATE}.txt


# All data VIDEO
echo "Starting new Data Experiment Configuration: all_data_v.json"
python entry_point.py all_data_v.json> results/${DATE}/all_data_v_${DATE}.txt
#
## All data AUDIO
echo "Starting new Data Experiment Configuration: all_data_a.json"
python entry_point.py all_data_a.json> results/${DATE}/all_data_a_${DATE}.txt

## All participants together - Video&Audio:  features (above) - Late fusion
echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
python entry_point.py all_data_va_late_fusion.json> results/${DATE}/all_data_va_late_fusion_${DATE}.txt
#
## All participants together - Video&Audio:  features (above) - Early fusion
echo "Starting new Data Experiment Configuration: all_data_va_early_fusion"
python entry_point.py all_data_va_early_fusion.json> results/${DATE}/all_data_va_early_fusion_${DATE}.txt
#