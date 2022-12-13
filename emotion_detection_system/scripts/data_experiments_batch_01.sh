#!/bin/bash
# Normalised data using linear normalisation (values vary from 0 to 1)
# Features included:
# audio: all
# video: AU, gaze, head_movement
# SVM model & Late Fusion

#MULTIMODAL
# Late Fusion
# All participants separately - Video&Audio: features (above) - Late fusion
#echo "Starting new Data Experiment Configuration: participant_01_va_late_fusion.json"
#python entry_point.py participant_01_va_late_fusion.json> results/participant_01_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration: participant_02_va_late_fusion.json"
# python entry_point.py participant_02_va_late_fusion.json> results/participant_02_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration: participant_03_va_late_fusion"
#python entry_point.py participant_03_va_late_fusion.json> results/participant_03_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration: participant_04_va_late_fusion"
#python entry_point.py participant_04_va_late_fusion.json> results/participant_04_va_late_fusion.txt
#
#
## Each sessions separately - Video&Audio: features (above) - Late fusion
#echo "Starting new Data Experiment Configuration: session_01_01_va_late_fusion"
#python entry_point.py session_01_01_va_late_fusion.json> results/session_01_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_01_va_late_fusion"
#python entry_point.py session_02_01_va_late_fusion.json> results/session_02_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_02_va_late_fusion"
#python entry_point.py session_02_02_va_late_fusion.json> results/session_02_02_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_01_va_late_fusion.json"
#python entry_point.py session_03_01_va_late_fusion.json> results/session_03_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_02_va_late_fusion.json"
#python entry_point.py session_03_02_va_late_fusion.json> results/session_03_02_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_01_va_late_fusion.json"
#python entry_point.py session_04_01_va_late_fusion.json> results/session_04_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_02_va_late_fusion.json"
#python entry_point.py session_04_02_va_late_fusion.json> results/session_04_02_va_late_fusion.txt
#
#
#
## Early Fusion
## All participants separately - Video&Audio: features (above) - Early fusion
#echo "Starting new Data Experiment Configuration: participant_01_va_early_fusion"
#python entry_point.py participant_01_va_early_fusion.json> results/participant_01_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: participant_02_va_early_fusion"
#python entry_point.py participant_02_va_early_fusion.json> results/participant_02_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: participant_03_va_early_fusion"
#python entry_point.py participant_03_va_early_fusion.json> results/participant_03_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: participant_04_va_early_fusion"
#python entry_point.py participant_04_va_early_fusion.json> results/participant_04_va_early_fusion.txt
#
## Each sessions separately - Video&Audio: all features - Early fusion
#echo "Starting new Data Experiment Configuration: session_01_01_va_early_fusion"
#python entry_point.py session_01_01_va_early_fusion.json> results/session_01_01_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_01_va_early_fusion"
#python entry_point.py session_02_01_va_early_fusion.json> results/session_02_01_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_02_va_early_fusion"
#python entry_point.py session_02_02_va_early_fusion.json> results/session_02_02_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_01_va_early_fusion.json"
#python entry_point.py session_03_01_va_early_fusion.json> results/session_03_01_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_02_va_early_fusion.json"
#python entry_point.py session_03_02_va_early_fusion.json> results/session_03_02_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_01_va_early_fusion.json"
#python entry_point.py session_04_01_va_early_fusion.json> results/session_04_01_va_early_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_02_va_early_fusion.json"
#python entry_point.py session_04_02_va_early_fusion.json> results/session_04_02_va_early_fusion.txt
#
#
#
## All participants together - Video&Audio:  features (above) - Late fusion
#echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
#python entry_point.py all_data_va_late_fusion.json> results/all_data_va_late_fusion.txt
#
## All participants together - Video&Audio:  features (above) - Early fusion
#echo "Starting new Data Experiment Configuration: all_data_va_early_fusion"
#python entry_point.py all_data_va_early_fusion.json> results/all_data_va_early_fusion.txt
#
## UNIMODAL
## Unimodality
## VIDEO
## Each participants separately - Video:  features (above) - SVM
#echo "Starting new Data Experiment Configuration: participant_01_v.json"
#python entry_point.py participant_01_v.json> results/participant_01_v.txt
#
#echo "Starting new Data Experiment Configuration: participant_02_v.json"
#python entry_point.py participant_02_v.json> results/participant_02_v.txt
#
#echo "Starting new Data Experiment Configuration: participant_03_v.json"
#python entry_point.py participant_03_v.json> results/participant_03_v.txt
#
#echo "Starting new Data Experiment Configuration: participant_04_v.json"
#python entry_point.py participant_04_v.json> results/participant_04_v.txt

# Each session separately - Video:  features (above) - SVM
echo "Starting new Data Experiment Configuration: session_01_01_v.json"
python entry_point.py session_01_01_v.json> results/session_01_01_v.txt

echo "Starting new Data Experiment Configuration: session_02_01_v.json"
python entry_point.py session_02_01_v.json> results/session_02_01_v.txt

echo "Starting new Data Experiment Configuration: session_02_02_v.json"
python entry_point.py session_02_02_v.json> results/session_02_02_v.txt

echo "Starting new Data Experiment Configuration: session_03_01_v.json"
python entry_point.py session_03_01_v.json> results/session_03_01_v.txt

echo "Starting new Data Experiment Configuration: session_03_02_v.json"
python entry_point.py session_03_02_v.json> results/session_03_02_v.txt

echo "Starting new Data Experiment Configuration: session_04_01_v.json"
python entry_point.py session_04_01_v.json> results/session_04_01_v.txt

echo "Starting new Data Experiment Configuration: session_04_02_v.json"
python entry_point.py session_04_02_v.json> results/session_04_02_v.txt

# AUDIO
# Each participants separately - Audio:  all - SVM
echo "Starting new Data Experiment Configuration: participant_01_a.json"
python entry_point.py participant_01_a.json> results/participant_01_a.txt

echo "Starting new Data Experiment Configuration: participant_02_a.json"
python entry_point.py participant_02_a.json> results/participant_02_a.txt

echo "Starting new Data Experiment Configuration: participant_03_a.json"
python entry_point.py participant_03_a.json> results/participant_03_a.txt

echo "Starting new Data Experiment Configuration: participant_04_a.json"
python entry_point.py participant_04_a.json> results/participant_04_a.txt

# Each session separately - Audio:  features all - SVM
echo "Starting new Data Experiment Configuration: session_01_01_a.json"
python entry_point.py session_01_01_a.json> results/session_01_01_a.txt

echo "Starting new Data Experiment Configuration: session_02_01_a.json"
python entry_point.py session_02_01_a.json> results/session_02_01_a.txt

echo "Starting new Data Experiment Configuration: session_02_02_a.json"
python entry_point.py session_02_02_a.json> results/session_02_02_a.txt

echo "Starting new Data Experiment Configuration: session_03_01_a.json"
python entry_point.py session_03_01_a.json> results/session_03_01_a.txt

echo "Starting new Data Experiment Configuration: session_03_02_a.json"
python entry_point.py session_03_02_a.json> results/session_03_02_a.txt

echo "Starting new Data Experiment Configuration: session_04_01_a.json"
python entry_point.py session_04_01_a.json> results/session_04_01_a.txt

echo "Starting new Data Experiment Configuration: session_04_02_a.json"
python entry_point.py session_04_02_a.json> results/session_04_02_a.txt
