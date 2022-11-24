#!/bin/bash
# All features included
# SVM model & Late Fusion

# All participants separately - Video&Audio: all features - Late fusion
#echo "Starting new Data Experiment Configuration"
#python entry_point.py participant_01_va_late_fusion.json> results/participant_01_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration"
# python entry_point.py participant_02_va_late_fusion.json> results/participant_02_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration"
#python entry_point.py participant_03_va_late_fusion.json> results/participant_03_va_late_fusion.txt
#echo "Starting new Data Experiment Configuration"
#python entry_point.py participant_04_va_late_fusion.json> results/participant_04_va_late_fusion.txt


# All sessions separately - Video&Audio: all features - Late fusion
#echo "Starting new Data Experiment Configuration: session_01_01_va_late_fusion"
#python entry_point.py session_01_01_va_late_fusion.json> results/session_01_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_01_va_late_fusion"
#python entry_point.py session_02_01_va_late_fusion.json> results/session_02_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_02_02_va_late_fusion"
#python entry_point.py session_02_02_va_late_fusion.json> results/session_02_02_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_01_va_late_fusion"
#python entry_point.py session_03_01_va_late_fusion.json> results/session_03_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_03_02_va_late_fusion"
#python entry_point.py session_03_02_va_late_fusion.json> results/session_03_02_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_01_va_late_fusion"
#python entry_point.py session_04_01_va_late_fusion.json> results/session_04_01_va_late_fusion.txt
#
#echo "Starting new Data Experiment Configuration: session_04_02_va_late_fusion"
#python entry_point.py session_04_02_va_late_fusion.json> results/session_04_02_va_late_fusion.txt

# All participants together - Video&Audio: all features - Late fusion
#echo "Starting new Data Experiment Configuration: all_data_va_late_fusion"
#python entry_point.py all_data_va_late_fusion.json> results/all_data_va_late_fusion.txt
#
## All participants together - Video&Audio: all features - Early fusion
#echo "Starting new Data Experiment Configuration: all_data_va_early_fusion"
#python entry_point.py all_data_va_early_fusion.json> results/all_data_va_early_fusion.txt


# SVM model & Early Fusion
# All participants separately - Video&Audio: all features - Late fusion
echo "Starting new Data Experiment Configuration: participant_01_va_early_fusion"
python entry_point.py participant_01_va_early_fusion.json> results/participant_01_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: participant_02_va_early_fusion"
python entry_point.py participant_02_va_early_fusion.json> results/participant_02_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: participant_03_va_early_fusion"
python entry_point.py participant_03_va_early_fusion.json> results/participant_03_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: participant_04_va_early_fusion"
python entry_point.py participant_04_va_early_fusion.json> results/participant_04_va_early_fusion.txt

# All sessions separately - Video&Audio: all features - Early fusion
echo "Starting new Data Experiment Configuration: session_01_01_va_early_fusion"
python entry_point.py session_01_01_va_early_fusion.json> results/session_01_01_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_02_01_va_early_fusion"
python entry_point.py session_02_01_va_early_fusion.json> results/session_02_01_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_02_02_va_early_fusion"
python entry_point.py session_02_02_va_early_fusion.json> results/session_02_02_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_03_01_va_early_fusion.json"
python entry_point.py session_03_01_va_early_fusion.json> results/session_03_01_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_03_02_va_early_fusion.json"
python entry_point.py session_03_02_va_early_fusion.json> results/session_03_02_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_04_01_va_early_fusion.json"
python entry_point.py session_04_01_va_early_fusion.json> results/session_04_01_va_early_fusion.txt

echo "Starting new Data Experiment Configuration: session_04_02_va_early_fusion.json"
python entry_point.py session_04_02_va_early_fusion.json> results/session_04_02_va_early_fusion.txt
