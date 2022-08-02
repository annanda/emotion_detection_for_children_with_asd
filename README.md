# Multimodal Emotion Detection System

An end-to-end multimodal system to classify emotion zones. It supports three different type of input modalities, i.e.
video, audio and physiological signals.

This system does not extract features from raw files, the user needs to provide a dataset of extracted features. The
other system ```features extraction``` does the part of extracting features, generating an output to be used as input by
this system.

The system supports any combination of features and inputs listed below.

### Modalities of input

- video
- audio
- physio

### Visual features list

- AU
- appearance
- BoVW
- geometric
- gaze
- 2d_eye_landmark
- 3d_eye_landmark
- head_pose

### Audio features list

- BoAW
- DeepSpectrum
- eGeMAPSfunct

### Physiological signals features list

- HRHRV

## How to use the system

1. Use the system ```annotation_tool``` to create the labels.
2. Use the system ```features_extraction``` to preprocess the features extracted from tools, e.g., OpenFace and OpenSMILE
   into groups and into correct time window.
3. Use the system ```split_dataset``` to split features and labels dataset into train, dev and test sets. The output from
   there will serve as input to this system.
4. Put the features/labels PKL files into the folder `/dataset/video` or `/dataset/audio` depending on the modality you want
   to use.


5. Change the `input_data` variable on file `main.py` to include the modality and features you want to use. The
   modalities and features you want to include need to have `True` value.
6. Run `main.py`