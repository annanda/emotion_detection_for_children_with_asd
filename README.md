# Multimodal Emotion Detection System

An end-to-end multimodal system to classify emotion zones. It supports three different type of input modalities, i.e. video, audio and physiological signals. 

This system does not extract features from raw files, the user needs to provide a dataset of extracted features. The other system ```features extraction``` does the part of extracting features, generating an output to be used as input by this system. 

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
1. Use the system ```annotation_tool``` to create the labels 
2. Use the system ```Features_Extraction``` to preprocess the features into groups. The output of that system will serve as input to this system.
3. Put the features CSVs files into the folder `/dataset/video` or `/dataset/audio` depending on the modality you want to use.
4. Change the `input_data` variable on file `main.py` to include the modality and features you want to use. The modalities and features you want to include need to have `True` valeu. 
5. Run `main.py`