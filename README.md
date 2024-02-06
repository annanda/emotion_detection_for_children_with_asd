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


## Pre-requisites
- Python 3.8 
- Docker

## Installation
### Development
1. Prepare the virtual environment (Create and activate virtual environment with venv).

`python -m venv ./venv`

`source ./venv/bin/activate`

2. Run the script 

`python app.py`

### Deployment with Docker
1. Build the images

`docker compose build`

2. Start the services

`docker compose up -d`

## User Manual

1. Use the system ```annotation_tool``` to create the labels.
2. Use the system ```features_extraction``` to preprocess the features extracted from tools, e.g., OpenFace and
   OpenSMILE
   into groups and into correct time window.
3. Use the system ```split_dataset``` to split features and labels dataset into train, dev and test sets. The output
   from
   there will serve as input to this system.
4. Put the features/labels PKL files into the folder `/dataset/video` or `/dataset/audio` depending on the modality you
   want
   to use.

### Configuration for each experiment

- classifier_model: the classifier model for the whole multimodal ED system, usually used together with early fusion
  method, or in case of just one modality.
- model (within modality): the classifier model for the specific modality, usually used together with late fusion type.


### Run data experiments script 
Run the script from the main folder /emotion_detection_system

* If you select a modality, you need to select at least one features type.

## Licence

This repository is released under dual-licencing:

For non-commercial use of the Software, it is released under
the [3-Cause BSD Licence](https://opensource.org/license/bsd-3-clause/).

For commercial use of the Software, you are required to contact the University of Galway to arrange a commercial
licence.

Please refer to [LICENSE.md](LICENSE.md) file for details on the licence.

----

Author: Annanda Sousa

Author's contact: [annanda.sousa@gmail.com](mailto:annanda.sousa@gmail.com)

----