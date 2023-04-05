import json
import os.path
import pathlib
from datetime import datetime
import pickle

import sys

from emotion_detection_system.ed_classifier import EmotionDetectionClassifier
from emotion_detection_system.processing_results.process_results import get_scenario
from conf import emotion_detection_system_folder, TRAINED_MODELS_FOLDER, DATA_EXPERIMENT_SLUG

folder_to_save = DATA_EXPERIMENT_SLUG.split('_')[:-2]
date_experiment = DATA_EXPERIMENT_SLUG.split('_')[-1]
folder_to_save.append(date_experiment)
folder_to_save_name = '_'.join(folder_to_save)


def script_entry(json_file):
    path_json = os.path.join(emotion_detection_system_folder, 'json_files', json_file)
    file_name = pathlib.Path(path_json).name

    f = open(path_json)
    configure_data = json.load(f)

    classifier = EmotionDetectionClassifier(configure_data)
    classifier.train_model_produce_predictions()
    classifier.show_results()

    # If want to generate structured json files with the results
    generate_json_results(classifier, path_json)

    save_model(classifier, file_name)


def save_model(classifier, file_name):
    file_name = file_name.split('.json')[0]
    path_to_save = os.path.join(TRAINED_MODELS_FOLDER, folder_to_save_name, classifier.configuration.annotation_type)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if classifier.dataset.x is not None:
        output_path = os.path.join(path_to_save, file_name + '.pickle')
        # uni modal or early fusion cases
        pickle.dump(classifier.executor, open(output_path, "wb"))
    else:
        # late fusion case
        output_path_video = os.path.join(path_to_save, file_name + '_video' + '.pickle')
        output_path_audio = os.path.join(path_to_save, file_name + '_audio' + '.pickle')
        pickle.dump(classifier.video_executor, open(output_path_video, "wb"))
        pickle.dump(classifier.audio_executor, open(output_path_audio, "wb"))


def generate_json_results(classifier, path_json):
    classifier.format_json_results()
    json_result = classifier.json_results_information

    file_name = pathlib.Path(path_json).name
    slug = file_name

    json_result['Data_Included_Slug'] = slug.split('.json')[0]
    json_result['Scenario'] = get_scenario(json_result['Data_Included_Slug'])
    save_json_result_file(json_result, file_name)


def save_json_result_file(dict_json, file_name):
    path_json_results = os.path.join(emotion_detection_system_folder, 'json_results', folder_to_save_name,
                                     dict_json['Annotation_Type'])

    if not os.path.exists(path_json_results):
        os.makedirs(path_json_results)

    output_json_path = os.path.join(path_json_results, file_name)

    with open(output_json_path, 'w') as json_to_write:
        json.dump(dict_json, json_to_write)


if __name__ == '__main__':
    # if using shell script
    json_file = sys.argv[1]
    # If using local run
    # json_file = 'example_annotation_specialist.json'
    # json_file = 'example_balance_dataset.json'
    # json_file = 'example_rfe.json'
    # json_file = 'parents_rfe/session_01_01_v.json'
    # json_file = 'specialist_oversampling_random/session_02_01_v.json'

    script_entry(json_file)

    # TODO add to README the hierarchic order of the configuration option for data experiments
    ############################################################
    # Example of configuration
    ############################################################
