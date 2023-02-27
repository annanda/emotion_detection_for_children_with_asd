import json
import os.path
import pathlib
from datetime import datetime

import sys

from emotion_detection_system.ed_classifier import EmotionDetectionClassifier
from emotion_detection_system.scripts.process_results import get_scenario
from conf import emotion_detection_system_folder


def script_entry(json_file):
    path_json = os.path.join(emotion_detection_system_folder, 'json_files', json_file)

    f = open(path_json)
    configure_data = json.load(f)

    classifier = EmotionDetectionClassifier(configure_data)
    classifier.train_model_produce_predictions()
    classifier.show_results()

    # If want to generate structured json files with the results
    generate_json_results(classifier, path_json)


def generate_json_results(classifier, path_json):
    classifier.format_json_results()
    json_result = classifier.json_results_information

    file_name = pathlib.Path(path_json).name
    slug = file_name

    json_result['Data_Included_Slug'] = slug.split('.json')[0]
    json_result['Scenario'] = get_scenario(json_result['Data_Included_Slug'])
    save_json_result_file(json_result, file_name)


def save_json_result_file(dict_json, file_name):
    date = f'{datetime.now():%d%m%y}'
    path_json_results = os.path.join(emotion_detection_system_folder, 'json_results', date,
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
    # json_file = 'example_2.json'
    # json_file = 'example_2_x_x_dev_training.json'
    # json_file = 'specialist_baseline/session_03_01_v.json'
    # json_file = 'specialist_undersampling/session_04_02_va_late_fusion.json'

    script_entry(json_file)

    # TODO add to README the hierarchic order of the configuration option for data experiments
    ############################################################
    # Example of configuration
    ############################################################
