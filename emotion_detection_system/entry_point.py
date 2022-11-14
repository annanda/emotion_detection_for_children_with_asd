import json
import os.path
import sys

from emotion_detection_system.ed_classifier import EmotionDetectionClassifier
from conf import emotion_detection_system_folder

if __name__ == '__main__':
    json_file = sys.argv[1]
    path_json = os.path.join(emotion_detection_system_folder, 'json_files', json_file)

    f = open(path_json)
    configure_data = json.load(f)

    classifier = EmotionDetectionClassifier(configure_data)
    classifier.train_model_produce_predictions()
    classifier.show_results()
    # TODO add to README the hierarchic order of the configuration option for data experiments
    ############################################################
    # Example of configuration
    ############################################################
