import os.path

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from setup.conf import DATASET_FOLDER

ORDER_EMOTIONS = ['blue', 'green', 'red', 'yellow']
PARTICIPANT_NUMBERS = [1, 2, 3, 4]


class EmotionDetectionClassifier:
    def __init__(self, configuration):
        self._emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}
        self.configuration = configuration
        # From configuration input
        self.participant_number = 0
        self._check_and_set_participant_number()
        self.session_number = self.configuration['session_number']
        self.all_participant_data = self.configuration['all_participant_data']
        self.sessions_to_consider = []
        self.dataset_split_type = self.configuration['dataset_split_type']
        self.person_independent_model = self.configuration['person_independent_model']
        self.balance_dataset = self.configuration['balanced_dataset']
        self.balance_dataset_technique = self.configuration['balance_dataset_technique']
        self.classifier_model = self.configuration['classifier_model']
        self.fusion_type = self.configuration['fusion_type']
        self.modalities = []
        self.features_types = {}
        self.models = {}
        # To define the path for person-independent model or individuals model
        self._person_independent_folder = 'cross-individuals' if self.configuration[
            'person_independent_model'] else 'individuals'
        self._setup_modalities_features_type_values()
        self._setup_classifier_models()
        self._setup_sessions_to_consider()
        # Results of processing
        self.x = None
        self.y = None
        self.x_dev = None
        self.y_dev = None
        self.x_test = None
        self.y_test = None
        self._prediction_probabilities = None
        self._prediction_labels = None
        self.accuracy = None
        self.confusion_matrix = None

    def _setup_modalities_features_type_values(self):
        """
        To set up the modalities and features type of each modality
        """
        self.modalities = list(self.configuration['modalities'].keys())
        features_type = {}

        for modality in self.modalities:
            features_type[modality] = [feature for feature in
                                       list(self.configuration['modalities'][modality]['features_type'].keys())
                                       if self.configuration['modalities'][modality]['features_type'][feature] is True]

        self.features_types = features_type

    def _setup_classifier_models(self):
        """
        To Set up the classifier model for each modality
        """
        models = {}
        for modality in self.modalities:
            models[modality] = self.configuration['modalities'][modality]['model_for_modality']
        self.models = models

    def _prepare_dataset(self):
        """
        To attribute the correct values for x, y, x_test, y_test, x_dev, y_dev
        """
        print('Preparing the dataset according to the configuration:')
        # Get the right path to the expected dataset
        if len(self.modalities) == 1:
            folder_read = self._prepare_dataset_path_one_modality()
        else:
            raise ValueError('Just One modality supported')

        # To define dataset split after defining which folder to read
        x = self._get_dataset_split(folder_read, 'train')
        self.x = x.iloc[:, 4:]
        self.y = x['emotion_zone']

        x_dev = self._get_dataset_split(folder_read, 'dev')
        self.x_dev = x_dev.iloc[:, 4:]
        self.y_dev = x_dev['emotion_zone']

        x_test = self._get_dataset_split(folder_read, 'test')
        self.x_test = x_test.iloc[:, 4:]
        self.y_test = x_test['emotion_zone']

    def _prepare_dataset_path_one_modality(self):
        folder_read_modality = os.path.join(DATASET_FOLDER,
                                            self.dataset_split_type,
                                            self._person_independent_folder,
                                            self.modalities[0])
        # For only one type of feature
        if len(self.features_types) == 1:
            folder_read = self._prepare_dataset_path_one_modality_one_feature(folder_read_modality)
        else:
            raise ValueError('More than one type of features is not yet supported')

        return folder_read

    def _prepare_dataset_path_one_modality_one_feature(self, folder_read_modality):
        return os.path.join(folder_read_modality,
                            self.features_types[self.modalities[0]][0],
                            self.session_number)

    def _get_dataset_split(self, folder_read, dataset_split):
        split_dfs = []
        type_files_in_folder = [type_file for type_file in os.listdir(folder_read) if dataset_split in type_file]
        for type_file in type_files_in_folder:
            dataset_df = pd.read_pickle(os.path.join(folder_read, type_file))
            split_dfs.append(dataset_df)

        concated_split_dataset = pd.concat(split_dfs)
        return concated_split_dataset

    def train_model_produce_predictions(self):
        """
        To train the model and get predictions for x_test
        """
        print(f'Running experiment with configuration:')
        print('. . . . . . . . . . . . . . .  . . .')
        print(self.__str__())
        print('. . . . . . . . . . . . . . . . . .')
        print('.\n.\n.')

        # Preparing the dataset for the experiment
        self._prepare_dataset()

        print('Starting to train the model')
        print('.\n.\n.')

        if self.classifier_model == 'SVM':
            clf = svm.SVC(probability=True)
        clf.fit(self.x, self.y)

        print('Calculating predictions for test set')
        print('.\n.\n.')

        prediction_probability = clf.predict_proba(self.x_test)
        indexes = list(self.y_test.index)

        # organising the prediction results with the labels
        self._prediction_probabilities = pd.DataFrame(prediction_probability,
                                                      columns=ORDER_EMOTIONS,
                                                      index=indexes)

        self._prediction_labels = self._get_final_label_prediction_array()
        self._calculate_accuracy()
        self._calculate_confusion_matrix()

    def _get_final_label_prediction_array(self):
        """
        To get the prediction array with labels, not probabilities prediction
        """
        predictions = []
        for _, row in self._prediction_probabilities.iterrows():
            label = self._get_predicted_label(np.array(row[ORDER_EMOTIONS]))
            predictions.append(label)
        return predictions

    def _get_predicted_label(self, probability_vector):
        result = np.where(probability_vector == np.amax(probability_vector))
        emotion_index = result[0][0]
        return self._emotion_class[emotion_index]

    def _calculate_accuracy(self):
        """
        To calculate accuracy
        """
        self.accuracy = accuracy_score(self.y_test, self._prediction_labels)

    def _calculate_confusion_matrix(self):
        """
        To calculate confusion matrix
        """
        self.confusion_matrix = confusion_matrix(self.y_test,
                                                 self._prediction_labels,
                                                 labels=ORDER_EMOTIONS)

    def show_results(self):
        """
        To show the results after running an experiment
        :return:
        """
        print(f'######################################')
        print(' ... Results for the experiment: ...')
        print(self.__str__())
        print(f'######################################')
        print('.\n.\n.')
        # print(f'Accuracy: {self.accuracy}')
        print(f'Accuracy: {self.accuracy:.4f}')
        print(f'Confusion matrix: labels={ORDER_EMOTIONS}')
        print(self.confusion_matrix)

    def __str__(self):
        return f'Session number: {self.session_number}\n' \
               f'All participant data: {self.all_participant_data}\n' \
               f'Dataset split type: {self.dataset_split_type}\n' \
               f'Person-Independent model: {self.person_independent_model}\n' \
               f'Modalities: {self.modalities}\n' \
               f'Features type: {self.features_types}\n' \
               f'Classifier model: {self.classifier_model}\n' \
               f'Models per modality: {self.models}\n' \
               f'Fusion type: {self.fusion_type}'

    def _setup_sessions_to_consider(self):
        if self.configuration['all_participant_data']:
            self.sessions_to_consider.append(f'session_0{self.participant_number}_01')
            if self.participant_number != 1:
                self.sessions_to_consider.append(f'session_0{self.participant_number}_02')
        else:
            self.sessions_to_consider.append(f'session_0{self.participant_number}_0{self.session_number}')

    def _check_and_set_participant_number(self):
        if self.configuration['participant_number'] in PARTICIPANT_NUMBERS:
            self.participant_number = self.configuration['participant_number']
        else:
            raise ValueError(f'Participant number must be one of: {PARTICIPANT_NUMBERS}')
