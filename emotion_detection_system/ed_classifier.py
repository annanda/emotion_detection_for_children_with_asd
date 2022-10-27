import os.path
from functools import reduce

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from emotion_detection_system.conf import DATASET_FOLDER, ORDER_EMOTIONS, TOTAL_SESSIONS, PARTICIPANT_NUMBERS, \
    LLD_PARAMETER_GROUP


class EmotionDetectionConfiguration:
    def __init__(self, config):
        self.configuration = config
        self.participant_number = 0
        self.session_number = str(self.configuration['session_number'])
        self.all_participant_data = self.configuration['all_participant_data']
        self.sessions_to_consider = []
        self.dataset_split_type = self.configuration['dataset_split_type']
        self.person_independent_model = self.configuration['person_independent_model']
        self.balance_dataset = self.configuration['balanced_dataset']
        self.balance_dataset_technique = self.configuration['balance_dataset_technique']
        self.classifier_model = self.configuration['classifier_model']
        self.fusion_type = self.configuration['fusion_type']
        self.modalities = self.configuration['modalities']
        self.modalities_config = self.configuration['modalities_config']
        self.video_features_types = []
        self.audio_features_level = ''
        self.audio_features_groups = []
        self.audio_features_types = {}
        self.models = {}
        self.is_multimodal = True if len(self.modalities) > 1 else False

        # To define the path for person-independent model or individuals model
        self.person_independent_folder = 'cross-individuals' if self.configuration[
            'person_independent_model'] else 'individuals'
        self._check_and_set_participant_number()
        self._setup_modalities_features_type_values()
        self._setup_classifier_models()
        self._setup_sessions_to_consider()

    def _setup_modalities_features_type_values(self):
        """
        To set up the modalities and features type of each modality
        """

        if 'video' in self.modalities:
            self._setup_features_video()

        if 'audio' in self.modalities:
            self._setup_features_audio()

    def _setup_features_video(self):
        self.video_features_types = [feature for feature in
                                     list(self.modalities_config['video']['features_type'].keys())
                                     if
                                     self.modalities_config['video']['features_type'][feature] is True]

    def _setup_features_audio(self):
        audio_config = self.configuration['modalities_config']['audio']
        self.audio_features_level = audio_config['feature_level']
        self.audio_features_groups = [group for group in list(audio_config['feature_group'].keys()) if
                                      audio_config['feature_group'][group] is True]

        for feature_group in self.audio_features_groups:
            if audio_config.get('all_features_from_group', False):
                # if I want to use all features type from all the groups
                self.audio_features_types[feature_group] = LLD_PARAMETER_GROUP[feature_group]
            else:
                # if I want to configure in more details which feature type I want to use from each group
                self.audio_features_types[feature_group] = audio_config['features_type'][feature_group] if not \
                    audio_config['features_type'][feature_group] == 'all' else LLD_PARAMETER_GROUP[feature_group]

    def _setup_classifier_models(self):
        """
        To Set up the classifier model for each modality
        """
        models = {}
        for modality in self.modalities:
            if self.configuration['classifier_model'].get('all_modalities', False):
                models[modality] = self.configuration['classifier_model']['all_modalities']
            else:
                models[modality] = self.configuration['classifier_model'][modality]
            if self.configuration['classifier_model'].get('early_fusion_model', False):
                models['early_fusion_model'] = self.configuration['classifier_model']['early_fusion_model']
        self.models = models

    def _setup_sessions_to_consider(self):

        if self.configuration.get('run_to_all_participants', False):
            self.sessions_to_consider = TOTAL_SESSIONS
            return
        if 'sessions_to_consider' in self.configuration.keys():
            self.sessions_to_consider = self.configuration['sessions_to_consider']
            return

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


class PrepareDataset:
    def __init__(self, configuration):
        self.configuration = EmotionDetectionConfiguration(configuration)

        # First we always fill the video or audio modalities, then, we make them become this main one. According to
        # the configuration of the system
        self.x = None
        self.x_dev = None
        self.x_test = None
        self.y = None
        self.y_dev = None
        self.y_test = None

        # video
        self.x_video = None
        self.x_dev_video = None
        self.x_test_video = None

        # audio
        self.x_audio = None
        self.x_dev_audio = None
        self.x_test_audio = None

        self.prepare_dataset()

    def prepare_dataset(self):
        print('Preparing the dataset according to the configuration:')
        if 'video' in self.configuration.modalities:
            self._prepare_dataset_video_modality()
        if 'audio' in self.configuration.modalities:
            self._prepare_dataset_audio_modality()
        if self.configuration.is_multimodal:
            if self.configuration.fusion_type == 'early_fusion':
                self._prepare_dataset_early_fusion()
        else:
            self._prepare_dataset_unimodality()

        print('hello!')

    def _prepare_dataset_unimodality(self):
        if self.configuration.modalities[0] == 'video':
            self.x = self.x_video.iloc[:, 4:]
            self.x_dev = self.x_dev_video.iloc[:, 4:]
            self.x_test = self.x_test_video.iloc[:, 4:]
        else:
            self.x = self.x_audio.iloc[:, 4:]
            self.x_dev = self.x_dev_audio.iloc[:, 4:]
            self.x_test = self.x_test_audio.iloc[:, 4:]

    def _prepare_dataset_early_fusion(self):
        dfs = [self.x_video, self.x_audio]
        dfs_dev = [self.x_dev_video, self.x_dev_audio]
        dfs_test = [self.x_test_video, self.x_test_audio]
        self.merge_dfs_columns(dfs, dfs_dev, dfs_test, 'multimodality')

    def concatenate_dfs_rows(self, folders_to_concat_df):
        df, df_dev, df_test = ([], [], [])
        for folder in folders_to_concat_df:
            df_temp, df_dev_temp, df_test_temp = self.read_dataset_split_parts(folder)
            df.append(df_temp)
            df_dev.append(df_dev_temp)
            df_test.append(df_test_temp)

        df = pd.concat(df)
        df_dev = pd.concat(df_dev)
        df_test = pd.concat(df_test)

        return df, df_dev, df_test

    def merge_dfs_columns(self, df_concatenate_columns,
                          df_dev_concatenate_columns,
                          df_test_concatenate_columns, modality):
        dfs = [df_concatenate_columns, df_dev_concatenate_columns, df_test_concatenate_columns]
        dfs_merged = []

        for df_to_merge in dfs:
            df_merged = reduce(lambda df1, df2: pd.merge(df1,
                                                         df2,
                                                         on=['time_of_video_seconds',
                                                             'emotion_zone',
                                                             'frametime',
                                                             'video_part']),
                               df_to_merge)
            dfs_merged.append(df_merged)

        self.set_dataset_values(dfs_merged, modality)

    def set_dataset_values(self, dfs_merged, modality):
        df = dfs_merged[0].dropna()
        df_dev = dfs_merged[1].dropna()
        df_test = dfs_merged[2].dropna()

        self.y = df['emotion_zone']
        self.y_dev = df_dev['emotion_zone']
        self.y_test = df_test['emotion_zone']

        if modality == 'video':
            self.x_video = df
            self.x_dev_video = df_dev
            self.x_test_video = df_test
            return
        if modality == 'audio':
            self.x_audio = df
            self.x_dev_audio = df_dev
            self.x_test_audio = df_test
            return
        self.x = df.iloc[:, 4:]
        self.x_dev = df_dev.iloc[:, 4:]
        self.x_test = df_test.iloc[:, 4:]

    def read_dataset_split_parts(self, folder_to_read):
        df = self._read_dataset_from_folders(folder_to_read, 'train')
        df_dev = self._read_dataset_from_folders(folder_to_read, 'dev')
        df_test = self._read_dataset_from_folders(folder_to_read, 'test')

        return df, df_dev, df_test

    def _prepare_dataset_video_modality(self):
        folder_modality = os.path.join(DATASET_FOLDER,
                                       'video',
                                       self.configuration.person_independent_folder,
                                       self.configuration.dataset_split_type)

        df_concatenate_columns = []
        df_dev_concatenate_columns = []
        df_test_concatenate_columns = []

        for feature in self.configuration.video_features_types:
            folders_read = self._prepare_dataset_path_video_modality_one_feature(folder_modality, feature)
            df, df_dev, df_test = self.concatenate_dfs_rows(folders_read)

            df_concatenate_columns.append(df)
            df_dev_concatenate_columns.append(df_dev)
            df_test_concatenate_columns.append(df_test)

        self.merge_dfs_columns(df_concatenate_columns,
                               df_dev_concatenate_columns,
                               df_test_concatenate_columns, 'video')

    def _prepare_dataset_audio_modality(self):
        folder_modality = os.path.join(DATASET_FOLDER,
                                       'audio',
                                       self.configuration.person_independent_folder,
                                       self.configuration.dataset_split_type,
                                       self.configuration.audio_features_level)

        df_concatenate_columns = []
        df_dev_concatenate_columns = []
        df_test_concatenate_columns = []

        for group in self.configuration.audio_features_groups:
            for feature in self.configuration.audio_features_types[group]:
                folders_to_read = self._prepare_dataset_path_audio_modality_one_feature(folder_modality, group, feature)
                df, df_dev, df_test = self.concatenate_dfs_rows(folders_to_read)

                df_concatenate_columns.append(df)
                df_dev_concatenate_columns.append(df_dev)
                df_test_concatenate_columns.append(df_test)

        self.merge_dfs_columns(df_concatenate_columns,
                               df_dev_concatenate_columns,
                               df_test_concatenate_columns, 'audio')

    def _prepare_dataset_path_video_modality_one_feature(self, folder_read_modality, feature):
        folders_one_modality_one_feature = []
        for session in self.configuration.sessions_to_consider:
            folders_one_modality_one_feature.append(os.path.join(
                folder_read_modality,
                feature,
                session))
        return folders_one_modality_one_feature

    def _prepare_dataset_path_audio_modality_one_feature(self, folder_modality, group, feature):
        folders_one_modality_one_feature = []
        for session in self.configuration.sessions_to_consider:
            folders_one_modality_one_feature.append(os.path.join(
                folder_modality,
                group,
                feature,
                session))
        return folders_one_modality_one_feature

    def _read_dataset_from_folders(self, folder_to_read, dataset_type):
        """
        param folder_to_read: list with folders to read for building the dataset to read
        param dataset_type: the part of the dataset desired, i.e., train, dev or test
        :return: the concatenated dataset for the specific type, from all the folders defined by 'folders_to_read'
        """
        split_dfs = []
        type_files_in_folder = [type_file for type_file in os.listdir(folder_to_read) if dataset_type in type_file]
        for type_file in type_files_in_folder:
            dataset_df = pd.read_pickle(os.path.join(folder_to_read, type_file))
            split_dfs.append(dataset_df)
        concatenated_split_dataset = pd.concat(split_dfs)
        return concatenated_split_dataset


class EmotionDetectionClassifier:

    def __init__(self, configuration):
        self._emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}
        self.configuration = EmotionDetectionConfiguration(configuration)

        # Setting up dataset
        dataset = PrepareDataset(configuration)
        self.y = dataset.y
        self.y_dev = dataset.y_dev
        self.y_test = dataset.y_test

        self.x = dataset.x
        self.x_dev = dataset.x_dev
        self.x_test = dataset.x_test

        self.x_video = dataset.x_video
        self.x_dev_video = dataset.x_dev_video
        self.x_test_video = dataset.x_test_video

        self.x_audio = dataset.x_audio
        self.x_dev_audio = dataset.x_dev_audio
        self.x_test_audio = dataset.x_test_audio

        self.data_sets_dic = {
            'x': [self.x, self.x_dev, self.x_test],
            'x_video': [self.x_video, self.x_dev_video, self.x_test_video],
            'x_audio': [self.x_audio, self.x_dev_audio, self.x_test_audio],
        }

        # Results of processing
        self._prediction_probabilities = None
        self._prediction_labels = None
        self.accuracy = None
        self.confusion_matrix = None
        self.classifier_model = None
        self._set_classifier_model()

    def train_model_produce_predictions(self):
        """
        To train the model and get predictions for x_test
        """
        print(f'Running experiment with configuration:')
        print('. . . . . . . . . . . . . . .  . . .')
        print(self.__str__())
        print('. . . . . . . . . . . . . . . . . .')
        print('.\n.\n.')

        if not self.configuration.is_multimodal or self.configuration.fusion_type == 'early_fusion':
            prediction_probability, indexes = self._train_model_produce_predictions_basic()
            self._prediction_probabilities = pd.DataFrame(prediction_probability,
                                                          columns=ORDER_EMOTIONS,
                                                          index=indexes)
            self._produce_final_predictions()
        else:
            self._train_model_produce_predictions_late_fusion()

    def _train_model_produce_predictions_basic(self, dataset):
        print('Starting to train the model')
        print('.\n.\n.')
        self.classifier_model.fit(self.data_sets_dic[dataset][0], self.y)

        print('Calculating predictions for test set')
        print('.\n.\n.')
        prediction_probability = self.classifier_model.predict_proba(self.data_sets_dic[dataset][2])
        indexes = list(self.y_test.index)
        return prediction_probability, indexes

    def _produce_final_predictions(self):
        self._prediction_labels = self._get_final_label_prediction_array()
        self._calculate_accuracy()
        self._calculate_confusion_matrix()

    def _train_model_produce_predictions_late_fusion(self):
        video_predictions = None
        audio_predictions = None

        #TODO do this part of late fusion
        self._produce_final_predictions()

    def _set_classifier_model(self):
        # simplest case
        # todo elaborate the selection and definition of models.
        self.classifier_model = svm.SVC(probability=True)

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
        print(f'Total train examples: {len(self.x)}')
        print(f'Total test examples: {np.sum(self.confusion_matrix)}')
        # print(f'Number of examples per class in test: \n{self.y_test.value_counts()}')
        print(f'Total number of features: {self.x.shape[1]}')
        print(f'Number of examples per class in training: \n{self.y.value_counts()}')

    def __str__(self):
        return f'Participant number: 0{self.configuration.participant_number}\n' \
               f'Session number: 0{self.configuration.session_number}\n' \
               f'All participant data: {self.configuration.all_participant_data}\n' \
               f'Sessions to consider: {self.configuration.sessions_to_consider}\n' \
               f'Dataset split type: {self.configuration.dataset_split_type}\n' \
               f'Person-Independent model: {self.configuration.person_independent_model}\n' \
               f'Modalities: {self.configuration.modalities}\n' \
               f'Features type video: {self.configuration.video_features_types}\n' \
               f'Features level audio: {self.configuration.audio_features_level}\n' \
               f'Features groups audio: {self.configuration.audio_features_groups}\n' \
               f'Features type audio: {self.configuration.audio_features_types}\n' \
               f'Classifier model: {self.configuration.classifier_model}\n' \
               f'Models per modality: {self.configuration.models}\n' \
               f'Fusion type: {self.configuration.fusion_type}'
