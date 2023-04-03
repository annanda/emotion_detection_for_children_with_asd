import os.path
from functools import reduce
from collections import Counter
import pickle

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, \
    precision_score, precision_recall_fscore_support, classification_report, multilabel_confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import RFECV, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV

from emotion_detection_system.conf import DATASET_FOLDER, ORDER_EMOTIONS, TOTAL_SESSIONS, PARTICIPANT_NUMBERS, \
    LLD_PARAMETER_GROUP, PARAMETER_GRID_SEARCH

CLASSES_NAME_TO_NUMBERS_DICT = {
    'blue': 0,
    'green': 1,
    'red': 2,
    'yellow': 3
}

CLASSES_NUMBERS_TO_NAMES_DICT = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'yellow'
}


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
        if self.balance_dataset:
            self.balance_dataset_technique = self.configuration['balance_dataset_technique']
        else:
            self.balance_dataset_technique = None
        if self.balance_dataset_technique == 'oversampling':
            self.oversampling_method = self.configuration.get('oversampling_method', 'random')
        else:
            self.oversampling_method = None
        self.classifier_model = self.configuration['classifier_model']
        self.modalities = self.configuration['modalities']
        self.modalities_config = self.configuration['modalities_config']
        self.video_features_types = []
        self.audio_features_level = ''
        self.audio_features_groups = []
        self.audio_features_types = {}
        self.models = {}
        self.is_multimodal = True if len(self.modalities) > 1 else False
        if self.is_multimodal:
            self.fusion_type = self.configuration['fusion_type']
        else:
            self.fusion_type = None
        self.annotation_type = self.configuration.get('annotation_type', 'parents')
        self.normaliser = {'default': MinMaxScaler(), 'audio': MinMaxScaler(), 'video': MinMaxScaler()}
        self.rfe = self.configuration.get('recursive_feature_elimination', False)
        if self.rfe:
            self.rfe_algorithm = self.configuration.get('RFE_algorithm', 'random_forest')
        else:
            self.rfe_algorithm = None
        self.grid_search = self.configuration.get('grid_search', False)
        self.load_trained_model = self.configuration.get('load_trained_model', False)

        # To define the path for person-independent model or individuals model
        self.person_independent_folder = 'cross-individuals' if self.configuration[
            'person_independent_model'] else 'individuals'
        self._check_and_set_participant_number()
        self._setup_modalities_features_type_values()
        self._setup_classifier_models()
        self._setup_sessions_to_consider()

        self.x_and_x_validation_for_training = self.configuration.get('x_and_x_validation_for_training', False)
        # self.x_dev_balanced = self.configuration.get('x_dev_balanced', False)

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
        if not self.video_features_types:
            raise ValueError(
                'You did not select a feature type for VIDEO modality. \n'
                'If you select a modality, you need to have at least one feature type.')

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
        if not self.audio_features_types:
            raise ValueError(
                'You did not select a feature type for AUDIO modality. \n'
                'If you select a modality, you need to have at least one feature type.')

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
        self.y_video = None
        self.y_dev_video = None
        self.y_test_video = None

        # audio
        self.x_audio = None
        self.x_dev_audio = None
        self.x_test_audio = None
        self.y_audio = None
        self.y_dev_audio = None
        self.y_test_audio = None

        self.dataset_path = self.set_annotation_type()

        self.prepare_dataset()

    def set_annotation_type(self):
        if self.configuration.annotation_type == 'parents':
            return os.path.join(DATASET_FOLDER, 'parents')
        elif self.configuration.annotation_type == 'specialist':
            return os.path.join(DATASET_FOLDER, 'specialist')

    def prepare_dataset(self):
        # print('Preparing the dataset....')
        if 'video' in self.configuration.modalities:
            self._prepare_dataset_video_modality()
        if 'audio' in self.configuration.modalities:
            self._prepare_dataset_audio_modality()
        if self.configuration.is_multimodal:
            if self.configuration.fusion_type == 'early_fusion':
                self._prepare_dataset_early_fusion()
            else:
                self._organise_multimodal_late_fusion_test_datasets()
        else:
            self._prepare_dataset_unimodality()

    def _prepare_dataset_unimodality(self):
        if self.configuration.modalities[0] == 'video':
            self.x = self.x_video
            self.x_dev = self.x_dev_video
            self.x_test = self.x_test_video
        else:
            self.x = self.x_audio
            self.x_dev = self.x_dev_audio
            self.x_test = self.x_test_audio

        self.y = self.x['emotion_zone']
        self.y_dev = self.x_dev['emotion_zone']
        self.y_test = self.x_test['emotion_zone']
        self.x = self.x.iloc[:, 4:]
        self.x_dev = self.x_dev.iloc[:, 4:]
        self.x_test = self.x_test.iloc[:, 4:]

    def _organise_multimodal_late_fusion_test_datasets(self):
        """
        It only gets here is the model is multimodal and late fusion type.
        Make the test set consistent. Keeping only the values that are present in both video and audio test datasets.
        :return: None
        """
        new_video_test = pd.merge(self.x_test_video,
                                  self.x_test_audio[['time_of_video_seconds',
                                                     'emotion_zone',
                                                     'frametime',
                                                     'video_part']],
                                  on=['time_of_video_seconds',
                                      'emotion_zone',
                                      'frametime',
                                      'video_part'])
        new_audio_test = pd.merge(self.x_test_audio,
                                  self.x_test_video[['time_of_video_seconds',
                                                     'emotion_zone',
                                                     'frametime',
                                                     'video_part']],
                                  on=['time_of_video_seconds',
                                      'emotion_zone',
                                      'frametime',
                                      'video_part'])
        self.x_test_video = new_video_test
        self.x_test_audio = new_audio_test
        self.y_test_video = self.x_test_video['emotion_zone']
        self.y_test_audio = self.x_test_audio['emotion_zone']

        # y_test_audio and y_test_video are exactly the same at this point.
        # So I can attribute either of them to y_test value.
        self.y_test = self.y_test_video

    def _prepare_dataset_early_fusion(self):
        dfs = [self.x_video, self.x_audio]
        dfs_dev = [self.x_dev_video, self.x_dev_audio]
        dfs_test = [self.x_test_video, self.x_test_audio]
        self.merge_dfs_columns(dfs, dfs_dev, dfs_test, 'early_fusion')

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
        df = dfs_merged[0].fillna(0)
        df_dev = dfs_merged[1].fillna(0)
        df_test = dfs_merged[2].fillna(0)

        # Balance Dataset (x_train) - undersampling
        # case of single modalities
        if self.configuration.balance_dataset and self.configuration.balance_dataset_technique in ['undersampling',
                                                                                                   'oversampling']:
            if self.configuration.is_multimodal:
                if modality == 'early_fusion' or self.configuration.fusion_type == 'late_fusion':
                    df = self.apply_balance(df)
                    # Balance x_dev if it will be used as part of training data
                    if self.configuration.x_and_x_validation_for_training:
                        df_dev = self.apply_balance(df_dev)
            else:
                df = self.apply_balance(df)
                # Balance x_dev if it will be used as part of training data
                if self.configuration.x_and_x_validation_for_training:
                    df_dev = self.apply_balance(df_dev)

        if modality == 'video':
            self.y_video = df['emotion_zone']
            self.y_dev_video = df_dev['emotion_zone']
            self.y_test_video = df_test['emotion_zone']
            self.x_video = df
            self.x_dev_video = df_dev
            self.x_test_video = df_test
            return
        if modality == 'audio':
            self.y_audio = df['emotion_zone']
            self.y_dev_audio = df_dev['emotion_zone']
            self.y_test_audio = df_test['emotion_zone']
            self.x_audio = df
            self.x_dev_audio = df_dev
            self.x_test_audio = df_test
            return
        self.y = df['emotion_zone']
        self.y_dev = df_dev['emotion_zone']
        self.y_test = df_test['emotion_zone']

        self.x = df.iloc[:, 4:]
        self.x_dev = df_dev.iloc[:, 4:]
        self.x_test = df_test.iloc[:, 4:]

    def apply_balance(self, df_to_balance):
        balanced_df = None
        if self.configuration.balance_dataset_technique == 'undersampling':
            group = df_to_balance.groupby('emotion_zone')
            resulting_df = group.apply(lambda x: x.sample(group.size().min()).reset_index())
            resulting_df = resulting_df.set_index(['index'])
            balanced_df = resulting_df
        if self.configuration.balance_dataset_technique == 'oversampling':
            if self.configuration.oversampling_method == 'random':
                ros = RandomOverSampler(random_state=0)
                X_resampled, y_resampled = ros.fit_resample(df_to_balance, df_to_balance[['emotion_zone']])
                balanced_df = X_resampled
            elif self.configuration.oversampling_method == 'adasyn':
                df_to_balance = self.represent_class_by_numbers(df_to_balance)
                df_to_balance_no_string = df_to_balance.drop(['frametime'], axis=1)
                X_resampled, y_resampled = ADASYN().fit_resample(df_to_balance_no_string,
                                                                 df_to_balance_no_string[['emotion_zone']])
                balanced_df = X_resampled
                balanced_df = self.represent_class_by_names(balanced_df)
                # Create an empty column to maintain the shape of the dataframe
                balanced_df.insert(3, 'frametime', np.nan)
                # print(y_resampled.value_counts())

        return balanced_df

    def represent_class_by_numbers(self, dataframe):
        dataframe['emotion_zone'] = dataframe.apply(self.get_class_number, axis=1)
        return dataframe

    def represent_class_by_names(self, dataframe):
        dataframe['emotion_zone'] = dataframe.apply(self.get_class_name, axis=1)
        return dataframe

    def read_dataset_split_parts(self, folder_to_read):
        df = self._read_dataset_from_folders(folder_to_read, 'train')
        df_dev = self._read_dataset_from_folders(folder_to_read, 'dev')
        df_test = self._read_dataset_from_folders(folder_to_read, 'test')

        return df, df_dev, df_test

    def get_class_number(self, df_row):
        emotion = df_row['emotion_zone']
        return CLASSES_NAME_TO_NUMBERS_DICT[emotion]

    def get_class_name(self, df_row):
        emotion_number = df_row['emotion_zone']
        return CLASSES_NUMBERS_TO_NAMES_DICT[emotion_number]

    def _prepare_dataset_video_modality(self):
        folder_modality = os.path.join(self.dataset_path,
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
        folder_modality = os.path.join(self.dataset_path,
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
        self.configuration = EmotionDetectionConfiguration(configuration)
        self._emotion_class = {}
        # self._emotion_class = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}

        # Setting up dataset
        self.dataset = PrepareDataset(configuration)

        self.data_sets_dic = {
            'x': [self.dataset.x, self.dataset.x_dev, self.dataset.x_test],
            'x_video': [self.dataset.x_video, self.dataset.x_dev_video, self.dataset.x_test_video],
            'x_audio': [self.dataset.x_audio, self.dataset.x_dev_audio, self.dataset.x_test_audio],
        }

        self.data_sets_dic_y = {
            'x': [self.dataset.y, self.dataset.y_dev, self.dataset.y_test],
            'x_video': [self.dataset.y_video, self.dataset.y_dev_video, self.dataset.y_test_video],
            'x_audio': [self.dataset.y_audio, self.dataset.y_dev_audio, self.dataset.y_test_audio],
        }

        # Results of processing
        self._prediction_probabilities = None
        self._prediction_labels = None
        self.accuracy = None
        self.balanced_accuracy = None
        # self.recall, self.precision, self.f1score, self.support = None, None, None, None
        self.classification_report = None
        self.confusion_matrix = None
        self.classifier_model = {}
        self._current_model = None
        self.emotion_from_classifier = None
        self.train_data_description = "x + x_validation sets" if self.configuration.x_and_x_validation_for_training else 'X set'

        self._set_classifier_model()
        self.json_results_information = {}

    def train_model_produce_predictions(self):
        """
        To train the model and get predictions for x_test
        """

        if not self.configuration.is_multimodal or self.configuration.fusion_type == 'early_fusion':
            indexes_test = list(self.dataset.y_test.index)
            prediction_probability = self._train_model_produce_predictions_basic()
            self._prediction_probabilities = pd.DataFrame(prediction_probability,
                                                          columns=self.emotion_from_classifier,
                                                          index=indexes_test)
        else:
            self._train_model_produce_predictions_late_fusion()
        self._produce_final_predictions()

    def set_x_y_to_train(self, dataset):
        if dataset:
            x = self.data_sets_dic[dataset][0]
            x = x.iloc[:, 4:]
            x_dev = self.data_sets_dic[dataset][1]
            x_dev = x_dev.iloc[:, 4:]
            x_test = self.data_sets_dic[dataset][2]
            x_test = x_test.iloc[:, 4:]
            y = self.data_sets_dic_y[dataset][0]
            y_dev = self.data_sets_dic_y[dataset][1]
        else:
            x = self.dataset.x
            y = self.dataset.y
            x_dev = self.dataset.x_dev
            y_dev = self.dataset.y_dev
            x_test = self.dataset.x_test

        if self.configuration.x_and_x_validation_for_training:
            x_added = pd.concat([x, x_dev])
            x = x_added
            y = pd.concat([y, y_dev])

        return x, y, x_dev, y_dev, x_test

    def get_modality(self, dataset):
        if not dataset:
            return "default"
        elif dataset == 'x_video':
            return "video"
        elif dataset == 'x_audio':
            return "audio"

    def _train_model_produce_predictions_basic(self, dataset=None):
        """
        :param: dataset: Can be x_video, x_audio or None (default).
        :return: The prediction probabilities for the current model + dataset.
        """
        x, y, x_dev, y_dev, x_test = self.set_x_y_to_train(dataset)

        x = x.fillna(0)

        modality = self.get_modality(dataset)

        # executor can be a pipeline or a search grid
        pipeline = self.create_pipeline(modality)

        if self.configuration.grid_search:
            executor = self.create_grid_search(pipeline)
        else:
            executor = pipeline

        if self.configuration.load_trained_model:
            executor = pickle.load(open(
                '/Users/annanda/PycharmProjects/emotion_detection_system/emotion_detection_system/trained_models/300323/specialist/example_rfe.pickle',
                'rb')
            )
            print('using saved model!')
        else:
            executor.fit(x, y)

        self.emotion_from_classifier = executor.classes_

        emotions = self.emotion_from_classifier
        for idx, emotion in enumerate(emotions):
            self._emotion_class[idx] = emotion

        prediction_probability = executor.predict_proba(x_test)

        if dataset:
            if dataset == 'x_video':
                self.video_executor = executor
            elif dataset == 'x_audio':
                self.audio_executor = executor
        else:
            self.executor = executor

        return prediction_probability

    def create_pipeline(self, modality):
        # basic pipeline (with just normaliser)
        steps = [('normalise', self.configuration.normaliser[modality]), ('model', self.classifier_model[modality])]

        # Pipeline with Recursive Features Elimination
        if self.configuration.rfe:
            if self.configuration.rfe_algorithm in ['svm_linear', 'random_forest']:
                # rfe_algorithm = RFECV(estimator=RandomForestClassifier(), scoring='accuracy', cv=3, verbose=1)
                rfe_algorithm = RFECV(estimator=svm.SVC(kernel='linear'), scoring='accuracy', cv=3, verbose=0)
                # rfe_algorithm = RFE(estimator=svm.SVC(kernel='linear'), verbose=1, n_features_to_select=0.6)
            steps = [('normalise', self.configuration.normaliser[modality]), ('rfe', rfe_algorithm),
                     ('model', self.classifier_model[modality])]

        pipeline = Pipeline(steps=steps)
        return pipeline

    def create_grid_search(self, pipeline):
        grid = GridSearchCV(pipeline, param_grid=PARAMETER_GRID_SEARCH, cv=5)
        return grid

    def _train_model_produce_predictions_late_fusion(self):
        # In the end I need to fill self._prediction_probabilities, so I can run _produce_final_predictions
        # So I need to calculate the video and audio predictions separately, then fusion them, and the result will
        # be stored in the self._prediction_probabilities variable.

        video_predictions = self._train_model_produce_predictions_basic('x_video')
        audio_predictions = self._train_model_produce_predictions_basic('x_audio')

        # Late fusion by averaging the predictions array
        indexes_test = list(self.dataset.y_test_video.index)
        fusion_by_mean = np.mean(np.array([video_predictions, audio_predictions]), axis=0)
        self._prediction_probabilities = pd.DataFrame(fusion_by_mean,
                                                      columns=self.emotion_from_classifier,
                                                      index=indexes_test)

    def _produce_final_predictions(self):
        self._prediction_labels = self._get_final_label_prediction_array()
        self._calculate_accuracy()
        self._calculate_balanced_accuracy()
        # self._calculate_recall()
        # self._calculate_precision()
        self._calculate_precision_recall_f1score_support()
        self._generate_classification_report()
        self._calculate_confusion_matrix()
        self._calculate_multilabel_confusion_matrix()

    def _set_classifier_model(self):
        # simplest case
        # todo elaborate the selection and definition of models.
        if self.configuration.balance_dataset and self.configuration.balance_dataset_technique == 'class_weight':
            self.classifier_model = svm.SVC(probability=True, class_weight="balanced")
        else:
            self.classifier_model['default'] = svm.SVC(probability=True)
            self.classifier_model['audio'] = svm.SVC(probability=True)
            self.classifier_model['video'] = svm.SVC(probability=True)
        self._current_model = 'SVM'

    def _get_final_label_prediction_array(self):
        """
        To get the prediction array with labels, not probabilities prediction
        """
        predictions = []
        for _, row in self._prediction_probabilities.iterrows():
            prob = np.array(row[self.emotion_from_classifier])
            label = self._get_predicted_label(prob)
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
        self.accuracy = accuracy_score(self.dataset.y_test, self._prediction_labels)

    def _calculate_balanced_accuracy(self):
        """
        average of recall obtained on each class, only count the classes the model predicted.
        """
        self.balanced_accuracy = balanced_accuracy_score(self.dataset.y_test, self._prediction_labels)

    def _calculate_recall(self):
        """
        True positive rate. What was correctly predicted divided by what should have been predicted. Per class.
        E.g.: “What percentage of the green class was identified correctly?”
        """
        self.recall = recall_score(self.dataset.y_test, self._prediction_labels, average=None, labels=ORDER_EMOTIONS)

    def _calculate_precision(self):
        """
        Precision is the percentage of data samples that a machine learning model correctly identifies for a class out
        of all samples predicted to belong to that class.
        e.g.: "What percentage of the green class classified by the model was indeed correct?"
        """
        self.precision = precision_score(self.dataset.y_test, self._prediction_labels, average=None,
                                         labels=ORDER_EMOTIONS)

    def _calculate_precision_recall_f1score_support(self):
        """
        Recall = True positive rate. What was correctly predicted divided by what should have been predicted. Per class.
        E.g.: “What percentage of the green class was identified correctly?”

        Precision is the percentage of data samples that a machine learning model correctly identifies for a class out
        of all samples predicted to belong to that class.
        e.g.: "What percentage of the green class classified by the model was indeed correct?"

        F1 score is harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and
        worst score at 0

        Support is the number of examples of each class present in the y_true.
        """
        self.precision, self.recall, self.f1score, self.support = precision_recall_fscore_support(self.dataset.y_test,
                                                                                                  self._prediction_labels,
                                                                                                  average=None,
                                                                                                  labels=self.emotion_from_classifier)

    def _generate_classification_report(self):
        self.classification_report = classification_report(self.dataset.y_test, self._prediction_labels,
                                                           labels=self.emotion_from_classifier, digits=4)

    def _calculate_confusion_matrix(self):
        """
        To calculate confusion matrix
        """
        self.confusion_matrix = confusion_matrix(self.dataset.y_test,
                                                 self._prediction_labels,
                                                 labels=self.emotion_from_classifier)

    def _calculate_multilabel_confusion_matrix(self):
        self.multilabel_confusion_matrix = multilabel_confusion_matrix(self.dataset.y_test, self._prediction_labels,
                                                                       labels=self.emotion_from_classifier)

    def format_json_results(self):
        # header_csv = ['Data_Included_Slug', 'Scenario', 'Annotation_Type', 'Accuracy', 'Accuracy_Balanced',
        #               'Precision_Blue', 'Precision_Green', 'Precision_Red', 'Precision_Yellow', 'Recall_Blue',
        #               'Recall_Green', 'Recall_Red', 'Recall_Yellow', 'F1score_Blue',
        #               'F1score_Green', 'F1score_Red', 'F1score_Yellow']

        self.json_results_information = {
            'Data_Included_Slug': '',
            'Scenario': '',
            'Participant': self.configuration.participant_number,
            'Session': self.configuration.session_number,
            'Annotation_Type': self.configuration.annotation_type,
            'Accuracy': self.accuracy,
            'Accuracy_Balanced': self.balanced_accuracy,
            'Precision_blue': None, 'Precision_green': None, 'Precision_red': None, 'Precision_yellow': None,
            'Recall_blue': None, 'Recall_green': None, 'Recall_red': None, 'Recall_yellow': None,
            'F1score_blue': None, 'F1score_green': None, 'F1score_red': None, 'F1score_yellow': None
        }

        self._fill_json_results()

    def _fill_json_results(self):
        classes = self.emotion_from_classifier
        for x, emotion in enumerate(classes):
            precision_key = f'Precision_{emotion}'
            recall_key = f'Recall_{emotion}'
            f1_key = f'F1score_{emotion}'
            self.json_results_information[precision_key] = self.precision[x]
            self.json_results_information[recall_key] = self.recall[x]
            self.json_results_information[f1_key] = self.f1score[x]

    def _get_train_examples_number_to_print(self):
        if self.dataset.x is not None:
            # if self.configuration.x_and_x_validation_for_training:
            #     return len(self.dataset.x) + len(self.dataset.x_dev)
            return len(self.dataset.y)
        else:
            # if self.configuration.x_and_x_validation_for_training:
            #     return len(self.dataset.x_video) + len(self.dataset.x_dev_video)
            return len(self.dataset.y_video)

    def print_train_example_number_per_class(self):
        print('Train examples of each class:')
        if self.dataset.x is not None:
            print(dict(self.dataset.y.value_counts()))
        else:
            print(dict(self.dataset.y_video.value_counts()))

    def print_train_features_number(self):
        if self.dataset.x is not None:
            print(f'Total number of features: {self.dataset.x.shape[1]}')
            if self.configuration.is_multimodal:
                print(f'Total number of video features: {self.dataset.x_video.shape[1] - 4}')
                print(f'Total number of audio features: {self.dataset.x_audio.shape[1] - 4}')
            if self.configuration.rfe:
                print(f'Total number of features - after RFE: {self.executor.n_features_in_}')
        else:
            print(f'Total number of features: {self.dataset.x_video.shape[1] - 4 + self.dataset.x_audio.shape[1] - 4}')
            print(f'Total number of video features: {self.dataset.x_video.shape[1] - 4}')
            print(f'Total number of audio features: {self.dataset.x_audio.shape[1] - 4}')
            if self.configuration.rfe:
                print(f'Total number of features (video) - after RFE: {self.video_executor.n_features_in_}')
                print(f'Total number of features (audio) - after RFE: {self.audio_executor.n_features_in_}')

    def print_search_grid_elements(self):
        if self.configuration.grid_search:
            print("##################")
            if self.dataset.x is not None:
                # uni modal or early fusion cases
                print("Best Search Grid Parameters:\n")
                print(self.executor.best_params_)
            else:
                # late fusion case
                print("Best Search Grid Parameters (video):\n")
                print(self.video_executor.best_params_)
                print("Best Search Grid Parameters (audio):\n")
                print(self.audio_executor.best_params_)

    def show_results(self):
        """
        To show the results after running an experiment
        :return:
        """
        print(f'######################################')
        print(' ... CONFIGURATION ... \n')
        print(self.__str__())
        print(f'###################################### \n')
        print(f' ... RESULTS ... \n')
        # print(f'Accuracy: {self.accuracy}')
        print(f'Accuracy: {self.accuracy:.4f}')
        print(f'Balanced Accuracy: {self.balanced_accuracy:.4f}')
        print(f'Classification report: {self.classification_report}')
        print(f'Confusion matrix: labels={self.emotion_from_classifier}')
        print(self.confusion_matrix)
        print(f'\nMultilabel Confusion matrix:')
        print(self.multilabel_confusion_matrix)
        print(f'Total train examples: {self._get_train_examples_number_to_print()}')
        print(f'Total test examples: {np.sum(self.confusion_matrix)}')
        self.print_train_example_number_per_class()
        # print(f'Number of examples per class in test: \n{self.y_test.value_counts()}')
        self.print_train_features_number()
        self.print_search_grid_elements()
        print(f'\n')
        print(f'######################################')

    def save_json_results(self):
        self.format_json_results()

    def __str__(self):
        return f'Participant number: 0{self.configuration.participant_number}\n' \
               f'Session number: 0{self.configuration.session_number}\n' \
               f'All participant data: {self.configuration.all_participant_data}\n' \
               f'Sessions to consider: {self.configuration.sessions_to_consider}\n' \
               f'Dataset split type: {self.configuration.dataset_split_type}\n' \
               f'Annotation type: {self.configuration.annotation_type}\n' \
               f'Person-Independent model: {self.configuration.person_independent_model}\n' \
               f'Modalities: {self.configuration.modalities}\n' \
               f'Features type video: {self.configuration.video_features_types}\n' \
               f'Features level audio: {self.configuration.audio_features_level}\n' \
               f'Features groups audio: {self.configuration.audio_features_groups}\n' \
               f'Features type audio: {self.configuration.audio_features_types}\n' \
               f'Models per modality: {self.configuration.models}\n' \
               f'Fusion type: {self.configuration.fusion_type}\n' \
               f'Data used for Training: {self.train_data_description}\n' \
               f'Loaded Trained model: {self.configuration.load_trained_model}\n' \
               f'Balanced Dataset: {self.configuration.balance_dataset}\n' \
               f'Balanced Dataset Technique: {self.configuration.balance_dataset_technique}\n' \
               f'Oversampling Method: {self.configuration.oversampling_method}\n' \
               f'RFE: {self.configuration.rfe}\n' \
               f'RFE algorithm: {self.configuration.rfe_algorithm}'
