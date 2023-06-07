from emotion_detection_system.ed_classifier import *

model_to_load_experiment_01 = 'data_experiments_nn_algorithm_300523'
model_to_load_config_01 = 'specialist_300523_all_data_va_late_fusion'
model_to_load_experiment_02 = 'data_experiments_nn_tuning_02_020623'
model_to_load_config_02 = 'specialist_020623_all_data_va_late_fusion_02'
model_to_load_experiment_03 = 'data_experiments_nn_tuning_03_030623'
model_to_load_config_03 = 'specialist_030623_all_data_va_late_fusion_03'


class EmotionDetectionEnsemble:
    """
    Only Implemented for late fusion models
    """
    def __init__(self, configure_data):
        self.classifier_basic = EmotionDetectionClassifier(configure_data)
        self.configuration = self.classifier_basic.configuration
        self.json_results_information = {}

    def produce_predictions(self):
        """
        This method fills the self.classifier_basic._prediction_probabilities attribute
        for a Late Fusion type
        :return:
        """
        models_ensemble = self.classifier_basic.configuration.ensemble_config
        prediction_from_models_list = []
        _, _, _, _, x_test_video = self.classifier_basic.set_x_y_to_train('x_video')
        _, _, _, _, x_test_audio = self.classifier_basic.set_x_y_to_train('x_audio')

        for model in models_ensemble:
            model_prediction = self.get_model_prediction(model, [x_test_video, x_test_audio])
            prediction_from_models_list.append(model_prediction)
        prediction_probabilities = self.compute_average_predictions(prediction_from_models_list)

        indexes_test = list(self.classifier_basic.dataset.y_test_video.index)
        self.classifier_basic._prediction_probabilities = pd.DataFrame(prediction_probabilities,
                                                                       columns=self.classifier_basic.emotion_from_classifier,
                                                                       index=indexes_test)

    def get_model_prediction(self, model, x_test_list):
        """
        Code for a Late Fusion type
        :param model:
        :param x_test:
        :return:
        """
        model_experiment = model[0]
        model_config = model[1]
        x_test = {
            'x_video': x_test_list[0],
            'x_audio': x_test_list[1]
        }
        dataset = ['x_video', 'x_audio']
        modality_prediction = []
        for modality in dataset:
            model_path = self.classifier_basic.get_model_path(modality, model_experiment, model_config)
            executor = pickle.load(open(model_path, 'rb'))
            prediction = executor.predict_proba(x_test[modality])
            modality_prediction.append(prediction)

            self.classifier_basic.emotion_from_classifier = executor.classes_
        emotions = self.classifier_basic.emotion_from_classifier
        for idx, emotion in enumerate(emotions):
            self.classifier_basic._emotion_class[idx] = emotion

        predictions = self.compute_prediction_late_fusion(modality_prediction[0], modality_prediction[1])
        return predictions

    def compute_prediction_late_fusion(self, video_predictions, audio_predictions):
        # indexes_test = list(self.classifier_basic.dataset.y_test_video.index)
        fusion_by_mean = np.mean(np.array([video_predictions, audio_predictions]), axis=0)
        # prediction_probabilities = pd.DataFrame(fusion_by_mean,
        #                                               columns=self.classifier_basic.emotion_from_classifier,
        #                                               index=indexes_test)
        return fusion_by_mean

    def compute_average_predictions(self, prediction_from_models_list):
        prediction = np.mean(np.array(prediction_from_models_list), axis=0)
        return prediction

    def produce_final_predictions(self):
        self.classifier_basic.produce_final_predictions()

    def show_results(self):
        self.classifier_basic.show_results()

    def format_json_results(self):
        self.classifier_basic.format_json_results()
        self.json_results_information = self.classifier_basic.json_results_information
