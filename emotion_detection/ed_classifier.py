class EmotionDetectionClassifier:
    def __init__(self, configuration):
        self.configuration = configuration
        # From configuration input
        self.session_number = self.configuration['session_number']
        self.all_participant_data = self.configuration['all_participant_data']
        self.dataset_split_type = self.configuration['dataset_split_type']
        self.individual_model = self.configuration['individual_model']
        self.balance_dataset = None
        self.classifier_model = {}
        self.fusion_type = self.configuration['fusion_type']
        self.modalities = []
        self.features_types = {}
        self._setup_modalities_features_type_values()
        # Results of processing
        self.x = None
        self.y = None
        self.x_dev = None
        self.y_dev = None
        self.x_test = None
        self.y_test = None
        self.accuracy = None
        self.confusion_matrix = None

    def _setup_modalities_features_type_values(self):
        """
        """
        self.modalities = list(self.configuration['modalities'].keys())
        features_type = {}

        for modality in self.modalities:
            features_type[modality] = [feature for feature in
                                       list(self.configuration['modalities'][modality]['features_type'].keys())
                                       if self.configuration['modalities'][modality]['features_type'][feature] is True]

        self.features_types = features_type

    def prepare_dataset(self):
        """
        To attribute the correct values for x, y, x_test, y_test, x_dev, y_dev
        """

    def _prepare_dataset_one_participant(self):
        pass

    def train_model_get_predictions(self):
        """
        To train the model and get predictions for x_test
        """

    def _calculate_accuracy(self):
        """
        To calculate accuracy
        """

    def _calculate_confusion_matrix(self):
        """
        To calculate confusion matrix
        """

    def show_results(self):
        """
        To show the results after running an experiment
        :return:
        """
        print(f'Running experiment with configuration:\n')
        print(f'{self.configuration}')
        print(f'######################################\n')
        print(f'Accuracy: {self.accuracy:.4f}')
        print('Confusion matrix: labels=["blue", "green", "yellow", "red"]')
        print(self.confusion_matrix)
