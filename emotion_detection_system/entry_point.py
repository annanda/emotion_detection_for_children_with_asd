from emotion_detection_system.ed_classifier import EmotionDetectionClassifier, EmotionDetectionConfiguration, \
    PrepareDataset

if __name__ == '__main__':
    configure_data = {
        'run_to_all_participants': False,
        # 'sessions_to_consider': ['session_03_01',
        #                          'session_04_02'],
        'participant_number': 3,
        'session_number': 1,
        'all_participant_data': False,
        'dataset_split_type': 'non_sequential',
        'person_independent_model': False,
        'modalities': {
            'video': {
                'features_type': {
                    'AU': False,
                    'appearance': False,
                    'BoVW': False,
                    'geometric': False,
                    'gaze': False,
                    '2d_eye_landmark': False,
                    '3d_eye_landmark': False,
                    'head_pose': False
                },
                'model_for_modality': 'SVM'
            },
            'audio': {
                'feature_group': {
                    'frequency': True,
                    'energy_amplitude': False,
                    'spectral_balance': False,
                    'temporal_features': False
                },
                'all_features_from_group': True,
                'features_type': {
                    'frequency': [],
                    'energy_amplitude': [],
                    'spectral_balance': [],
                    'temporal_features': []
                },
                'feature_level': 'functionals',
            }
        },
        'classifier_model': 'SVM',
        'fusion_type': 'late_fusion',
        'balanced_dataset': False,
        'balance_dataset_technique': '',
    }

    classifier = EmotionDetectionClassifier(configure_data)
    classifier.train_model_produce_predictions()
    classifier.show_results()
    # TODO add to README the hierarchic order of the configuration option for data experiments
    ############################################################
    # Example of configuration
    ############################################################
