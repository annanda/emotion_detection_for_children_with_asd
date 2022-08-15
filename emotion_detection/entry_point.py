from emotion_detection.ed_classifier import EmotionDetectionClassifier

configure_data = {
    'session_number': 'session_02_02',
    'all_participant_data': False,
    'dataset_split_type': 'non_sequential',
    'individual_model': True,
    'modalities': {
        'video': {
            'features_type': {
                'AU': True,
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
            'features_type': {
                'BoAW': True,
                'DeepSpectrum': False,
                'eGeMAPSfunct': False
                # eGeMAPSfunct feature_type can only be used alone
            },
            'model_for_modality': 'Other'
        },
    },
    'classifier_model': 'SVM',
    'fusion_type': 'late_fusion',
    'balanced_dataset': False,
    'balance_dataset_technique': '',
}

if __name__ == '__main__':
    classifier = EmotionDetectionClassifier(configure_data)
    # classifier.show_results()
    # classifier._setup_values()

    ############################################################
    # Example of configuration
    ############################################################

    # configure_data = {
    #     'session_number': 'session_02_02',
    #     'all_participant_data': False,
    #     'dataset_split_type': 'non_sequential',
    #     'individual_model': True,
    #     'modalities': {
    #         'video': {
    #             'features_type': {
    #                 'AU': True,
    #                 'appearance': False,
    #                 'BoVW': False,
    #                 'geometric': False,
    #                 'gaze': False,
    #                 '2d_eye_landmark': False,
    #                 '3d_eye_landmark': False,
    #                 'head_pose': False
    #             },
    #             'model': 'SVM'
    #         },
    #         'audio': {
    #             'features_type': {
    #                 'BoAW': True,
    #                 'DeepSpectrum': False,
    #                 'eGeMAPSfunct': False
    #                 # eGeMAPSfunct feature_type can only be used alone
    #             },
    #             'model': 'SVM'
    #         },
    #     },
    #     'fusion_type': 'late_fusion',
    # }
