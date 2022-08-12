from emotion_detection.ed_classifier import EmotionDetectionClassifier

if __name__ == '__main__':
    configure_data = {
        'session_number': 'session_02_02',
        'all_participant_data': False,
        'dataset_split_type': 'non_sequential',
        'individual_model': True,
        'modalities': {
            'video': {
                'features_type': {'AU': True, 'appearance': True, 'BoVW': False, 'geometric': False,
                                  'gaze': False,
                                  '2d_eye_landmark': False, '3d_eye_landmark': False, 'head_pose': False},
                'model': 'SVM'
            }
        },
        'fusion_type': 'late_fusion',
    }

    classifier = EmotionDetectionClassifier(configure_data)
    # classifier.show_results()
    # classifier._setup_values()
