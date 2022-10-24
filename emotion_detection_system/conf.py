import pathlib
import os.path

from decouple import config

ORDER_EMOTIONS = ['blue', 'green', 'red', 'yellow']
PARTICIPANT_NUMBERS = [1, 2, 3, 4]
TOTAL_SESSIONS = ['session_01_01',
                  'session_02_01',
                  'session_02_02',
                  'session_03_01',
                  'session_03_02',
                  'session_04_01',
                  'session_04_02']

main_folder = pathlib.Path(__file__).parent.parent.absolute()

MAIN_FOLDER = config('MAIN_FOLDER', default=main_folder)
emotion_folder = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names')
emotions_file = os.path.join(MAIN_FOLDER, 'labels', 'emotion_zones', 'emotion_names', 'emotions_annotation.csv')

# EMOTION_ANNOTATION_FOLDER = config('EMOTION_ANNOTATION_FOLDER', default=emotion_folder)
# EMOTION_ANNOTATION_FILE = config('EMOTION_ANNOTATION_FILE', default=emotions_file)
# DATASET_VIDEO_AU_FOLDER = config('DATASET_VIDEO_AU', default=os.path.join(MAIN_FOLDER, 'dataset/video/AU'))
# DATASET_VIDEO_FOLDER = config('DATASET_VIDEO_FOLDER', default=os.path.join(MAIN_FOLDER, 'dataset/video'))

DATASET_FOLDER = config('DATASET_FOLDER', default=os.path.join(MAIN_FOLDER, 'dataset'))