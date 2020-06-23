"""
Script to map values of arousal and valence into the 4 emotion zones.
"""
import numpy as np

emotion_zone = {
    'blue': np.array([1, 0, 0, 0]),
    'green': np.array([0, 1, 0, 0]),
    'yellow': np.array([0, 0, 1, 0]),
    'red': np.array([0, 0, 0, 1])
}


def get_emotion_zone(valence, arousal):
    """
    each emotion zone is represented by an np array of 4 dimension
    on each orthogonal the emotion zone is the one located on the right of the orthogonal in question.
    the point 0,0 corresponds to green zone.
    """
    emotion = None
    if arousal == 0 and valence == 0:
        emotion = 'green'
    elif arousal > 0:
        if valence >= 0:
            emotion = 'yellow'
        else:
            emotion = 'red'
    elif arousal < 0:
        if valence > 0:
            emotion = 'green'
        else:
            emotion = 'blue'
    elif arousal == 0:
        if valence > 0:
            emotion = 'green'
        else:
            emotion = 'red'
    elif valence == 0:
        if arousal > 0:
            emotion = 'yellow'
        else:
            emotion = 'blue'
    return emotion
