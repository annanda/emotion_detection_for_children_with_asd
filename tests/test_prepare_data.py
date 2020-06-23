import unittest
from map_annotation_to_emotion_zones import get_emotion_zone


class MapAnnotationTest(unittest.TestCase):
    def test_mapping_emotion_zones_zero(self):
        emotion = get_emotion_zone(0, 0)
        correct = 'green'
        self.assertEqual(emotion, correct)

    def test_mapping_emotion_zones_orthogonal_1(self):
        emotion = get_emotion_zone(0, 1)
        correct = 'yellow'
        self.assertEqual(emotion, correct)

    def test_mapping_emotion_zones_orthogonal_2(self):
        emotion = get_emotion_zone(1, 0)
        correct = 'green'
        self.assertEqual(emotion, correct)

    def test_mapping_emotion_zones_orthogonal_3(self):
        emotion = get_emotion_zone(0, -1)
        correct = 'blue'
        self.assertEqual(emotion, correct)

    def test_mapping_emotion_zones_orthogonal_4(self):
        emotion = get_emotion_zone(-1, 0)
        correct = 'red'
        self.assertEqual(emotion, correct)
