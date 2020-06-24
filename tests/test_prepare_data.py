import unittest
from prepare_data.map_annotation_to_emotion_zones import get_emotion_zone


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

    def test_mapping_emotion_zones_green(self):
        emotion = get_emotion_zone(0.5, -0.5)
        self.assertEqual(emotion, 'green')

    def test_mapping_emotion_zones_yellow(self):
        emotion = get_emotion_zone(0.5, 0.5)
        self.assertEqual(emotion, 'yellow')

    def test_mapping_emotion_zones_blue(self):
        emotion = get_emotion_zone(-0.5, -0.5)
        self.assertEqual(emotion, 'blue')

    def test_mapping_emotion_zones_red(self):
        emotion = get_emotion_zone(-0.5, 0.5)
        self.assertEqual(emotion, 'red')

    def test_mapping_emotion_zones_random(self):
        emotion = get_emotion_zone(0.238405, 0.3618)
        self.assertEqual(emotion, 'yellow')
