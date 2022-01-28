# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer_tagrel import RecognizerTagRel

__all__ = [
    'BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer',
    'RecognizerTagRel'
]
