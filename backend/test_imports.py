try:
    from google import genai
    print('Google GenAI import successful')
except Exception as e:
    print('Import error:', e)

try:
    import whisper
    print('Whisper import successful')
except Exception as e:
    print('Whisper import error:', e)

try:
    from gtts import gTTS
    print('gTTS import successful')
except Exception as e:
    print('gTTS import error:', e)

try:
    from deepface import DeepFace
    print('DeepFace import successful')
except Exception as e:
    print('DeepFace import error:', e)

try:
    import cv2
    print('OpenCV import successful')
except Exception as e:
    print('OpenCV import error:', e)

try:
    import tensorflow
    print('TensorFlow import successful')
except Exception as e:
    print('TensorFlow import error:', e)
