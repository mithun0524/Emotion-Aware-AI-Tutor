import base64
import io
from PIL import Image
import requests


def test_emotion_endpoint():
    # create a small image
    img = Image.new('RGB', (32, 32), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

    resp = requests.post('http://127.0.0.1:8000/api/emotion', json={'image': data}, timeout=10)
    assert resp.status_code == 200
    j = resp.json()
    assert 'emotion' in j
    assert 'confidence' in j
    assert isinstance(j['emotion'], str)
    assert isinstance(j['confidence'], (int, float))
*** End Patch
