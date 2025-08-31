from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
import whisper
import numpy as np
import io
from scipy.io.wavfile import write
import soundfile as sf
from dotenv import load_dotenv
import asyncio
import time
import cv2
import base64
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Try to import a lightweight Hugging Face text-generation pipeline as a fallback
try:
    from transformers import pipeline
    HF_AVAILABLE = True
    hf_generator = None
except Exception:
    HF_AVAILABLE = False
    hf_generator = None

# Try importing the Google GenAI client using either namespace that may be present
# (the package sometimes exposes a `google.genai` namespace or a top-level `genai`).
genai = None
client = None
GENAI_AVAILABLE = False
GENAI_INIT_METHOD = None
_import_err = None
try:
    # preferred import used by some installs/docs
    from google import genai as _genai_pkg
    genai = _genai_pkg
except Exception:
    try:
        import genai as _genai_pkg
        genai = _genai_pkg
    except Exception as e:
        _import_err = e

# If genai was imported, try to initialize a client using env vars if present
if genai:
    try:
        # prefer explicit API key if provided
        api_key = os.environ.get('GENAI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        adc = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if api_key:
            client = genai.Client(api_key=api_key)
            GENAI_AVAILABLE = True
            GENAI_INIT_METHOD = 'api_key'
            logging.info('Initialized GenAI client using GENAI_API_KEY.')
        elif adc:
            # rely on Application Default Credentials file being present
            client = genai.Client()
            GENAI_AVAILABLE = True
            GENAI_INIT_METHOD = 'adc'
            logging.info('Initialized GenAI client using GOOGLE_APPLICATION_CREDENTIALS (ADC).')
        else:
            # Try a default client init (it will raise if no creds configured)
            try:
                client = genai.Client()
                GENAI_AVAILABLE = True
                GENAI_INIT_METHOD = 'default'
                logging.info('Initialized GenAI client using default environment.')
            except Exception as e:
                _import_err = e
                GENAI_AVAILABLE = False
    except Exception as e:
        _import_err = e

if not GENAI_AVAILABLE:
    logging.warning("GenAI client not available: %s", _import_err)

# Initialize voice models
whisper_model = None
tts_engine = None
TTS_AVAILABLE = False

try:
    whisper_model = whisper.load_model("base")
    logging.info("Whisper STT model loaded.")
except Exception as e:
    logging.warning("Failed to load Whisper model: %s", e)

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    logging.info("gTTS engine available.")
except Exception as e:
    TTS_AVAILABLE = False
    logging.warning("Failed to initialize gTTS: %s", e)

# Initialize emotion detection
EMOTION_AVAILABLE = False
face_cascade = None
emotion_model = None

try:
    # Try to load DeepFace for emotion detection
    from deepface import DeepFace
    EMOTION_AVAILABLE = True
    logging.info("DeepFace emotion detection available.")
except Exception as e:
    logging.warning("Failed to initialize DeepFace: %s", e)
    try:
        # Fallback to OpenCV Haar cascades for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logging.info("OpenCV face detection available.")
    except Exception as e2:
        logging.warning("Failed to initialize OpenCV face detection: %s", e2)

router = APIRouter()

class Question(BaseModel):
    question: str
    conversation_history: list[dict] | None = None

class Answer(BaseModel):
    answer: str
    provider: str | None = None

class Status(BaseModel):
    genai_available: bool
    genai_init_method: str | None = None
    hf_available: bool
    stt_available: bool
    tts_available: bool
    emotion_available: bool
    import_error: str | None = None

@router.post("/tutor", response_model=Answer)
def get_tutor_response(question: Question):
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # If GenAI client isn't available, try a local HF fallback before failing
    if not GENAI_AVAILABLE:
        if HF_AVAILABLE:
            # initialize the generator lazily (avoids heavy import at startup)
            global hf_generator
            if hf_generator is None:
                try:
                    hf_generator = pipeline("text-generation", model="gpt2")
                except Exception as _init_err:
                    logging.exception("Failed to initialize HF generator: %s", _init_err)
                    hf_generator = None

            if hf_generator is not None:
                try:
                    # short generation for concise answers
                    gen = hf_generator(question.question, max_length=150, do_sample=True, top_p=0.95, temperature=0.7)
                    answer = gen[0].get("generated_text") if isinstance(gen, list) else str(gen)
                    return Answer(answer=answer, provider="hf")
                except Exception as e:
                    logging.exception("HF generation failed")
                    raise HTTPException(status_code=500, detail=f"Local HF generation failed: {str(e)}")

        # if we reach here, neither GenAI nor HF is usable
        raise HTTPException(
            status_code=503,
            detail=(
                "GenAI client is not installed or failed to initialize. "
                "Install the official GenAI Python package (pip install genai) and its build deps (tiktoken may require Rust), "
                "or enable the local Hugging Face fallback by installing transformers and torch, then restart the server."
            ),
        )

    # Generate content using Google GenAI with strict clean, minimal prompt engineering
    try:
        # Build conversation context
        context = ""
        if question.conversation_history and len(question.conversation_history) > 0:
            context = "\n\nPrevious conversation:\n"
            for msg in question.conversation_history[-6:]:  # Keep last 6 messages for context
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content.strip():  # Only include non-empty messages
                    context += f"{role.title()}: {content}\n"

        logging.info(f"Question: {question.question}")
        logging.info(f"Conversation history length: {len(question.conversation_history) if question.conversation_history else 0}")
        logging.info(f"Context built: {bool(context.strip())}")

        prompt = f"""You are an expert AI tutor specializing in educational support and learning assistance. Your role is to help students understand concepts, provide clear explanations, and guide their learning journey.

{context}
Current question: {question.question}

As an AI tutor, you should:
- Provide clear, educational explanations
- Use examples and analogies when helpful
- Build upon previous conversation context
- Ask follow-up questions to deepen understanding
- Stay focused on educational topics and learning
- Be encouraging and supportive
- Keep responses comprehensive but not overwhelming
- Format your responses using markdown for better readability (use **bold**, *italic*, lists, etc.)

Answer the student's question thoughtfully and educationally."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.1,
                "top_p": 0.5,
                "max_output_tokens": 1500,  # Significantly increased for comprehensive educational responses
            }
        )

        # Simplified response extraction
        answer = None
        if hasattr(response, 'text') and response.text:
            answer = response.text
            logging.info(f"Response extracted from text attribute, length: {len(answer)}")
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                answer = part.text
                                logging.info(f"Response extracted from parts, length: {len(answer)}")
                                break
                        if answer:
                            break
                    elif hasattr(content, 'text') and content.text:
                        answer = content.text
                        logging.info(f"Response extracted from content.text, length: {len(answer)}")
                        break

        # Log the final answer length
        if answer:
            logging.info(f"Final answer length: {len(answer)}")
        else:
            logging.warning("No answer extracted from response")

    except Exception as e:
        logging.exception("Error generating response from GenAI")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    return Answer(answer=answer, provider="genai")

@router.post("/stt", response_model=dict)
async def speech_to_text(audio: UploadFile = File(...)):
    if not whisper_model:
        raise HTTPException(status_code=503, detail="STT model not available.")

    try:
        # Read audio file
        audio_data = await audio.read()

        # Try to use soundfile, fallback to basic numpy if not available
        try:
            import soundfile as sf
            audio_np, sr = sf.read(io.BytesIO(audio_data))
        except ImportError:
            # Fallback: assume raw PCM data
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            sr = 16000  # Assume 16kHz

        # Ensure correct dtype and sample rate
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sr != 16000:
            try:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            except ImportError:
                # Simple resampling fallback
                if sr > 16000:
                    audio_np = audio_np[::sr//16000]
                sr = 16000

        # Transcribe
        result = whisper_model.transcribe(audio_np)
        transcription = result["text"].strip()

        return {"transcription": transcription}
    except Exception as e:
        logging.exception("STT error")
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@router.post("/tts")
def text_to_speech(text: str):
    if not TTS_AVAILABLE:
        # Return a simple beep sound as fallback
        sample_rate = 22050
        duration = 0.5
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(frequency * 2 * np.pi * t)
        beep = (beep * 32767).astype(np.int16)

        buffer = io.BytesIO()
        write(buffer, sample_rate, beep)
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="audio/wav")

    try:
        from gtts import gTTS
        import tempfile
        import os

        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en', slow=False)

        # Create a temporary MP3 file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name

        # Save the speech to file
        tts.save(temp_path)

        # Read the generated audio file
        with open(temp_path, 'rb') as f:
            audio_data = f.read()

        # Clean up temp file
        os.unlink(temp_path)

        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        logging.exception("TTS error")
        # Fallback to beep
        sample_rate = 22050
        duration = 0.5
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(frequency * 2 * np.pi * t)
        beep = (beep * 32767).astype(np.int16)

        buffer = io.BytesIO()
        write(buffer, sample_rate, beep)
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="audio/wav")

@router.post("/tts-stream")
async def text_to_speech_stream(text: str):
    """
    Streaming TTS endpoint using gTTS.
    Generates MP3 audio and streams it in chunks for faster perceived playback.
    """
    if not TTS_AVAILABLE:
        # Fallback to mock streaming
        async def beep_generator():
            sample_rate = 22050
            duration = 0.5
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = np.sin(frequency * 2 * np.pi * t)
            beep = (beep * 32767).astype(np.int16)

            yield sample_rate.to_bytes(4, 'little')
            yield beep.tobytes()

        return StreamingResponse(
            beep_generator(),
            media_type="application/octet-stream",
            headers={"X-Format": "pcm16"}
        )

    async def audio_generator():
        try:
            from gtts import gTTS
            import tempfile
            import os

            # Generate speech to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name

            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_path)

            # Read the MP3 file
            with open(temp_path, 'rb') as f:
                mp3_data = f.read()

            # Clean up temp file
            os.unlink(temp_path)

            # For streaming, we'll send the MP3 data as-is
            # The frontend will handle MP3 playback
            if len(mp3_data) > 0:
                chunk_size = 4096  # Larger chunks for MP3
                for i in range(0, len(mp3_data), chunk_size):
                    chunk = mp3_data[i:i + chunk_size]
                    yield chunk
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

        except Exception as e:
            logging.exception("Streaming TTS error")
            # Fallback to simple tone
            sample_rate = 22050
            duration = 0.3
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(frequency * 2 * np.pi * t)
            tone = (tone * 32767).astype(np.int16)

            yield sample_rate.to_bytes(4, 'little')
            yield tone.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers={"Content-Type": "audio/mpeg"}
    )

@router.post("/test-image")
async def test_image_processing(image_data: dict):
    """
    Test endpoint to validate image processing without emotion detection.
    Useful for debugging image format issues.
    """
    try:
        # Decode base64 image
        image_b64 = image_data.get('image', '')
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided.")

        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        # Validate base64 data
        if not image_b64:
            raise HTTPException(status_code=400, detail="Invalid image data: no base64 content after data URL prefix.")

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

        # Check if image data is empty
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Image data is empty.")

        # Try to open image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify image was loaded correctly
            image.verify()
            width, height = image.size
            mode = image.mode
            image.close()

            return {
                "success": True,
                "image_info": {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": len(image_bytes)
                },
                "message": "Image processed successfully"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format or corrupted image data: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Unexpected error in test_image_processing")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/emotion")
async def detect_emotion(image_data: dict):
    """
    Detect emotion from image data sent as base64 string.
    Expects JSON with 'image' field containing base64 encoded image.
    """
    if not EMOTION_AVAILABLE and face_cascade is None:
        raise HTTPException(status_code=503, detail="Emotion detection not available.")

    try:
        # Decode base64 image
        image_b64 = image_data.get('image', '')
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided.")

        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        # Validate base64 data
        if not image_b64:
            raise HTTPException(status_code=400, detail="Invalid image data: no base64 content after data URL prefix.")

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

        # Check if image data is empty
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Image data is empty.")

        # Try to open image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify image was loaded correctly
            image.verify()  # This will raise an exception if the image is corrupted
            image.close()  # Close the verified image

            # Re-open for processing (PIL keeps file open after verify)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format or corrupted image data: {str(e)}")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image)

        # Validate image dimensions
        if image_np.size == 0:
            raise HTTPException(status_code=400, detail="Image has no data or invalid dimensions.")

        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        emotion = "neutral"  # Default
        confidence = 0.0

        # Short-circuit for test/mock mode to avoid heavy ML deps during CI/tests
        if os.environ.get('USE_MOCK_DETECTOR') == '1':
            # Use a lightweight deterministic in-process detector for fast CI/local tests
            def dummy_detect_emotion(img_np):
                # img_np expected as BGR or grayscale numpy array
                try:
                    # compute approximate luminance
                    if img_np.ndim == 3 and img_np.shape[2] >= 3:
                        b = img_np[..., 0].astype(float)
                        g = img_np[..., 1].astype(float)
                        r = img_np[..., 2].astype(float)
                        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    else:
                        lum = img_np.astype(float)
                    avg = float(lum.mean())
                except Exception:
                    avg = 0.0

                # simple heuristic mapping
                if avg > 150:
                    return ("happy", 0.9)
                if avg > 90:
                    return ("neutral", 0.6)
                if avg > 40:
                    return ("sad", 0.45)
                return ("no_face", 0.0)

            emo_label, emo_conf = dummy_detect_emotion(image_np)
            return {
                "emotion": str(emo_label),
                "confidence": float(emo_conf),
                "timestamp": float(time.time())
            }

        if EMOTION_AVAILABLE:
            logging.info("Using DeepFace for emotion detection")
            try:
                # Use DeepFace for emotion detection
                result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
                logging.info(f"DeepFace result: {result}")
                logging.info(f"Image shape: {image_np.shape}")
                logging.info(f"Image dtype: {image_np.dtype}")
                logging.info(f"Image min/max values: {image_np.min()}/{image_np.max()}")
                # DeepFace may return a dict or a list containing a dict depending on version/options
                res_obj = None
                if isinstance(result, list) and len(result) > 0:
                    res_obj = result[0]
                elif isinstance(result, dict):
                    res_obj = result

                if res_obj:
                    # dominant_emotion is usually a string; emotion scores may be numpy types
                    try:
                        emotion = str(res_obj.get('dominant_emotion', emotion))
                    except Exception:
                        emotion = str(emotion)

                    try:
                        raw_conf = res_obj.get('emotion', {}).get(emotion, None)
                        if raw_conf is None:
                            # try other likely keys
                            raw_conf = res_obj.get('confidence') or res_obj.get('score') or raw_conf
                        # Convert numpy types to native float
                        if raw_conf is not None:
                            confidence = float(raw_conf)
                    except Exception:
                        # fallback - keep previous default
                        confidence = float(confidence)
            except Exception as e:
                logging.warning("DeepFace emotion detection failed: %s", e)
                emotion = "unknown"
        elif face_cascade is not None:
            try:
                # Fallback to basic face detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    emotion = "face_detected"
                    confidence = 0.5
                else:
                    emotion = "no_face"
                    confidence = 0.0
            except Exception as e:
                logging.warning("OpenCV face detection failed: %s", e)
                emotion = "error"

        # Ensure returned values are JSON-serializable native Python types
        try:
            resp = {
                "emotion": str(emotion),
                "confidence": float(confidence),
                "timestamp": float(time.time())
            }
        except Exception:
            # As a last resort, coerce via string/zero values
            resp = {
                "emotion": str(emotion),
                "confidence": float(0.0),
                "timestamp": float(time.time())
            }

        return resp

    except Exception as e:
        logging.exception("Emotion detection error")
        raise HTTPException(status_code=500, detail=f"Emotion detection failed: {str(e)}")

@router.get("/status", response_model=Status)
def get_status():
    err_str = None
    if _import_err:
        try:
            err_str = str(_import_err)
        except Exception:
            err_str = "<unprintable import error>"

    return Status(
        genai_available=GENAI_AVAILABLE,
        genai_init_method=GENAI_INIT_METHOD,
        hf_available=HF_AVAILABLE,
        stt_available=whisper_model is not None,
        tts_available=TTS_AVAILABLE,
        emotion_available=EMOTION_AVAILABLE or (face_cascade is not None),
        import_error=err_str,
    )
