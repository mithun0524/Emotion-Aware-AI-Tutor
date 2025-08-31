/* eslint-disable */
'use client';

import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

// Web Speech API types
declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onend: ((this: SpeechRecognition, ev: Event) => any) | null;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

declare var SpeechRecognition: {
  prototype: SpeechRecognition;
  new(): SpeechRecognition;
};

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

declare var SpeechRecognition: {
  prototype: SpeechRecognition;
  new(): SpeechRecognition;
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Status {
  genai_available: boolean;
  stt_available: boolean;
  tts_available: boolean;
  emotion_available: boolean;
  import_error: string | null;
}

const API_BASE = 'http://127.0.0.1:8000';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recognition, setRecognition] = useState<SpeechRecognition | null>(null);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [currentEmotion, setCurrentEmotion] = useState<string>('neutral');
  const [emotionConfidence, setEmotionConfidence] = useState<number>(0);
  const [webcamEnabled, setWebcamEnabled] = useState(false);
  const [emotionInterval, setEmotionInterval] = useState<NodeJS.Timeout | null>(null);
  const [cameraStatus, setCameraStatus] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Inline SVG icons (more polished)
  const TutorIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M12 2L2 7l10 5 10-5-10-5z" fill="#111827" opacity="0.9" />
      <path d="M6.5 8.5v3.5c0 3 3 5.5 5.5 5.5s5.5-2.5 5.5-5.5V8.5" stroke="#f9fafb" strokeWidth="0.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 13.5v7" stroke="#f9fafb" strokeWidth="0.9" strokeLinecap="round" />
    </svg>
  );

  const LightbulbIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M9 18h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 2a7 7 0 00-4 12c0 1.5.6 2.9 1.6 3.9L10 20h4l.4-1.1c1-1 1.6-2.4 1.6-3.9A7 7 0 0012 2z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="10" y="20" width="4" height="1.5" fill="currentColor" />
    </svg>
  );

  const RocketIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M5 19c1-2 4-4 7-4s6 2 7 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 2s3 1 5 3 3 5 3 5-2 1-5 1-5-1-5-1 1-3 0-6z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="9.5" cy="10.5" r="1" fill="currentColor" />
    </svg>
  );

  const MicIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M12 14a3 3 0 003-3V6a3 3 0 00-6 0v5a3 3 0 003 3z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M19 11v1a7 7 0 01-14 0v-1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <path d="M12 19v3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );

  const BulletIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="4" cy="4" r="3" fill="currentColor" />
    </svg>
  );

  // Emotion vector icons (simple, clean, aesthetic)
  const HappyIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M8 14c1.2 1.2 2.8 2 4 2s2.8-.8 4-2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9 10h.01M15 10h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const SadIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M8 16c1.2-1.2 2.8-2 4-2s2.8.8 4 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9 10h.01M15 10h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const AngryIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M8 10s1-2 4-2 4 2 4 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9 15h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M8.5 9l-1-1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M15.5 9l1-1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const NeutralIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M9 14h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M9 10h.01M15 10h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const SurpriseIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <circle cx="12" cy="13" r="2" fill="currentColor" />
      <path d="M9 9h.01M15 9h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const FearIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M9 9h.01M15 9h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M8 15c1.2-2 5.8-2 7 0" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );

  const DisgustIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.5" fill="none" />
      <path d="M8 15c1.5-1 4.5-1 6 0" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M9 9h.01M15 9h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );

  const NoFaceIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M12 12a3 3 0 100-6 3 3 0 000 6z" fill="currentColor" />
      <path d="M4 20c1-4 7-6 8-6s7 2 8 6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );

  const DetectingIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.2" fill="none" />
      <path d="M12 6v6l3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );

  const CameraIcon = ({ className = '' }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <rect x="3" y="6" width="18" height="12" rx="2" stroke="currentColor" strokeWidth="1.4" fill="none" />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.4" fill="none" />
      <path d="M7 6l1.5-2h7L17 6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );

  useEffect(() => {
    // Cleanup webcam on unmount
    return () => {
      if (emotionInterval) {
        clearInterval(emotionInterval);
      }
      stopWebcam();
    };
  }, []);

  useEffect(() => {
    // Load conversation from localStorage
    const savedMessages = localStorage.getItem('chatMessages');
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(parsedMessages);
      } catch (error) {
        console.error('Failed to load saved messages:', error);
      }
    }

    // Initialize speech recognition
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognitionInstance = new SpeechRecognition();
        recognitionInstance.continuous = false;
        recognitionInstance.interimResults = false;
        recognitionInstance.lang = 'en-US';

        recognitionInstance.onstart = () => {
          setIsRecording(true);
        };

        recognitionInstance.onend = () => {
          setIsRecording(false);
        };

        recognitionInstance.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          setInput(prev => prev + transcript);
        };

        recognitionInstance.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
        };

        setRecognition(recognitionInstance);
        setSpeechSupported(true);
      } else {
        setSpeechSupported(false);
        console.warn('Speech recognition not supported in this browser');
      }
    }
  }, []);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'end'
        });
      }
    };

    // Small delay to ensure DOM has updated
    const timeoutId = setTimeout(scrollToBottom, 100);

    return () => clearTimeout(timeoutId);
  }, [messages]);

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chatMessages', JSON.stringify(messages));
    }
  }, [messages]);

  // Check camera support on component mount
  useEffect(() => {
    console.log('🔍 Checking camera support...');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn('🔍 Camera not supported in this browser');
    } else {
      console.log('🔍 Camera supported');

      // Check if we already have permission
      navigator.permissions.query({ name: 'camera' as PermissionName }).then(result => {
        console.log('🔍 Camera permission status:', result.state);
      }).catch(err => {
        console.log('🔍 Could not check camera permission:', err);
      });
    }
  }, []);

  // Cleanup webcam on unmount (run once)
  useEffect(() => {
    return () => {
      if (emotionInterval) {
        clearInterval(emotionInterval);
      }
      stopWebcam();
    };
  }, []);

  const clearConversation = () => {
    setMessages([]);
    localStorage.removeItem('chatMessages');
  };

  const startSpeechRecognition = () => {
    if (recognition && !isRecording) {
      try {
        recognition.start();
      } catch (error) {
        console.error('Failed to start speech recognition:', error);
      }
    }
  };

  const stopSpeechRecognition = () => {
    if (recognition && isRecording) {
      recognition.stop();
    }
  };

  const toggleSpeechRecognition = () => {
    if (isRecording) {
      stopSpeechRecognition();
    } else {
      startSpeechRecognition();
    }
  };

  const startWebcam = async () => {
    console.log('📹 Starting webcam...');
    console.log('📹 Checking camera permissions...');

    try {
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('📹 getUserMedia not supported');
        alert('Camera not supported in this browser');
        return;
      }

      console.log('📹 Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: 'user',
          frameRate: { ideal: 30, min: 15 }
        }
      });

      console.log('📹 Webcam stream obtained:', stream);
      console.log('📹 Stream tracks:', stream.getTracks().length);

      if (videoRef.current) {
        console.log('📹 Setting up video element...');
        videoRef.current.srcObject = stream;

        // Ensure video is set to autoplay and muted (required for autoplay in modern browsers)
        videoRef.current.muted = true;
        videoRef.current.autoplay = true;

        // Wait for video to be ready
        await new Promise((resolve) => {
          const checkVideoReady = () => {
            if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0 && videoRef.current.readyState >= 2) {
              console.log('📹 Video dimensions ready:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight, 'ReadyState:', videoRef.current.readyState);
              resolve(true);
            } else {
              console.log('📹 Waiting for video... current state:', videoRef.current?.readyState);
              setTimeout(checkVideoReady, 100);
            }
          };
          checkVideoReady();
        });

        setWebcamEnabled(true);
        setCameraStatus('Camera active - detecting emotions...');
        console.log('📹 Webcam enabled, starting emotion detection interval');

        // Add initial instruction
        setTimeout(() => {
          if (currentEmotion === 'neutral' || currentEmotion === 'no_face') {
            setCameraStatus('Tip: Ensure good lighting and center your face in the camera for better detection');
          }
        }, 3000);

        // Start emotion detection with a slight delay to ensure first frame is ready
        setTimeout(() => {
          const interval = setInterval(() => {
            // Call capture regardless; captureAndAnalyzeEmotion will validate video state itself.
            // This avoids a stale-closure problem where `webcamEnabled` may be false inside the timer.
            // Mark as detecting immediately for UI feedback.
            try {
              setCurrentEmotion('detecting');
            } catch (e) {
              // ignore
            }
            captureAndAnalyzeEmotion();
          }, 2000); // Analyze every 2 seconds
          setEmotionInterval(interval);
        }, 500); // Wait 500ms after video is ready
      } else {
        console.error('📹 Video ref is null');
        alert('Video element not found');
      }
    } catch (error) {
      console.error('📹 Error accessing webcam:', error);

      // Provide specific error messages
      if (error instanceof DOMException) {
        switch (error.name) {
          case 'NotAllowedError':
            alert('Camera permission denied. Please allow camera access and try again.');
            break;
          case 'NotFoundError':
            alert('No camera found. Please connect a camera and try again.');
            break;
          case 'NotReadableError':
            alert('Camera is already in use by another application.');
            break;
          case 'OverconstrainedError':
            alert('Camera does not support the requested resolution.');
            break;
          default:
            alert(`Camera error: ${error.message}`);
        }
      } else {
        alert('Could not access webcam. Please check permissions.');
      }
      setCameraStatus('');
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      videoRef.current.muted = false; // Reset muted state
      setWebcamEnabled(false);
      setCameraStatus('');

      if (emotionInterval) {
        clearInterval(emotionInterval);
        setEmotionInterval(null);
      }
    }
    // Also clear interval if it exists but video ref is not available
    if (emotionInterval) {
      clearInterval(emotionInterval);
      setEmotionInterval(null);
    }
  };

  const captureAndAnalyzeEmotion = async (retryCount = 0) => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('Video or canvas ref not available');
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) {
      console.log('Canvas context not available');
      return;
    }

    // Check if video has a valid stream
    if (!video.srcObject) {
      console.log('Video has no source stream');
      return;
    }

    // Check if video is actually playing
    if (video.paused || video.ended || video.readyState < 2) {
      console.log('Video is not in a playable state:', {
        paused: video.paused,
        ended: video.ended,
        readyState: video.readyState
      });
      return;
    }

    // Validate video dimensions with more tolerance
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('Video dimensions not ready:', video.videoWidth, video.videoHeight);
      // If this is the first attempt and video seems temporarily unavailable, retry once
      if (retryCount === 0) {
        console.log('Retrying emotion capture in 500ms...');
        setTimeout(() => captureAndAnalyzeEmotion(1), 500);
      }
      return;
    }

    // Additional validation: check if dimensions are reasonable
    if (video.videoWidth < 100 || video.videoHeight < 100) {
      console.log('Video dimensions too small:', video.videoWidth, 'x', video.videoHeight);
      return;
    }

    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Validate canvas has content by checking a few pixels
    const testData = context.getImageData(0, 0, Math.min(10, canvas.width), Math.min(10, canvas.height));
    const pixels = testData.data;
    // Check if any pixel has meaningful color
    const hasContent = pixels.some(p => p > 10);

    // Compute average luminance and uniformity for a small tile to detect covered/black screens
    let totalLum = 0;
    let minLum = 255;
    let maxLum = 0;
    const pixelCount = Math.floor(pixels.length / 4);
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      // Rec. 709 luminance
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      totalLum += lum;
      if (lum < minLum) minLum = lum;
      if (lum > maxLum) maxLum = lum;
    }
    const avgLum = pixelCount > 0 ? totalLum / pixelCount : 0;

    // If there is effectively no content or the frame is very dark or nearly uniform, skip analysis
    const DARK_LUMINANCE_THRESHOLD = 8; // avg luminance below this -> probably covered/black
    const UNIFORMITY_THRESHOLD = 6; // max-min below this -> mostly uniform (covered/static)

    if (!hasContent || avgLum < DARK_LUMINANCE_THRESHOLD || (maxLum - minLum) < UNIFORMITY_THRESHOLD) {
      console.log('Skipping analysis: frame too dark or uniform', { avgLum, minLum, maxLum });
      setLastEmotionRaw({ note: 'frame_too_dark_or_uniform', avgLum, minLum, maxLum });
      setCameraStatus('No valid camera frame (dark or covered)');
      // Preserve previous emotion but mark confidence 0 to avoid misleading values
      setCurrentEmotion('no_frame');
      setEmotionConfidence(0);
      return;
    }

    // Convert to base64
    const base64Data = canvas.toDataURL('image/jpeg', 0.8);

    // Validate base64 data
    if (!base64Data || base64Data.length < 100) { // Minimum length for a valid image
      console.log('Generated base64 data appears invalid or too short');
      return;
    }

    try {
      console.log(`Attempting emotion detection (attempt ${retryCount + 1})... Image size: ${base64Data.length} chars, Dimensions: ${video.videoWidth}x${video.videoHeight}`);
      // provide immediate UI feedback
      setCurrentEmotion('detecting');
      setCameraStatus('Processing...');
      const response = await fetch(`${API_BASE}/api/emotion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Data }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Emotion detection successful:', result);
        console.log('Raw emotion from backend:', result.emotion);
        console.log('Confidence:', result.confidence);
        // Keep raw response for debugging
        setLastEmotionRaw(result);

        // Robustly extract emotion label from several possible keys
        const rawEmotion = (result && (result.emotion || result.label || result.prediction || null)) as string | null;

        // Try several fields for confidence/score and normalize to 0..1
        let rawConf: any = undefined;
        if (result) {
          rawConf = result.confidence ?? result.conf ?? result.score ?? result.probability ?? undefined;
          // If there is a per-class scores object, try to pick the score for the chosen label
          if ((rawConf === undefined || rawConf === null) && result.scores && rawEmotion && typeof result.scores === 'object') {
            rawConf = result.scores[rawEmotion];
          }
        }

        let confNum = 0;
        if (typeof rawConf === 'number') {
          confNum = rawConf;
        } else if (typeof rawConf === 'string') {
          const parsed = Number(rawConf);
          confNum = isNaN(parsed) ? 0 : parsed;
        }

        // If value appears to be a percentage (e.g. 75), convert to 0..1
        if (confNum > 1) confNum = confNum / 100;

        setCurrentEmotion(rawEmotion || (result && result.emotion ? result.emotion : 'neutral'));
        setEmotionConfidence(confNum || 0);

        // Provide helpful feedback based on the result
        if (rawEmotion === 'no_face') {
          setCameraStatus('No face detected - please ensure your face is clearly visible in the camera');
        } else if (rawEmotion === 'neutral' && confNum < 0.3) {
          setCameraStatus('Low confidence detection - try better lighting or center your face');
        } else {
          setCameraStatus('');
        }
      } else {
        const errorText = await response.text();
        console.error(`Emotion detection failed with status ${response.status}:`, errorText);
        setLastEmotionRaw({ status: response.status, error: errorText });

        if (response.status === 503) {
          // Service unavailable - emotion detection not configured
          setCurrentEmotion('emotion_unavailable');
        } else if (response.status >= 500) {
          // Server error - retry once
          if (retryCount < 1) {
            console.log('Retrying emotion detection in 2 seconds...');
            setTimeout(() => captureAndAnalyzeEmotion(retryCount + 1), 2000);
            return;
          }
          setCurrentEmotion('server_error');
        } else {
          // Other client errors
          setCurrentEmotion('backend_unavailable');
        }
        setEmotionConfidence(0);
      }
    } catch (error) {
      console.error('Network error during emotion detection:', error);
      setLastEmotionRaw({ error: String(error) });

      // Check if it's a network error (Failed to fetch)
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        console.log('Network connection issue detected');

        // Retry once after a short delay
        if (retryCount < 1) {
          console.log('Retrying emotion detection in 3 seconds...');
          setTimeout(() => captureAndAnalyzeEmotion(retryCount + 1), 3000);
          return;
        }
      }

      // Backend not available, show fallback status
      setCurrentEmotion('backend_unavailable');
      setEmotionConfidence(0);
    }
  };

  const [emotionHistory, setEmotionHistory] = useState<Array<{emotion: string, confidence: number, timestamp: Date}>>([]);
  const [lastEmotionRaw, setLastEmotionRaw] = useState<any>(null);
  const [smoothedEmotion, setSmoothedEmotion] = useState<string | null>(null);
  const [smoothedConfidence, setSmoothedConfidence] = useState<number>(0);

  // Update emotion history when emotion changes
  useEffect(() => {
    if (currentEmotion && currentEmotion !== 'detecting' && currentEmotion !== 'backend_unavailable' && currentEmotion !== 'server_error' && currentEmotion !== 'emotion_unavailable') {
      setEmotionHistory(prev => {
        const newHistory = [...prev, {
          emotion: currentEmotion,
          confidence: emotionConfidence,
          timestamp: new Date()
        }];
        // Keep only last 10 emotions
        return newHistory.slice(-10);
      });
    }
  }, [currentEmotion, emotionConfidence]);

  // Compute a smoothed emotion and confidence to reduce jitter.
  // Uses a short window average and exponential smoothing across updates.
  useEffect(() => {
    if (!emotionHistory || emotionHistory.length === 0) return;

    const windowSize = 5;
    const recent = emotionHistory.slice(-windowSize);

    // Majority vote for emotion label
    const counts: Record<string, number> = {};
    for (const e of recent) {
      counts[e.emotion] = (counts[e.emotion] || 0) + 1;
    }
    let majority = recent[recent.length - 1].emotion;
    let maxCount = 0;
    for (const k in counts) {
      if (counts[k] > maxCount) {
        maxCount = counts[k];
        majority = k;
      }
    }

    // Normalize confidences to 0..1 and average
    const confs = recent.map(r => {
      let c = typeof r.confidence === 'number' ? r.confidence : Number(r.confidence) || 0;
      if (c > 1) c = c / 100; // convert percent to 0..1
      return Math.max(0, Math.min(1, c));
    });
    const avg = confs.reduce((a, b) => a + b, 0) / confs.length;

    // Exponential smoothing
    const alpha = 0.4;
    setSmoothedConfidence(prev => {
      const smoothed = prev === 0 ? avg : (prev * (1 - alpha) + avg * alpha);
      return smoothed;
    });

    setSmoothedEmotion(majority);
  }, [emotionHistory]);

  const toggleWebcam = () => {
    console.log('🎥 Toggle webcam called, current state:', webcamEnabled);
    console.log('🎥 Video ref exists:', !!videoRef.current);
    console.log('🎥 Canvas ref exists:', !!canvasRef.current);

    setCameraStatus('Processing...');

    if (webcamEnabled) {
      console.log('🎥 Stopping webcam...');
      setCameraStatus('Stopping camera...');
      stopWebcam();
      setCameraStatus('');
    } else {
      console.log('🎥 Starting webcam...');
      // First enable the DOM for the video element so `videoRef.current` exists.
      // setState is async, so wait a frame before starting the actual camera.
      setCameraStatus('Starting camera...');
      setWebcamEnabled(true);
      // Wait for the next paint to ensure the video element is mounted and ref is attached.
      requestAnimationFrame(() => {
        // small extra delay to allow React to attach refs
        setTimeout(() => {
          startWebcam();
        }, 50);
      });
    }
  };

  const [isTestingConnection, setIsTestingConnection] = useState(false);

  const testBackendConnection = async () => {
    setIsTestingConnection(true);
    try {
      console.log('Testing backend connection...');
      const response = await fetch(`${API_BASE}/docs`);
      if (response.ok) {
        console.log('Backend connection successful');
        // Refresh status
        await checkStatus();
        alert('✅ Backend connection successful! Status updated.');
        return true;
      } else {
        console.error('Backend responded with error:', response.status);
        alert(`❌ Backend responded with error: ${response.status}`);
        return false;
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      alert('❌ Backend connection failed. Please check if the server is running.');
      return false;
    } finally {
      setIsTestingConnection(false);
    }
  };

  const testImageProcessing = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Validate video dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('Video dimensions not ready');
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    try {
      console.log('Testing image processing...');
      const response = await fetch(`${API_BASE}/api/test-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Image processing test successful:', result);
        alert(`✅ Image processing works!\nSize: ${result.image_info.width}x${result.image_info.height}\nMode: ${result.image_info.mode}\nBytes: ${result.image_info.size_bytes}`);
      } else {
        const errorText = await response.text();
        console.error('Image processing test failed:', errorText);
        alert(`❌ Image processing failed: ${errorText}`);
      }
    } catch (error) {
      console.error('Network error during image test:', error);
      alert('❌ Network error during image test');
    }
  };

  const checkStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`);
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Failed to check status:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Prepare conversation history (include all previous messages + current user message)
      const allMessages = [...messages, userMessage];
      const conversationHistory = allMessages.slice(-10).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      console.log('Sending message:', userMessage.content);
      console.log('Conversation history length:', conversationHistory.length);

      const response = await fetch(`${API_BASE}/api/tutor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          conversation_history: conversationHistory
        }),
      });

      const data = await response.json();
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || 'Sorry, I encountered an error.',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Network error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleClearConversation = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      clearConversation();
    }
  };

  return (
    <div className="app-container">
      {/* Status Bar - Fixed at top */}
      {status && (
        <div className="chat-header">
          <div className="status-grid">
            <div className="status-item">
              <div className={`status-indicator ${status.genai_available ? 'available' : 'unavailable'}`}></div>
              <span className="status-text">AI Model: {status.genai_available ? 'Available' : 'Unavailable'}</span>
            </div>
            <div className="status-item">
              <div className={`status-indicator ${status.stt_available ? 'available' : 'unavailable'}`}></div>
              <span className="status-text">Speech Recognition: {status.stt_available ? 'Available' : 'Unavailable'}</span>
            </div>
            <div className="status-item">
              <div className={`status-indicator ${status.tts_available ? 'available' : 'unavailable'}`}></div>
              <span className="status-text">Text-to-Speech: {status.tts_available ? 'Available' : 'Unavailable'}</span>
            </div>
            <div className="status-item">
              <div className={`status-indicator ${speechSupported ? 'available' : 'unavailable'}`}></div>
              <span className="status-text">Voice Input: {speechSupported ? 'Available' : 'Not Supported'}</span>
            </div>
            <div className="status-item">
              <div className={`status-indicator ${status?.emotion_available ? 'available' : 'unavailable'}`}></div>
              <span className="status-text">Emotion Detection: {status?.emotion_available ? 'Available' : 'Unavailable'}</span>
            </div>
            {webcamEnabled && (
              <div className="status-item">
                <div className={`status-indicator ${
                  currentEmotion === 'backend_unavailable' || currentEmotion === 'server_error' || currentEmotion === 'emotion_unavailable'
                    ? 'unavailable'
                    : 'available animate-pulse'
                }`}></div>
                <span className="status-text">
                  Current Emotion: {
                    currentEmotion === 'backend_unavailable'
                      ? 'Backend Offline'
                      : currentEmotion === 'server_error'
                      ? 'Server Error'
                      : currentEmotion === 'emotion_unavailable'
                      ? 'Emotion Detection Unavailable'
                      : currentEmotion === 'detecting'
                      ? 'Detecting...'
                      : currentEmotion === 'no_frame'
                      ? 'No Face Detected'
                      : currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1)
                  }
                </span>
              </div>
            )}
            {cameraStatus && (
              <div className="status-item">
                <div className="status-indicator available animate-pulse"></div>
                <span className="status-text">{cameraStatus}</span>
              </div>
            )}
          </div>
          {/* Connection Test Button */}
          {(currentEmotion === 'backend_unavailable' || currentEmotion === 'server_error') && (
            <div className="connection-test">
              <button
                onClick={testBackendConnection}
                disabled={isTestingConnection}
                className="btn btn-secondary"
                title="Test backend connection"
              >
                {isTestingConnection ? (
                  <>
                    <div className="spinner"></div>
                    Testing...
                  </>
                ) : (
                  '🔄 Test Connection'
                )}
              </button>
            </div>
          )}
          {/* Image Processing Test Button */}
          {webcamEnabled && (
            <div className="connection-test">
              <button
                onClick={testImageProcessing}
                className="btn btn-secondary"
                title="Test image processing"
              >
                🖼️ Test Image
              </button>
            </div>
          )}
        </div>
      )}

      {/* Main Content Area - Two Column Layout */}
      <div className="main-content">
        {/* Left Column - Chat */}
        <div className="chat-section">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="text-center py-8">
                <div className="mb-6">
                    <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                      <TutorIcon className="w-12 h-12" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">Welcome to AI Tutor!</h3>
                  <p className="text-gray-400 mb-6 max-w-md mx-auto">
                    I'm here to help you learn and understand any topic. Ask me questions, request explanations, or explore new concepts together.
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h4 className="font-medium text-white mb-2 flex items-center gap-2"><LightbulbIcon className="w-4 h-4" />Get Help With</h4>
                    <ul className="text-sm text-gray-300 space-y-1">
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />Math problems</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />Science concepts</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />Programming</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />History & literature</li>
                    </ul>
                  </div>
                  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h4 className="font-medium text-white mb-2 flex items-center gap-2"><RocketIcon className="w-4 h-4" />Try Asking</h4>
                    <ul className="text-sm text-gray-300 space-y-1">
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />"Explain quantum physics"</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />"Help me with algebra"</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />"What is machine learning?"</li>
                      <li className="flex items-center gap-2"><BulletIcon className="w-2 h-2 flex-shrink-0" />"Tell me about World War II"</li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.role === 'user' ? 'user' : ''}`}
                >
                  <div className="message-avatar">
                    {message.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className="message-content">
                    <ReactMarkdown>
                      {message.content}
                    </ReactMarkdown>
                    <div className="message-timestamp">
                      {new Date(message.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="message">
                <div className="message-avatar">AI</div>
                <div className="message-content">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-gray-400 ml-2">AI is thinking...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input - Fixed at bottom */}
          <div className="chat-input">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <textarea
                  value={input}
                  onChange={handleInputChange}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything... (Press Enter to send, Shift+Enter for new line)"
                  className="input w-full resize-none"
                  rows={1}
                  style={{ minHeight: '44px', maxHeight: '120px' }}
                  disabled={isLoading}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = 'auto';
                    target.style.height = Math.min(target.scrollHeight, 120) + 'px';
                  }}
                />
              </div>
              <button
                onClick={toggleSpeechRecognition}
                disabled={isLoading || !speechSupported}
                className={`btn voice-btn ${isRecording ? 'btn-danger recording' : 'btn-secondary'} px-4`}
                title={!speechSupported ? 'Speech recognition not supported in this browser' : isRecording ? 'Stop recording' : 'Start voice input'}
              >
                {isRecording ? (
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                    <MicIcon className="w-4 h-4" />
                  </div>
                ) : (
                  <MicIcon className="w-4 h-4" />
                )}
              </button>
              <button
                onClick={() => {
                  console.log('📷 Camera button clicked!');
                  toggleWebcam();
                }}
                disabled={isLoading}
                className={`btn ${webcamEnabled ? 'btn-danger' : 'btn-secondary'} px-4`}
                title={webcamEnabled ? 'Stop webcam' : 'Start webcam for emotion detection'}
              >
                  {webcamEnabled ? (
                    <div className="flex items-center gap-2">
                      <CameraIcon className="w-4 h-4" />
                      <span>Stop</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <CameraIcon className="w-4 h-4" />
                      <span>Start</span>
                    </div>
                  )}
              </button>
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="btn btn-primary px-6"
                title="Send message (Enter)"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="spinner"></div>
                    <span>Sending...</span>
                  </div>
                ) : (
                  'Send'
                )}
              </button>
              {messages.length > 0 && (
                <button
                  onClick={handleClearConversation}
                  disabled={isLoading}
                  className="btn btn-secondary"
                  title="Clear conversation"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Camera and Emotion */}
        <div className="camera-section">
          {webcamEnabled ? (
            <div className="camera-container">
              <div className="camera-view">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="camera-video"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    borderRadius: '0.5rem'
                  }}
                />
                <div className="camera-overlay">
                  <div className="camera-status">
                    <div className="status-dot active animate-pulse"></div>
                    <span>Camera Active</span>
                  </div>
                  {cameraStatus && (
                    <div className="camera-instructions">
                      <small className="text-xs text-gray-600 bg-white bg-opacity-80 px-2 py-1 rounded flex items-center gap-2">
                        <LightbulbIcon className="w-4 h-4 text-gray-700" />
                        <span>{cameraStatus}</span>
                      </small>
                    </div>
                  )}
                </div>
              </div>

              {/* Emotion Monitor - Below Camera */}
              <div className="emotion-monitor">
                <div className="emotion-display">
                  <div className="emotion-icon" aria-hidden>
                    {currentEmotion === 'happy' && <HappyIcon className="w-12 h-12 text-emerald-600" />}
                    {currentEmotion === 'sad' && <SadIcon className="w-12 h-12 text-sky-600" />}
                    {currentEmotion === 'angry' && <AngryIcon className="w-12 h-12 text-rose-600" />}
                    {currentEmotion === 'fear' && <FearIcon className="w-12 h-12 text-indigo-600" />}
                    {currentEmotion === 'surprise' && <SurpriseIcon className="w-12 h-12 text-yellow-600" />}
                    {currentEmotion === 'neutral' && <NeutralIcon className="w-12 h-12 text-gray-700" />}
                    {currentEmotion === 'disgust' && <DisgustIcon className="w-12 h-12 text-lime-700" />}
                    {currentEmotion === 'no_face' && <NoFaceIcon className="w-12 h-12 text-gray-500" />}
                    {(currentEmotion === 'detecting' || currentEmotion === 'backend_unavailable' || currentEmotion === 'server_error' || currentEmotion === 'emotion_unavailable') && <DetectingIcon className="w-12 h-12 text-gray-400 animate-pulse" />}
                    {(!currentEmotion || currentEmotion === '') && <NoFaceIcon className="w-12 h-12 text-gray-500" />}
                  </div>
                  <div className="emotion-details">
                    <div className="emotion-label">
                        {smoothedEmotion
                          ? smoothedEmotion.charAt(0).toUpperCase() + smoothedEmotion.slice(1)
                          : currentEmotion === 'backend_unavailable'
                          ? 'Backend Offline'
                          : currentEmotion === 'server_error'
                          ? 'Server Error'
                          : currentEmotion === 'emotion_unavailable'
                          ? 'Detection Unavailable'
                          : currentEmotion === 'detecting'
                          ? 'Analyzing...'
                          : currentEmotion === 'no_face'
                          ? 'No Face Detected'
                          : currentEmotion
                          ? currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1)
                          : 'Ready'}
                      </div>
                  </div>
                </div>
                <div className="emotion-status">
                  <div className={`status-dot ${currentEmotion === 'detecting' ? 'analyzing' : 'active'}`}></div>
                  <span>Live Emotion Monitoring Active</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="camera-placeholder flex items-center justify-center h-full">
              <div className="placeholder-content flex flex-col items-center justify-center gap-4 text-center p-6">
                <div className="camera-icon mb-2"><CameraIcon className="w-14 h-14 text-gray-400" /></div>
                <h3 className="placeholder-title text-lg font-semibold text-white">Camera Off</h3>
                <p className="placeholder-text text-sm text-gray-400">Click "Start" to enable camera and emotion detection</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for emotion detection */}
      <canvas
        ref={canvasRef}
        className="hidden"
      />
    </div>
  );
}
