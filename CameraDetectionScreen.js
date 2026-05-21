import { MaterialIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';

const BACKEND_BASE_URL = 'https://api.tinigkamay.live';
const AGGRESSIVE_DEMO_PRESET = Platform.OS === 'android';
const SHOW_ANDROID_LANDMARK_OVERLAY = false;
const PLATFORM_PROFILE = Platform.select({
  android: {
    detectionIntervalMs: AGGRESSIVE_DEMO_PRESET ? 520 : 900,
    captureQuality: AGGRESSIVE_DEMO_PRESET ? 0.42 : 0.62,
    confThreshold: AGGRESSIVE_DEMO_PRESET ? 0.36 : 0.4,
    cameraWarmupMs: 350,
    maxCaptureAttempts: AGGRESSIVE_DEMO_PRESET ? 1 : 2,
  },
  web: {
    detectionIntervalMs: 650,
    captureQuality: 0.5,
    confThreshold: 0.35,
    cameraWarmupMs: 350,
    maxCaptureAttempts: 2,
  },
  default: {
    detectionIntervalMs: 800,
    captureQuality: 0.58,
    confThreshold: 0.35,
    cameraWarmupMs: 280,
    maxCaptureAttempts: 1,
  },
});
const N_FRAME_WINDOW = Platform.OS === 'web' ? 2 : AGGRESSIVE_DEMO_PRESET ? 2 : 4;
const MIN_VOTES_TO_ACCEPT = Platform.OS === 'web' ? 2 : AGGRESSIVE_DEMO_PRESET ? 2 : 3;
const REPEAT_COOLDOWN_MS = Platform.OS === 'web' ? 300 : AGGRESSIVE_DEMO_PRESET ? 280 : 800;
const REQUEST_TIMEOUT_MS = AGGRESSIVE_DEMO_PRESET ? 7000 : 9000;
const MIN_DETECTION_INTERVAL_MS = AGGRESSIVE_DEMO_PRESET ? 140 : 700;
const MAX_DETECTION_INTERVAL_MS = AGGRESSIVE_DEMO_PRESET ? 1000 : 1800;
const RELEASE_STREAK_TO_REPEAT = Platform.OS === 'web' ? 1 : AGGRESSIVE_DEMO_PRESET ? 1 : 2;
const LANDMARK_FAST_ACCEPT_CONFIDENCE = AGGRESSIVE_DEMO_PRESET ? 0.74 : 0.82;
const FRONT_CAMERA_CONF_OFFSET = 0.08;
const HIGH_NO_DETECT_STREAK = 7;
const TTS_MIN_INTERVAL_MS = 1700;
const WEB_AUDIO_UNLOCK_TIMEOUT_MS = 1600;
const SILENT_WAV_DATA_URI =
  'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA=';
const SHOW_LANDMARK_OVERLAY = Platform.OS !== 'android' || SHOW_ANDROID_LANDMARK_OVERLAY;
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];
const DATASET_MODE_OPTIONS = [
  { key: 'alphabet', label: 'Alphabets' },
  { key: 'numbers', label: 'Numbers' },
  { key: 'words_phrases', label: 'Words/Phrases' },
];

const getDatasetModeLabel = (mode) =>
  DATASET_MODE_OPTIONS.find((option) => option.key === mode)?.label || 'Alphabets';

export default function CameraDetectionScreen({ onBack }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isDetecting, setIsDetecting] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(Platform.OS === 'web');
  const [cameraFacing, setCameraFacing] = useState('front');
  const [isTtsEnabled, setIsTtsEnabled] = useState(false);
  const [isWebAudioUnlocked, setIsWebAudioUnlocked] = useState(Platform.OS !== 'web');
  const [ttsMode, setTtsMode] = useState('');
  const [translatedLines, setTranslatedLines] = useState([]);
  const [status, setStatus] = useState('Idle');
  const [datasetMode, setDatasetMode] = useState('alphabet');
  const [overlayData, setOverlayData] = useState({
    bbox: null,
    landmarks: [],
    label: '',
    confidence: 0,
    source: '',
    modelUsed: false,
    modelName: '',
    yoloConfidence: 0,
  });
  const [cameraLayout, setCameraLayout] = useState({ width: 0, height: 0 });

  const cameraRef = useRef(null);
  const timerRef = useRef(null);
  const isDetectingRef = useRef(false);
  const isRequestInFlightRef = useRef(false);
  const lastSpokenLabelRef = useRef('');
  const ttsSoundRef = useRef(null);
  const ttsAudioFileRef = useRef('');
  const ttsBlobUrlRef = useRef('');
  const webHtmlAudioRef = useRef(null);
  const isUnlockingWebAudioRef = useRef(false);
  const ttsInFlightRef = useRef(false);
  const lastTtsAttemptAtRef = useRef(0);
  const predictionBufferRef = useRef([]);
  const lastAcceptedPredictionRef = useRef({ label: '', timestamp: 0 });
  const cameraReadyAtRef = useRef(0);
  const datasetModeRef = useRef('alphabet');
  const noDetectStreakRef = useRef(0);
  const repeatReleaseRef = useRef(true);
  const lastCommittedLabelRef = useRef('');
  const dynamicIntervalMsRef = useRef(PLATFORM_PROFILE.detectionIntervalMs);
  const latestNetworkMsRef = useRef(0);
  const backendInfoRef = useRef({
    mediapipeMode: 'unknown',
    mirrorFrontInput: false,
    mediapipeError: '',
  });

  useEffect(
    () => () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      Speech.stop();
      const currentSound = ttsSoundRef.current;
      ttsSoundRef.current = null;
      if (currentSound) {
        currentSound.stopAsync().catch(() => {});
        currentSound.unloadAsync().catch(() => {});
      }
      const staleAudioPath = ttsAudioFileRef.current;
      ttsAudioFileRef.current = '';
      if (staleAudioPath) {
        FileSystem.deleteAsync(staleAudioPath, { idempotent: true }).catch(() => {});
      }
      const staleBlobUrl = ttsBlobUrlRef.current;
      ttsBlobUrlRef.current = '';
      if (staleBlobUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(staleBlobUrl);
      }
      const htmlAudio = webHtmlAudioRef.current;
      webHtmlAudioRef.current = null;
      if (htmlAudio) {
        htmlAudio.pause();
        htmlAudio.removeAttribute('src');
        htmlAudio.load();
      }
    },
    []
  );

  useEffect(() => {
    if (Platform.OS === 'web') {
      return;
    }
    Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: true,
      staysActiveInBackground: false,
      shouldDuckAndroid: true,
      playThroughEarpieceAndroid: false,
    }).catch(() => {});
  }, []);

  useEffect(() => {
    datasetModeRef.current = datasetMode;
  }, [datasetMode]);

  useEffect(() => {
    if (Platform.OS === 'web' && permission?.granted && !isCameraReady) {
      setStatus('Camera permission granted (web), waiting for stream...');
    }
  }, [isCameraReady, permission?.granted]);

  useEffect(() => {
    let isMounted = true;

    const loadBackendHealth = async () => {
      try {
        const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/health`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        if (!isMounted) {
          return;
        }
        backendInfoRef.current = {
          mediapipeMode: data?.mediapipe_mode || 'unknown',
          mirrorFrontInput: Boolean(data?.mirror_front_camera_input),
          mediapipeError: data?.mediapipe?.last_error || '',
        };
        if (backendInfoRef.current.mediapipeMode === 'disabled') {
          const backendReason = backendInfoRef.current.mediapipeError
            ? ` Reason: ${backendInfoRef.current.mediapipeError}`
            : '';
          setStatus(`Backend running YOLO-only mode. Landmark assist is disabled.${backendReason}`);
        }
      } catch {
        // Ignore initial health failures; detection flow reports runtime errors.
      }
    };

    loadBackendHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  const rawTranscript = useMemo(() => {
    if (translatedLines.length === 0) {
      return 'Waiting for translated text...';
    }
    return translatedLines.join(' ');
  }, [translatedLines]);
  const transcript = rawTranscript;
  const isFrontCamera = cameraFacing === 'front';
  const isWeb = Platform.OS === 'web';
  const shouldMirrorOverlayX = false;

  const clearDetectionTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const scheduleNextDetection = () => {
    clearDetectionTimer();
    if (!isDetectingRef.current) {
      return;
    }
    timerRef.current = setTimeout(() => {
      runDetection();
    }, dynamicIntervalMsRef.current);
  };

  const wait = (ms) =>
    new Promise((resolve) => {
      setTimeout(resolve, ms);
    });

  const isRetriableWebCaptureError = (message) => {
    const lowered = (message || '').toLowerCase();
    return (
      lowered.includes('not have enough camera data') ||
      lowered.includes('video element') ||
      lowered.includes('ready state')
    );
  };

  const fetchWithTimeout = async (url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const getAdaptiveConfThreshold = () => {
    let threshold = PLATFORM_PROFILE.confThreshold;
    if (isFrontCamera) {
      threshold -= FRONT_CAMERA_CONF_OFFSET;
    }
    if (noDetectStreakRef.current >= HIGH_NO_DETECT_STREAK) {
      threshold -= 0.06;
    }
    return Number(clamp(threshold, 0.22, 0.65).toFixed(2));
  };

  const updateCadenceFromNetwork = (elapsedMs) => {
    latestNetworkMsRef.current = elapsedMs;
    // Request runtime already dominates when network is slow.
    // Keep post-request wait short to maintain a live-feeling loop.
    const target = clamp(
      Math.round(elapsedMs * (AGGRESSIVE_DEMO_PRESET ? 0.12 : 0.35) + 70),
      MIN_DETECTION_INTERVAL_MS,
      MAX_DETECTION_INTERVAL_MS
    );
    dynamicIntervalMsRef.current = Math.round((dynamicIntervalMsRef.current * 0.7) + (target * 0.3));
  };

  const appendPrediction = (label) => {
    const isSameAsLastCommitted = lastCommittedLabelRef.current === label;
    if (isSameAsLastCommitted && !repeatReleaseRef.current) {
      return false;
    }

    lastCommittedLabelRef.current = label;
    repeatReleaseRef.current = false;
    setTranslatedLines((prev) => {
      return [...prev, label];
    });
    return true;
  };

  const resetSmoothingState = () => {
    predictionBufferRef.current = [];
    lastAcceptedPredictionRef.current = { label: '', timestamp: 0 };
    repeatReleaseRef.current = true;
    lastCommittedLabelRef.current = '';
  };

  const clearOverlay = () => {
    setOverlayData({
      bbox: null,
      landmarks: [],
      label: '',
      confidence: 0,
      source: '',
      modelUsed: false,
      modelName: '',
      yoloConfidence: 0,
    });
  };

  const getSmoothedPrediction = (label, confidence) => {
    const now = Date.now();
    const next = predictionBufferRef.current.concat({
      label,
      confidence: Number(confidence) || 0,
      timestamp: now,
    });
    predictionBufferRef.current = next.slice(-N_FRAME_WINDOW);

    const counts = {};
    const confidenceSums = {};
    for (const item of predictionBufferRef.current) {
      counts[item.label] = (counts[item.label] || 0) + 1;
      confidenceSums[item.label] = (confidenceSums[item.label] || 0) + item.confidence;
    }

    let topLabel = '';
    let topVotes = 0;
    for (const key of Object.keys(counts)) {
      const votes = counts[key];
      if (votes > topVotes) {
        topVotes = votes;
        topLabel = key;
      }
    }

    if (!topLabel || topVotes < MIN_VOTES_TO_ACCEPT) {
      return null;
    }

    const previous = lastAcceptedPredictionRef.current;
    if (
      previous.label === topLabel &&
      now - previous.timestamp < REPEAT_COOLDOWN_MS
    ) {
      return null;
    }

    const avgConfidence = confidenceSums[topLabel] / topVotes;
    lastAcceptedPredictionRef.current = { label: topLabel, timestamp: now };
    predictionBufferRef.current = [];

    return {
      label: topLabel,
      votes: topVotes,
      avgConfidence,
    };
  };

  const stopBackendAudioPlayback = async () => {
    const currentSound = ttsSoundRef.current;
    ttsSoundRef.current = null;
    if (!currentSound) {
      return;
    }
    try {
      await currentSound.stopAsync();
    } catch {
      // Ignore stop errors when the sound already ended.
    }
    try {
      await currentSound.unloadAsync();
    } catch {
      // Ignore unload errors on disposed sounds.
    }
    const staleAudioPath = ttsAudioFileRef.current;
    ttsAudioFileRef.current = '';
    if (staleAudioPath) {
      await FileSystem.deleteAsync(staleAudioPath, { idempotent: true }).catch(() => {});
    }
    const staleBlobUrl = ttsBlobUrlRef.current;
    ttsBlobUrlRef.current = '';
    if (staleBlobUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(staleBlobUrl);
    }
    const htmlAudio = webHtmlAudioRef.current;
    webHtmlAudioRef.current = null;
    if (htmlAudio) {
      htmlAudio.pause();
      htmlAudio.removeAttribute('src');
      htmlAudio.load();
    }
  };

  const unlockWebAudio = async () => {
    if (Platform.OS !== 'web') {
      return true;
    }
    if (isWebAudioUnlocked) {
      return true;
    }
    if (isUnlockingWebAudioRef.current) {
      return false;
    }
    const HtmlAudioCtor = globalThis?.Audio;
    if (typeof HtmlAudioCtor !== 'function') {
      return false;
    }
    isUnlockingWebAudioRef.current = true;
    try {
      const probe = new HtmlAudioCtor(SILENT_WAV_DATA_URI);
      probe.muted = false;
      probe.volume = 0.01;
      probe.playsInline = true;
      const playPromise = probe.play();
      const settledPlay = playPromise && typeof playPromise.then === 'function'
        ? playPromise
        : Promise.resolve();
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Web audio unlock timed out')), WEB_AUDIO_UNLOCK_TIMEOUT_MS);
      });
      await Promise.race([settledPlay, timeoutPromise]);
      probe.pause();
      probe.currentTime = 0;
      probe.removeAttribute('src');
      probe.load();
      setIsWebAudioUnlocked(true);
      return true;
    } catch {
      return false;
    } finally {
      isUnlockingWebAudioRef.current = false;
    }
  };

  const buildWebBlobUrlFromBase64 = (audioBase64, mimeType) => {
    if (typeof atob !== 'function') {
      throw new Error('Browser base64 decoder is unavailable');
    }
    if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
      throw new Error('Blob URL APIs are unavailable');
    }
    const binary = atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: mimeType || 'audio/mpeg' });
    return URL.createObjectURL(blob);
  };

  const playWithHtmlAudioElement = async (blobUrl) => {
    const HtmlAudioCtor = globalThis?.Audio;
    if (typeof HtmlAudioCtor !== 'function') {
      throw new Error('HTMLAudioElement is unavailable');
    }
    const audio = new HtmlAudioCtor(blobUrl);
    webHtmlAudioRef.current = audio;
    audio.preload = 'auto';

    await new Promise((resolve, reject) => {
      const cleanup = () => {
        audio.onended = null;
        audio.onerror = null;
      };
      audio.onended = () => {
        cleanup();
        resolve();
      };
      audio.onerror = () => {
        cleanup();
        reject(new Error('HTML audio playback failed'));
      };
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch((err) => {
          cleanup();
          reject(err);
        });
      }
    });
    if (webHtmlAudioRef.current === audio) {
      webHtmlAudioRef.current = null;
    }
  };

  const speakWithBackendTts = async (label) => {
    const response = await fetchWithTimeout(
      `${BACKEND_BASE_URL}/tts`,
      {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: label,
          language_code: 'fil-PH',
        }),
      },
      7000
    );
    if (!response.ok) {
      throw new Error(`Backend TTS error: ${response.status}`);
    }
    const payload = await response.json();
    if (!payload?.audio_base64) {
      throw new Error('Backend TTS returned empty audio');
    }

    await Speech.stop();
    await stopBackendAudioPlayback();
    let playbackSource = null;
    let webBlobUrl = '';
    if (Platform.OS === 'web') {
      webBlobUrl = buildWebBlobUrlFromBase64(payload.audio_base64, payload?.mime_type);
      ttsBlobUrlRef.current = webBlobUrl;
      if (!isWebAudioUnlocked) {
        throw new Error('WebAudioLocked');
      }
      playbackSource = { uri: webBlobUrl };
    } else {
      const baseCachePath = FileSystem.cacheDirectory || FileSystem.documentDirectory;
      if (!baseCachePath) {
        throw new Error('No writable cache directory available for audio playback');
      }
      const audioPath = `${baseCachePath}tts-${Date.now()}-${Math.random().toString(36).slice(2)}.mp3`;
      await FileSystem.writeAsStringAsync(audioPath, payload.audio_base64, {
        encoding: FileSystem.EncodingType.Base64,
      });
      ttsAudioFileRef.current = audioPath;
      playbackSource = { uri: audioPath };
    }

    let soundResult = null;
    try {
      soundResult = await Audio.Sound.createAsync(playbackSource, {
        shouldPlay: true,
        progressUpdateIntervalMillis: 250,
      });
    } catch (audioError) {
      if (Platform.OS !== 'web' || !webBlobUrl) {
        throw audioError;
      }
      await playWithHtmlAudioElement(webBlobUrl);
      const staleBlobUrl = ttsBlobUrlRef.current;
      ttsBlobUrlRef.current = '';
      if (staleBlobUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(staleBlobUrl);
      }
      return;
    }

    const { sound } = soundResult;
    ttsSoundRef.current = sound;
    sound.setOnPlaybackStatusUpdate((status) => {
      if (!status.isLoaded || !status.didJustFinish) {
        return;
      }
      sound.unloadAsync().catch(() => {});
      if (ttsSoundRef.current === sound) {
        ttsSoundRef.current = null;
      }
      const staleAudioPath = ttsAudioFileRef.current;
      ttsAudioFileRef.current = '';
      if (staleAudioPath) {
        FileSystem.deleteAsync(staleAudioPath, { idempotent: true }).catch(() => {});
      }
      const staleBlobUrl = ttsBlobUrlRef.current;
      ttsBlobUrlRef.current = '';
      if (staleBlobUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(staleBlobUrl);
      }
    });
  };

  const speakWithExpoSpeech = async (label) => {
    await stopBackendAudioPlayback();
    await Speech.stop();
    return new Promise((resolve, reject) => {
      Speech.speak(label, {
        language: 'fil-PH',
        pitch: 1.0,
        rate: 1.0,
        onDone: resolve,
        onStopped: resolve,
        onError: reject,
      });
    });
  };

  const speakLabel = async (label) => {
    if (!isTtsEnabled || !label || lastSpokenLabelRef.current === label) {
      return;
    }
    if (ttsInFlightRef.current) {
      return;
    }
    const now = Date.now();
    if (now - lastTtsAttemptAtRef.current < TTS_MIN_INTERVAL_MS) {
      return;
    }

    ttsInFlightRef.current = true;
    lastTtsAttemptAtRef.current = now;
    lastSpokenLabelRef.current = label;

    try {
      await speakWithBackendTts(label);
      setTtsMode('Voice: gTTS');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (Platform.OS === 'web' && errorMessage.includes('WebAudioLocked')) {
        setTtsMode('Voice: tap speaker to enable web audio');
        setStatus('Web audio is locked. Tap the speaker button to enable TTS.');
        return;
      }
      console.warn('Backend TTS failed, falling back to device speech', error);
      setTtsMode('Voice: backend unavailable');
      setStatus('Backend TTS unavailable, using device speech...');
      try {
        await speakWithExpoSpeech(label);
        setTtsMode('Voice: device fallback');
      } catch (fallbackError) {
        setTtsMode('Voice: unavailable');
        setStatus('TTS unavailable on this device');
        console.error(fallbackError);
      }
    } finally {
      ttsInFlightRef.current = false;
    }
  };

  const stopDetectionLoop = () => {
    isDetectingRef.current = false;
    clearDetectionTimer();
    Speech.stop();
    stopBackendAudioPlayback().catch(() => {});
    resetSmoothingState();
    noDetectStreakRef.current = 0;
    dynamicIntervalMsRef.current = PLATFORM_PROFILE.detectionIntervalMs;
    clearOverlay();
    setIsDetecting(false);
    setStatus('Stopped');
  };

  const runDetection = async () => {
    const mustWaitForReady = Platform.OS !== 'web' && !isCameraReady;
    if (!cameraRef.current || isRequestInFlightRef.current || mustWaitForReady) {
      if (mustWaitForReady) {
        setStatus('Waiting for camera...');
      }
      if (isDetectingRef.current) {
        scheduleNextDetection();
      }
      return;
    }

    const cameraReadyForMs = Date.now() - cameraReadyAtRef.current;
    if (
      isCameraReady &&
      cameraReadyAtRef.current > 0 &&
      cameraReadyForMs < PLATFORM_PROFILE.cameraWarmupMs
    ) {
      setStatus(
        `Warming up camera... (${Math.max(
          0,
          PLATFORM_PROFILE.cameraWarmupMs - cameraReadyForMs
        )}ms)`
      );
      if (isDetectingRef.current) {
        scheduleNextDetection();
      }
      return;
    }

    try {
      isRequestInFlightRef.current = true;
      const requestDatasetMode = datasetModeRef.current;
      const requestConfThreshold = getAdaptiveConfThreshold();
      let photo = null;
      let lastCaptureError = null;
      for (let attempt = 1; attempt <= PLATFORM_PROFILE.maxCaptureAttempts; attempt += 1) {
        try {
          photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: PLATFORM_PROFILE.captureQuality,
            ...(Platform.OS === 'android' ? { skipProcessing: true } : {}),
          });
          if (photo?.base64) {
            break;
          }
          lastCaptureError = new Error('Camera returned empty frame');
        } catch (captureError) {
          lastCaptureError = captureError;
          const msg = captureError instanceof Error ? captureError.message : String(captureError);
          const canRetry =
            isWeb &&
            attempt < PLATFORM_PROFILE.maxCaptureAttempts &&
            isRetriableWebCaptureError(msg);
          if (!canRetry) {
            break;
          }
          await wait(150);
        }
      }

      if (!photo?.base64) {
        const msg =
          lastCaptureError instanceof Error ? lastCaptureError.message : String(lastCaptureError);
        setStatus(
          `Camera capture failed (${isWeb ? 'web' : 'mobile'}): ${msg || 'Empty frame'}`
        );
        return;
      }

      const requestStartedAt = Date.now();
      // Keep MediaPipe enabled for assist, but avoid hard-blocking detection
      // on frames where landmarks are temporarily unavailable.
      const requireLandmarks = false;
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/detect`, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: photo.base64,
          conf_threshold: requestConfThreshold,
          dataset_mode: requestDatasetMode,
          is_front_camera: isFrontCamera,
          require_landmarks: requireLandmarks,
        }),
      });
      const requestElapsedMs = Date.now() - requestStartedAt;
      updateCadenceFromNetwork(requestElapsedMs);

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const result = await response.json();
      if (requestDatasetMode !== datasetModeRef.current) {
        // User switched modes while this request was in flight.
        // Ignore stale response so UI instantly reflects the new mode.
        return;
      }
      setOverlayData({
        bbox: result?.bbox || null,
        landmarks: Array.isArray(result?.landmarks) ? result.landmarks : [],
        label: result?.label || '',
        confidence: Number(result?.confidence) || 0,
        source: result?.prediction_source || '',
        modelUsed: Boolean(result?.model_used),
        modelName: result?.model_name || '',
        yoloConfidence: Number(result?.yolo_confidence) || 0,
      });
      if (result?.detected && result?.label) {
        noDetectStreakRef.current = 0;
        const isLandmarkFastPath =
          result?.prediction_source === 'landmark_classifier' &&
          Number(result.confidence) >= LANDMARK_FAST_ACCEPT_CONFIDENCE;
        if (isLandmarkFastPath) {
          appendPrediction(result.label);
          setStatus(`${result.label} (${(Number(result.confidence) * 100).toFixed(1)}%)`);
          speakLabel(result.label).catch((speechError) => {
            console.error('TTS error', speechError);
          });
          return;
        }
        const smoothed = getSmoothedPrediction(result.label, result.confidence);
        if (smoothed) {
          appendPrediction(smoothed.label);
          setStatus(`${smoothed.label} (${(smoothed.avgConfidence * 100).toFixed(1)}%)`);
          speakLabel(smoothed.label).catch((speechError) => {
            console.error('TTS error', speechError);
          });
        } else {
          setStatus(`${result.label} (${(result.confidence * 100).toFixed(1)}%)`);
        }
      } else {
        noDetectStreakRef.current += 1;
        if (noDetectStreakRef.current >= RELEASE_STREAK_TO_REPEAT) {
          repeatReleaseRef.current = true;
        }
        clearOverlay();
        setStatus('No sign detected');
      }
    } catch (error) {
      const msg =
        error?.name === 'AbortError'
          ? 'Request timed out. Network jitter is high.'
          : error instanceof Error
            ? error.message
            : String(error);
      dynamicIntervalMsRef.current = clamp(
        dynamicIntervalMsRef.current + 180,
        MIN_DETECTION_INTERVAL_MS,
        MAX_DETECTION_INTERVAL_MS
      );
      setStatus(`Detection error: ${msg}`);
      console.error('Detection pipeline error', error);
    } finally {
      isRequestInFlightRef.current = false;
      if (isDetectingRef.current) {
        scheduleNextDetection();
      }
    }
  };

  const handleStartStop = async () => {
    if (isDetecting) {
      stopDetectionLoop();
      return;
    }

    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        setStatus('Camera permission denied');
        return;
      }
    }

    const startupHints = [];
    if (isFrontCamera && !backendInfoRef.current.mirrorFrontInput) {
      startupHints.push('front-camera mirror correction is disabled on backend');
    }
    if (backendInfoRef.current.mediapipeMode === 'disabled') {
      const reason = backendInfoRef.current.mediapipeError
        ? `MediaPipe disabled: ${backendInfoRef.current.mediapipeError}`
        : 'MediaPipe is disabled';
      startupHints.push(reason);
    }
    setStatus(
      startupHints.length > 0
        ? `Detecting... (${startupHints.join('; ')})`
        : 'Detecting...'
    );
    setIsDetecting(true);
    isDetectingRef.current = true;
    noDetectStreakRef.current = 0;
    dynamicIntervalMsRef.current = PLATFORM_PROFILE.detectionIntervalMs;
    clearDetectionTimer();
    await runDetection();
  };

  const clearTranscript = () => {
    setTranslatedLines([]);
    lastSpokenLabelRef.current = '';
    Speech.stop();
    stopBackendAudioPlayback().catch(() => {});
    resetSmoothingState();
    clearOverlay();
  };

  const toggleTts = () => {
    if (isTtsEnabled) {
      Speech.stop();
      stopBackendAudioPlayback().catch(() => {});
      lastSpokenLabelRef.current = '';
      setTtsMode('');
      setIsTtsEnabled(false);
      return;
    }
    if (Platform.OS === 'web') {
      setStatus('Enabling web audio...');
      unlockWebAudio().then((unlocked) => {
        if (!unlocked) {
          setIsTtsEnabled(false);
          setTtsMode('Voice: tap speaker to enable web audio');
          setStatus('Browser blocked audio. Tap the speaker button again to enable TTS.');
        } else {
          setIsTtsEnabled(true);
          setTtsMode('Voice: gTTS');
          setStatus('Web audio enabled.');
        }
      });
      return;
    }
    setIsTtsEnabled(true);
  };

  const toggleCameraFacing = () => {
    setCameraFacing((prev) => {
      const next = prev === 'front' ? 'back' : 'front';
      setStatus(`Camera: ${next === 'front' ? 'Front' : 'Back'}. Detection threshold auto-tuned.`);
      return next;
    });
    setIsCameraReady(false);
    cameraReadyAtRef.current = 0;
    if (isDetectingRef.current) {
      clearDetectionTimer();
      scheduleNextDetection();
    }
  };


  const selectDatasetMode = (nextMode) => {
    datasetModeRef.current = nextMode;
    setDatasetMode(nextMode);
    setStatus(`Active mode: ${getDatasetModeLabel(nextMode)}`);
    resetSmoothingState();
    clearOverlay();
    if (isDetectingRef.current) {
      clearDetectionTimer();
      if (!isRequestInFlightRef.current) {
        runDetection();
      }
    }
  };

  const handleBack = () => {
    stopDetectionLoop();
    onBack();
  };

  const renderLandmarkConnections = () => {
    if (cameraLayout.width <= 0 || cameraLayout.height <= 0 || overlayData.landmarks.length < 2) {
      return null;
    }
    return HAND_CONNECTIONS.map(([startIndex, endIndex], idx) => {
      const start = overlayData.landmarks[startIndex];
      const end = overlayData.landmarks[endIndex];
      if (!start || !end) {
        return null;
      }
      const startXNorm = shouldMirrorOverlayX && isFrontCamera ? 1 - start.x : start.x;
      const endXNorm = shouldMirrorOverlayX && isFrontCamera ? 1 - end.x : end.x;
      const startX = startXNorm * cameraLayout.width;
      const startY = start.y * cameraLayout.height;
      const endX = endXNorm * cameraLayout.width;
      const endY = end.y * cameraLayout.height;
      const dx = endX - startX;
      const dy = endY - startY;
      const length = Math.sqrt(dx * dx + dy * dy);
      const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;

      return (
        <View
          key={`connection-${idx}`}
          style={[
            styles.landmarkLine,
            {
              width: length,
              left: midX - length / 2,
              top: midY - 1,
              transform: [{ rotate: `${angle}deg` }],
            },
          ]}
        />
      );
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.topRow}>
        <Pressable onPress={handleBack} style={styles.backBtn}>
          <MaterialIcons name="arrow-back" size={20} color="#E2E8F0" />
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <Text style={styles.title}>Camera Detection</Text>
        <View style={styles.spacer} />
      </View>

      {permission?.granted ? (
        <View
          style={styles.cameraFrame}
          onLayout={(event) => {
            const { width, height } = event.nativeEvent.layout;
            setCameraLayout({ width, height });
          }}
        >
          <CameraView
            key={isWeb ? `camera-web-${cameraFacing}` : 'camera-native'}
            ref={cameraRef}
            style={styles.camera}
            facing={cameraFacing}
            {...(!isWeb ? { animateShutter: false } : {})}
            onCameraReady={() => {
              setIsCameraReady(true);
              cameraReadyAtRef.current = Date.now();
              setStatus((current) =>
                current === 'Waiting for camera...' ||
                current.startsWith('Camera') ||
                current.includes('waiting for stream')
                  ? `Camera ready (${cameraFacing === 'front' ? 'Front' : 'Back'})`
                  : current
              );
            }}
            onMountError={(error) => {
              setIsCameraReady(false);
              cameraReadyAtRef.current = 0;
              const msg = error?.message || 'Unknown mount error';
              if (isWeb && cameraFacing === 'back') {
                setCameraFacing('front');
                setStatus(`Back camera unavailable on this browser/device. Switched to front camera.`);
                return;
              }
              setStatus(`Camera failed to start (${isWeb ? 'web' : cameraFacing}): ${msg}`);
              console.error('Camera mount error', error);
            }}
          />
          <View pointerEvents="none" style={styles.overlayFrame}>
            <View pointerEvents="none" style={styles.overlayLayer}>
              {overlayData?.bbox ? (
                <View
                  style={[
                    styles.boundingBox,
                    {
                      left: `${
                        (shouldMirrorOverlayX && isFrontCamera
                          ? 1 - (overlayData.bbox.x + overlayData.bbox.width)
                          : overlayData.bbox.x) * 100
                      }%`,
                      top: `${overlayData.bbox.y * 100}%`,
                      width: `${overlayData.bbox.width * 100}%`,
                      height: `${overlayData.bbox.height * 100}%`,
                    },
                  ]}
                />
              ) : null}
              {SHOW_LANDMARK_OVERLAY ? renderLandmarkConnections() : null}
              {SHOW_LANDMARK_OVERLAY
                ? overlayData?.landmarks?.map((landmark, index) => (
                    <View
                      key={`landmark-${index}`}
                      style={[
                        styles.landmarkPoint,
                        {
                          left: `${
                            (shouldMirrorOverlayX && isFrontCamera ? 1 - landmark.x : landmark.x) *
                            100
                          }%`,
                          top: `${landmark.y * 100}%`,
                        },
                      ]}
                    />
                  ))
                : null}
              {overlayData?.label ? (
                <View style={styles.overlayBadge}>
                  <Text style={styles.overlayBadgeText}>
                    {overlayData.label} ({(overlayData.confidence * 100).toFixed(1)}%)
                  </Text>
                  {overlayData.source ? (
                    <Text style={styles.overlaySubText}>{overlayData.source}</Text>
                  ) : null}
                  {overlayData.modelUsed ? (
                    <Text style={styles.overlaySubText}>
                      Model: {overlayData.modelName || 'YOLO'}
                      {overlayData.yoloConfidence
                        ? ` (${(overlayData.yoloConfidence * 100).toFixed(1)}%)`
                        : ''}
                    </Text>
                  ) : null}
                </View>
              ) : null}
            </View>
          </View>
          <Pressable
            onPress={toggleCameraFacing}
            style={styles.flipCameraBtn}
            accessibilityRole="button"
            accessibilityLabel="Switch camera"
          >
            <MaterialIcons
              name={Platform.OS === 'ios' ? 'flip-camera-ios' : 'flip-camera-android'}
              size={24}
              color="#E2E8F0"
            />
          </Pressable>
        </View>
      ) : (
        <View style={styles.permissionCard}>
          <Text style={styles.permissionTitle}>Camera permission required</Text>
          <Pressable onPress={requestPermission} style={styles.primaryBtn}>
            <Text style={styles.primaryBtnText}>Grant Permission</Text>
          </Pressable>
        </View>
      )}

      <View style={styles.controls}>
        <View style={styles.activeModeBanner}>
          <Text style={styles.activeModeText}>Active Mode: {getDatasetModeLabel(datasetMode)}</Text>
        </View>

        <View style={styles.controlsRow}>
          {DATASET_MODE_OPTIONS.map((option) => {
            const isActive = datasetMode === option.key;
            return (
              <Pressable
                key={option.key}
                onPress={() => selectDatasetMode(option.key)}
                style={[styles.modeBtn, styles.rowThirdBtn, isActive && styles.modeBtnActive]}
              >
                <Text style={styles.modeBtnText}>{option.label}</Text>
              </Pressable>
            );
          })}
        </View>

        <View style={styles.controlsRow}>
          <Pressable
            onPress={handleStartStop}
            style={[styles.primaryBtn, styles.rowFillBtn, isDetecting && styles.stopBtn]}
          >
            <Text style={styles.primaryBtnText}>
              {isDetecting ? 'Stop Detection' : 'Start Detection'}
            </Text>
          </Pressable>
          <Pressable onPress={clearTranscript} style={[styles.clearBtn, styles.rowCompactBtn]}>
            <Text style={styles.clearText}>Clear</Text>
          </Pressable>
        </View>

        <View style={styles.controlsRow}>
          <View style={styles.rowIconRight}>
            <Pressable
              onPress={toggleTts}
              style={[styles.ttsIconBtn, isTtsEnabled && styles.ttsIconBtnActive]}
              accessibilityRole="button"
              accessibilityLabel={isTtsEnabled ? 'Disable text to speech' : 'Enable text to speech'}
            >
              <MaterialIcons
                name={isTtsEnabled ? 'volume-up' : 'volume-off'}
                size={20}
                color={isTtsEnabled ? '#FFFFFF' : '#CBD5E1'}
              />
            </Pressable>
          </View>
        </View>

      </View>

      <Text style={styles.statusText}>{status}</Text>
      {ttsMode ? <Text style={styles.ttsModeText}>{ttsMode}</Text> : null}

      <View style={styles.transcriptBox}>
        <Text style={styles.transcriptLabel}>Translated Text</Text>
        <Text style={styles.transcriptText}>{transcript}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#101622',
    padding: 16,
  },
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  backBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#1F2937',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  backText: {
    color: '#E2E8F0',
    fontWeight: '600',
  },
  title: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '700',
  },
  spacer: {
    width: 68,
  },
  cameraFrame: {
    flex: 1,
    minHeight: 280,
    borderRadius: Platform.OS === 'android' ? 0 : 14,
    overflow: 'visible',
    marginBottom: 14,
    backgroundColor: '#000000',
  },
  camera: {
    flex: 1,
  },
  flipCameraBtn: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: '#334155',
    backgroundColor: '#0F172AD9',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  overlayFrame: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 14,
    overflow: 'hidden',
  },
  overlayLayer: {
    ...StyleSheet.absoluteFillObject,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#22C55E',
    borderRadius: 8,
  },
  landmarkPoint: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#F8FAFC',
    marginLeft: -3,
    marginTop: -3,
  },
  landmarkLine: {
    position: 'absolute',
    height: 2,
    backgroundColor: '#38BDF8',
  },
  overlayBadge: {
    position: 'absolute',
    left: 10,
    top: 10,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 10,
    backgroundColor: '#0F172AE8',
    borderWidth: 1,
    borderColor: '#38BDF8',
  },
  overlayBadgeText: {
    color: '#E2E8F0',
    fontSize: 12,
    fontWeight: '700',
  },
  overlaySubText: {
    color: '#94A3B8',
    fontSize: 11,
    marginTop: 2,
  },
  permissionCard: {
    flex: 1,
    borderRadius: 14,
    backgroundColor: '#0F172ACC',
    borderWidth: 1,
    borderColor: '#1F2937',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 14,
  },
  permissionTitle: {
    color: '#CBD5E1',
    fontSize: 14,
  },
  controls: {
    gap: 8,
    marginBottom: 10,
  },
  activeModeBanner: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#256AF4',
    backgroundColor: '#1D4ED822',
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  activeModeText: {
    color: '#DBEAFE',
    fontWeight: '700',
    fontSize: 12,
    textAlign: 'center',
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  primaryBtn: {
    backgroundColor: '#256AF4',
    borderRadius: 12,
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  stopBtn: {
    backgroundColor: '#EF4444',
  },
  primaryBtnText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  clearBtn: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
  },
  clearText: {
    color: '#CBD5E1',
    fontWeight: '600',
  },
  rowIconRight: {
    flex: 1,
    alignItems: 'flex-end',
  },
  ttsIconBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111827',
  },
  modeBtn: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111827',
  },
  modeBtnActive: {
    borderColor: '#22C55E',
    backgroundColor: '#14532D',
  },
  modeBtnText: {
    color: '#E2E8F0',
    fontWeight: '700',
    fontSize: 12,
    textAlign: 'center',
  },
  rowFillBtn: {
    flex: 1,
  },
  rowCompactBtn: {
    width: 96,
    minHeight: 48,
  },
  rowThirdBtn: {
    flex: 1,
    minHeight: 48,
    minWidth: 0,
  },
  ttsIconBtnActive: {
    backgroundColor: '#256AF4',
    borderColor: '#256AF4',
  },
  statusText: {
    color: '#94A3B8',
    marginBottom: 10,
    minHeight: 18,
  },
  ttsModeText: {
    color: '#60A5FA',
    marginBottom: 10,
    minHeight: 18,
    fontWeight: '700',
  },
  transcriptBox: {
    minHeight: 100,
    borderRadius: 12,
    backgroundColor: '#0F172ACC',
    borderWidth: 1,
    borderColor: '#1F2937',
    padding: 12,
  },
  transcriptLabel: {
    color: '#94A3B8',
    fontSize: 12,
    marginBottom: 8,
    textTransform: 'uppercase',
    fontWeight: '700',
  },
  transcriptText: {
    color: '#F1F5F9',
    lineHeight: 20,
  },
});
