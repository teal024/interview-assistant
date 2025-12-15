"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import styles from "./page.module.css";

type Style = "supportive" | "neutral" | "cold";
type ChatMessage = {
  role: "interviewer" | "user";
  content: string;
  turn: number;
  style?: Style;
};
type Tip = { summary: string; detail: string };

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/interview";
const HTTP_BASE = (WS_URL.startsWith("ws") ? WS_URL.replace(/^ws/, "http") : WS_URL).replace(/\/ws\/interview$/, "");

const STYLE_LABELS: Record<Style, string> = {
  supportive: "Supportive",
  neutral: "Neutral",
  cold: "Cold / Pressuring",
};

const STYLE_HELP: Record<Style, string> = {
  supportive: "Warm tone, short pauses, light encouragement.",
  neutral: "Balanced prompts, steady pacing.",
  cold: "Direct challenges, strategic silence, blunt follow-ups.",
};
const SILENCE_TIMEOUT_MS = 2600;
const MIN_RECORDING_MS = 1200;

type SessionStatus = "idle" | "connecting" | "connected" | "active" | "error" | "closed";

type Analytics = {
  speakingRate: number; // words per minute (estimate)
  pauseRatio: number; // fraction 0-1
  gaze: number; // percent
  fillers: number; // count in last answer
  speechSeconds?: number; // rolling speech duration used for rate calculations
  volume?: number; // normalized RMS (0-1)
};

const initialAnalytics: Analytics = { speakingRate: 0, pauseRatio: 0, gaze: 0, fillers: 0, speechSeconds: 0, volume: 0 };

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function useMediaSensors(active: boolean) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [metrics, setMetrics] = useState<Analytics>(initialAnalytics);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  const totalWindowMsRef = useRef<number>(0);
  const speechMsRef = useRef<number>(0);
  const activityRef = useRef<{ speaking: boolean; duration: number }[]>([]);
  const faceLoopRef = useRef<number | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);

  const stopAll = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (faceLoopRef.current) clearTimeout(faceLoopRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioCtxRef.current?.close();
    streamRef.current = null;
    setStream(null);
    audioCtxRef.current = null;
    analyserRef.current = null;
    dataArrayRef.current = null;
    lastTsRef.current = null;
    totalWindowMsRef.current = 0;
    speechMsRef.current = 0;
    activityRef.current = [];
    setMetrics(initialAnalytics);
  }, []);

  const runAudioLoop = useCallback(() => {
    const loop = () => {
      const analyser = analyserRef.current;
      const dataArray = dataArrayRef.current;
      if (!analyser || !dataArray) return;

      analyser.getByteTimeDomainData(dataArray);
      const len = dataArray.length;
      let sumSquares = 0;
      for (let i = 0; i < len; i += 1) {
        const v = (dataArray[i] - 128) / 128;
        sumSquares += v * v;
      }
      const rms = Math.sqrt(sumSquares / len);
      const speaking = rms > 0.035; // empirically safe threshold for voice activity
      const now = performance.now();
      const delta = lastTsRef.current ? now - lastTsRef.current : 16;
      lastTsRef.current = now;

      activityRef.current.push({ speaking, duration: delta });
      totalWindowMsRef.current += delta;
      if (speaking) speechMsRef.current += delta;

      // Keep an 8s rolling window
      const windowMs = 8000;
      while (totalWindowMsRef.current > windowMs && activityRef.current.length > 0) {
        const removed = activityRef.current.shift();
        if (removed) {
          totalWindowMsRef.current -= removed.duration;
          if (removed.speaking) speechMsRef.current -= removed.duration;
        }
      }

      const windowDuration = Math.max(totalWindowMsRef.current, 1);
      const pauseRatio = clamp((windowDuration - speechMsRef.current) / windowDuration, 0, 1);
      const speechSeconds = speechMsRef.current / 1000;
      const speakingRate = Math.round(90 + (1 - pauseRatio) * 80); // heuristic mapping to keep UI responsive

      setMetrics((prev) => ({
        ...prev,
        pauseRatio: Number(pauseRatio.toFixed(2)),
        speechSeconds: Number(speechSeconds.toFixed(2)),
        speakingRate,
        volume: Number(rms.toFixed(3)),
      }));

      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
  }, []);

  const runFaceLoop = useCallback(() => {
    const loop = async () => {
      const videoEl = videoRef.current;
      if (!videoEl) return;
      if (landmarkerRef.current) {
        const result = landmarkerRef.current.detectForVideo(videoEl, performance.now());
        if (result.faceLandmarks.length > 0 && videoEl.videoWidth > 0) {
          const lm = result.faceLandmarks[0];
          const xs = lm.map((p) => p.x);
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const centerX = (minX + maxX) / 2;
          const ratio = 1 - Math.min(1, Math.abs(centerX - 0.5) * 2);
          setMetrics((prev) => ({ ...prev, gaze: Math.round(ratio * 100) }));
        }
      }
      faceLoopRef.current = window.setTimeout(loop, 200);
    };
    loop();
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!active) {
      return () => {
        cancelled = true;
        stopAll();
      };
    }

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setStream(stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => undefined);
        }

        const audioCtx = new AudioContext();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.6;
        const dataArray = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(dataArray);
        analyserRef.current = analyser;
        dataArrayRef.current = dataArray;
        source.connect(analyser);

        runAudioLoop();

        try {
          const fileset = await FilesetResolver.forVisionTasks("/mediapipe");
          landmarkerRef.current = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: "/mediapipe/face_landmarker.task" },
            runningMode: "VIDEO",
            numFaces: 1,
          });
          runFaceLoop();
        } catch {
          setStreamError("Vision model failed to load; gaze metrics disabled.");
        }
      } catch {
        setStreamError("Microphone/camera access failed. Enable permissions to view live metrics.");
      }
    };

    start();
    return () => {
      cancelled = true;
      stopAll();
    };
  }, [active, runAudioLoop, runFaceLoop, stopAll]);

  return { metrics, videoRef, streamError, stream };
}

function useInterview() {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [style, setStyle] = useState<Style>("neutral");
  const [group, setGroup] = useState<"control" | "treatment">("treatment");
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tips, setTips] = useState<Tip[]>([]);
  const [turn, setTurn] = useState<number>(0);
  const [analytics, setAnalytics] = useState<Analytics>(initialAnalytics);
  const pendingStart = useRef<{ style: Style; group: "control" | "treatment"; consent: boolean; accent?: string; notes?: string } | null>(null);
  const latencyRef = useRef<Record<number, number>>({});

  const resetState = useCallback(() => {
    setMessages([]);
    setTips([]);
    setQuestion("");
    setTurn(0);
    setAnalytics(initialAnalytics);
    setSessionId(null);
    setGroup("treatment");
    setStatus("idle");
  }, []);

  const sendTelemetry = useCallback(
    (event: string, latencyMs?: number, data?: Record<string, unknown>) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: "telemetry", event, latencyMs, data }));
    },
    [socket],
  );

  const connect = useCallback(() => {
    setStatus("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setStatus("connected");
      if (pendingStart.current) {
        const { style: s, group: g, consent, accent, notes } = pendingStart.current;
        ws.send(JSON.stringify({ type: "start_session", style: s, group: g, consent, accent, notes }));
        setStatus("active");
        pendingStart.current = null;
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case "session_ready":
          setSessionId(data.session_id);
          setStyle(data.style ?? "neutral");
          if (data.group) setGroup(data.group);
          break;
        case "session_started":
          setSessionId(data.session_id);
          setStyle(data.style ?? "neutral");
          setTurn(data.turn ?? 0);
          if (data.group) setGroup(data.group);
          setStatus("active");
          break;
        case "question":
          setQuestion(data.question);
          setStyle(data.style ?? style);
          break;
        case "interviewer_message":
          setMessages((prev) => [
            ...prev,
            {
              role: "interviewer",
              content: data.message,
              turn: data.turn ?? prev.length,
              style: data.style ?? style,
            },
          ]);
          break;
        case "tips":
          setTips(data.items ?? []);
          if (typeof data.turn === "number" && latencyRef.current[data.turn] !== undefined) {
            const sent = latencyRef.current[data.turn];
            const latency = Date.now() - sent;
            sendTelemetry("latency", latency, { turn: data.turn });
            delete latencyRef.current[data.turn];
          }
          break;
        case "style_switched":
          setStyle(data.style ?? style);
          break;
        case "pong":
          break;
        case "error":
          setStatus("error");
          break;
        default:
          // Ignore unknown events
          break;
      }
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus("closed");
    };

    setSocket(ws);
  }, [sendTelemetry, style]);

  useEffect(() => {
    return () => {
      socket?.close();
    };
  }, [socket]);

  const start = useCallback(
    (initialStyle: Style, initialGroup: "control" | "treatment", consent: boolean, accent?: string, notes?: string) => {
      setStyle(initialStyle);
      setGroup(initialGroup);
      setTips([]);
      setMessages([]);
      setAnalytics(initialAnalytics);
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(
          JSON.stringify({ type: "start_session", style: initialStyle, group: initialGroup, consent, accent, notes }),
        );
        setStatus("active");
        return;
      }
      pendingStart.current = { style: initialStyle, group: initialGroup, consent, accent, notes };
      connect();
    },
    [connect, socket],
  );

  const stop = useCallback(() => {
    socket?.close();
    resetState();
    setStatus("closed");
  }, [resetState, socket]);

  const sendAnswer = useCallback(
    (answer: string, metricsOverride?: Analytics) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      const trimmed = answer.trim();
      if (!trimmed) return;
      setMessages((prev) => [...prev, { role: "user", content: trimmed, turn: prev.length }]);
      if (metricsOverride) {
        setAnalytics(metricsOverride);
      }
      latencyRef.current[turn] = Date.now();
      socket.send(JSON.stringify({ type: "user_answer", answer: trimmed, metrics: metricsOverride }));
    },
    [socket, turn],
  );

  const sendCheckIn = useCallback(
    (groupName: "control" | "treatment", confidence: number, stress: number) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: "checkin", group: groupName, confidence, stress }));
    },
    [socket],
  );

  const switchStyle = useCallback(
    (next: Style) => {
      setStyle(next);
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "switch_style", style: next }));
      }
    },
    [socket],
  );

  return {
    status,
    start,
    stop,
    style,
    setStyle,
    group,
    setGroup,
    switchStyle,
    question,
    messages,
    tips,
    sessionId,
    analytics,
    setAnalytics,
    turn,
    connect,
    socket,
    setQuestion,
    setMessages,
    setTips,
    sendAnswer,
    sendCheckIn,
    sendTelemetry,
  };
}

export default function Home() {
  const {
    status,
    start,
    stop,
    style,
    group,
    setGroup,
    switchStyle,
    question,
    messages,
    tips,
    sessionId,
    analytics,
    setAnalytics,
    sendAnswer,
    setMessages,
    sendCheckIn,
  } = useInterview();

  const [draft, setDraft] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(70);
  const [stress, setStress] = useState<number>(30);
  const { metrics: sensorMetrics, videoRef, streamError, stream: mediaStream } = useMediaSensors(status === "active");
  const [consent, setConsent] = useState<boolean>(false);
  const [accent, setAccent] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  const [exporting, setExporting] = useState<boolean>(false);
  const [exportStatus, setExportStatus] = useState<string>("");
  const [interviewerTalking, setInterviewerTalking] = useState<boolean>(false);
  const [autoSendVoice, setAutoSendVoice] = useState<boolean>(true);
  const [autoListen, setAutoListen] = useState<boolean>(true);
  const [recording, setRecording] = useState<boolean>(false);
  const [sttPending, setSttPending] = useState<boolean>(false);
  const [sttError, setSttError] = useState<string>("");
  const [ttsError, setTtsError] = useState<string>("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const ttsVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const silenceTimerRef = useRef<number | null>(null);
  const recordStartRef = useRef<number | null>(null);
  const lastVoiceRef = useRef<number | null>(null);
  const draftRef = useRef<string>("");
  const handleSendRef = useRef<() => void>(() => {});
  const lastSpokenQuestionRef = useRef<string>("");

  const disabled = useMemo(() => status === "connecting" || status === "error", [status]);
  const userTalking = useMemo(() => (sensorMetrics.volume || 0) > 0.045, [sensorMetrics.volume]);
  const listening = recording || sttPending;

  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);

  useEffect(() => {
    if (status !== "active") return;
    setAnalytics((prev) => ({
      ...prev,
      speakingRate: sensorMetrics.speakingRate || prev.speakingRate,
      pauseRatio: sensorMetrics.pauseRatio,
      gaze: sensorMetrics.gaze || prev.gaze,
      volume: sensorMetrics.volume,
      speechSeconds: sensorMetrics.speechSeconds ?? prev.speechSeconds,
    }));
  }, [sensorMetrics, setAnalytics, status]);

  useEffect(() => {
    return () => {
      if (ttsSourceRef.current) {
        try {
          ttsSourceRef.current.stop();
        } catch {
          // ignore
        }
      }
      ttsAbortRef.current?.abort();
    };
  }, []);

  const buildMetrics = useCallback(
    (text: string): Analytics => {
      const words = text.split(/\s+/).filter(Boolean);
      const fillerCount = words.filter((w) => ["um", "uh", "like"].includes(w.toLowerCase())).length;
      const speechSeconds =
        sensorMetrics.speechSeconds && sensorMetrics.speechSeconds > 0
          ? sensorMetrics.speechSeconds
          : Math.max(words.length / 2.5, 0.5);
      const speakingRate =
        speechSeconds > 0 ? Math.round((words.length / speechSeconds) * 60) : sensorMetrics.speakingRate || 0;
      return {
        ...sensorMetrics,
        fillers: fillerCount,
        speakingRate,
        pauseRatio: sensorMetrics.pauseRatio,
        gaze: sensorMetrics.gaze || analytics.gaze,
        speechSeconds,
      };
    },
    [analytics.gaze, sensorMetrics],
  );

  const sendToStt = useCallback(
    async (blob: Blob) => {
      setSttPending(true);
      setSttError("");
      try {
        const form = new FormData();
        form.append("file", blob, "answer.webm");
        if (sessionId) form.append("sessionId", sessionId);
        const res = await fetch(`${HTTP_BASE}/stt`, {
          method: "POST",
          body: form,
        });
        const json = await res.json();
        if (!res.ok || json.error) {
          throw new Error(json.error || "Transcription failed");
        }
        const transcript = (json.transcript as string) || "";
        setDraft(transcript);
        draftRef.current = transcript;
        if (autoSendVoice && transcript.trim()) {
          const finalMetrics = buildMetrics(transcript);
          setAnalytics(finalMetrics);
          sendAnswer(transcript, finalMetrics);
        }
      } catch (err) {
        setSttError("Transcription failed. Try again.");
      } finally {
        setSttPending(false);
      }
    },
    [autoSendVoice, buildMetrics, sendAnswer, sessionId, setAnalytics],
  );

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
    setRecording(false);
    recordStartRef.current = null;
    lastVoiceRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    if (recording || sttPending) return;
    setSttError("");
    try {
      const readyStream =
        (mediaStream && mediaStream.getAudioTracks().length > 0 ? mediaStream : null) ||
        (await navigator.mediaDevices.getUserMedia({
          audio: true,
          video: false,
        }));
      if (!readyStream || readyStream.getAudioTracks().length === 0) {
        throw new Error("no-audio-track");
      }
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : undefined;
      if (typeof MediaRecorder === "undefined") {
        throw new Error("mediarecorder-unsupported");
      }
      const recorder = new MediaRecorder(readyStream, mimeType ? { mimeType } : undefined);
      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        audioChunksRef.current = [];
        if (blob.size > 0) {
          void sendToStt(blob);
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      recordStartRef.current = performance.now();
      lastVoiceRef.current = recordStartRef.current;
      setRecording(true);
    } catch (err) {
      const message =
        err instanceof Error && err.message === "mediarecorder-unsupported"
          ? "Recording not supported in this browser. Please use Chrome/Edge."
          : "Microphone access failed. Confirm mic permission or close other apps using the mic.";
      setSttError(message);
      setRecording(false);
    }
  }, [mediaStream, recording, sendToStt, sttPending]);

  useEffect(() => {
    if (status !== "active") {
      stopRecording();
    }
  }, [status, stopRecording]);

  useEffect(() => {
    if (!recording) {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      return;
    }
    const now = performance.now();
    const startTs = recordStartRef.current ?? now;
    if (userTalking) {
      lastVoiceRef.current = now;
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      return;
    }

    const lastVoiceTs = lastVoiceRef.current ?? startTs;
    const elapsedMs = now - startTs;
    const silenceMs = now - lastVoiceTs;
    if (elapsedMs < MIN_RECORDING_MS) {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      return;
    }

    if (silenceMs >= SILENCE_TIMEOUT_MS) {
      stopRecording();
      return;
    }

    if (!silenceTimerRef.current) {
      const remaining = Math.max(SILENCE_TIMEOUT_MS - silenceMs, 400);
      silenceTimerRef.current = window.setTimeout(() => {
        stopRecording();
      }, remaining);
    }

    return () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    };
  }, [MIN_RECORDING_MS, SILENCE_TIMEOUT_MS, recording, stopRecording, userTalking]);

  const fallbackSpeak = useCallback(
    (text: string, onDone?: () => void) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        onDone?.();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = style === "cold" ? 0.95 : style === "supportive" ? 1.05 : 1;
      utterance.pitch = style === "supportive" ? 1.05 : style === "cold" ? 0.92 : 1;
      const voices = window.speechSynthesis.getVoices();
      const preferred =
        voices.find((v) => /female|woman|samantha|allison|ava|en-us/i.test(v.name)) ||
        voices.find((v) => /en-/i.test(v.lang)) ||
        voices[0];
      if (preferred) utterance.voice = preferred;
      utterance.onstart = () => setInterviewerTalking(true);
      utterance.onend = () => {
        setInterviewerTalking(false);
        onDone?.();
      };
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    },
    [style],
  );

  const speakQuestion = useCallback(
    async (text: string, onDone?: () => void) => {
      if (!text) {
        onDone?.();
        return;
      }
      setTtsError("");
      ttsAbortRef.current?.abort();
      const controller = new AbortController();
      ttsAbortRef.current = controller;
      setInterviewerTalking(true);
      try {
        const res = await fetch(`${HTTP_BASE}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, style }),
          signal: controller.signal,
        });
        const contentType = res.headers.get("content-type") || "";
        if (!res.ok || !contentType.includes("audio")) {
          let msg = "tts_failed";
          try {
            const json = await res.json();
            msg = json.error || msg;
          } catch {
            // ignore
          }
          throw new Error(msg);
        }
        const arrayBuffer = await res.arrayBuffer();
        const ctx = audioCtxRef.current ?? new AudioContext();
        audioCtxRef.current = ctx;
        if (ctx.state === "suspended") {
          await ctx.resume();
        }
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
        await new Promise<void>((resolve) => {
          const src = ctx.createBufferSource();
          ttsSourceRef.current = src;
          src.buffer = audioBuffer;
          src.connect(ctx.destination);
          src.onended = () => {
            setInterviewerTalking(false);
            onDone?.();
            resolve();
          };
          src.start(0);
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "tts_failed";
        setTtsError(`TTS failed (${msg}); using fallback voice.`);
        fallbackSpeak(text, onDone);
        setInterviewerTalking(false);
      }
    },
    [fallbackSpeak, style],
  );

  useEffect(() => {
    if (status !== "active" || !question) {
      if (status !== "active") lastSpokenQuestionRef.current = "";
      return;
    }
    if (lastSpokenQuestionRef.current === question) return; // dedupe same question re-renders
    lastSpokenQuestionRef.current = question;
    stopRecording();
    speakQuestion(question, () => {
      if (autoListen) {
        startRecording();
      }
    });
  }, [autoListen, question, speakQuestion, startRecording, status, stopRecording]);

  const handleSend = useCallback(() => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    const finalMetrics = buildMetrics(trimmed);
    setAnalytics(finalMetrics);
    sendAnswer(trimmed, finalMetrics);
    setDraft("");
  }, [buildMetrics, draft, sendAnswer, setAnalytics]);

  useEffect(() => {
    handleSendRef.current = handleSend;
  }, [handleSend]);

  const logCheckIn = useCallback(() => {
    sendCheckIn(group, confidence, stress);
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: `Check-in (${group}): confidence ${confidence}/100, stress ${stress}/100.`,
        turn: prev.length,
      },
    ]);
  }, [confidence, group, sendCheckIn, setMessages, stress]);

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>BG Hacker • Cognitive Ergonomics</p>
          <h1 className={styles.title}>Multi-style AI Interview Trainer</h1>
          <p className={styles.subtitle}>
            Practice with supportive, neutral, and cold interviewer personas. Get glanceable feedback on delivery,
            pacing, and adaptability.
          </p>
        </div>
        <div className={styles.sessionMeta}>
          <div className={styles.badge}>{status.toUpperCase()}</div>
          <div className={styles.sessionId}>{sessionId ? `Session: ${sessionId.slice(0, 8)}` : "No session yet"}</div>
          <div className={styles.styleHint}>{STYLE_HELP[style]}</div>
        </div>
      </header>

      <section className={styles.controls}>
        <div className={styles.styleSwitch}>
          {(["supportive", "neutral", "cold"] as Style[]).map((s) => (
            <button
              key={s}
              className={`${styles.styleButton} ${style === s ? styles.styleButtonActive : ""}`}
              onClick={() => switchStyle(s)}
              aria-pressed={style === s}
            >
              <span className={styles.styleLabel}>{STYLE_LABELS[s]}</span>
              <span className={styles.styleSub}>{STYLE_HELP[s]}</span>
            </button>
          ))}
        </div>
        <div className={styles.sessionControls}>
          <button className={styles.primary} disabled={disabled || !consent} onClick={() => start(style, group, consent, accent, notes)}>
            {status === "active" ? "Restart" : "Start Interview"}
          </button>
          <button className={styles.secondary} onClick={stop} disabled={status === "idle"}>
            Stop
          </button>
          <button
            className={styles.secondary}
            disabled={!sessionId || exporting}
            onClick={async () => {
              if (!sessionId) return;
              setExporting(true);
              try {
                const res = await fetch(`${HTTP_BASE}/export/session/${sessionId}`);
                const json = await res.json();
                const blob = new Blob([JSON.stringify(json, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `session-${sessionId}.json`;
                a.click();
                URL.revokeObjectURL(url);
                setExportStatus("Exported session JSON.");
              } catch {
                setExportStatus("Export failed.");
              } finally {
                setExporting(false);
              }
            }}
          >
            {exporting ? "Exporting..." : "Export session JSON"}
          </button>
        </div>
      </section>

      <section className={styles.consentCard}>
        <div>
          <p className={styles.kicker}>Consent & fairness</p>
          <h3 className={styles.cardTitle}>Data handling agreement</h3>
          <p className={styles.subtitle}>
            We store session metrics (answers, delivery signals, latency) locally for coaching and aggregate analysis.
            No emotion inference; you can export/delete your session data.
          </p>
          <label className={styles.checkboxRow}>
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>I consent to capture and local processing of mic/cam signals for this session.</span>
          </label>
          <div className={styles.fairnessRow}>
            <label>
              Accent / dialect (optional)
              <input
                type="text"
                value={accent}
                onChange={(e) => setAccent(e.target.value)}
                placeholder="e.g., US Southern, Nigerian English"
              />
            </label>
            <label>
              Notes (optional)
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any accessibility or fairness notes"
              />
            </label>
          </div>
          <p className={styles.helperText}>
            Accent metadata helps audit ASR latency/fairness. Export anytime with “Export session JSON.” {exportStatus}
          </p>
        </div>
      </section>

      <section className={styles.columns}>
        <div className={styles.leftColumn}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <p className={styles.kicker}>Voice-led interview</p>
              <h3 className={styles.cardTitle}>Avatar & mic flow</h3>
            </div>
            <div className={styles.avatarRow}>
              <Avatar style={style} talking={interviewerTalking} userTalking={userTalking} listening={listening} />
              <div className={styles.voicePanel}>
                <div className={styles.voiceStatus}>
                  Questions are spoken in her adaptive voice (Coqui TTS). {ttsError && ttsError}
                </div>
                <div className={styles.voiceControls}>
                  <button
                    className={styles.secondary}
                    disabled={status !== "active" || sttPending}
                    onClick={() => (recording ? stopRecording() : startRecording())}
                  >
                    {recording ? "Stop & transcribe" : "Tap to answer"}
                  </button>
                  <button
                    className={styles.secondary}
                    disabled={!question || sttPending}
                    onClick={() => {
                      stopRecording();
                      speakQuestion(question);
                    }}
                  >
                    Replay question
                  </button>
                </div>
                <div className={styles.voiceToggles}>
                  <label className={styles.checkboxRow}>
                    <input type="checkbox" checked={autoListen} onChange={(e) => setAutoListen(e.target.checked)} />
                    <span>Auto-listen after each question</span>
                  </label>
                  <label className={styles.checkboxRow}>
                    <input type="checkbox" checked={autoSendVoice} onChange={(e) => setAutoSendVoice(e.target.checked)} />
                    <span>Auto-send when speech finishes</span>
                  </label>
                </div>
                <p className={styles.helperText}>
                  Mic pickup drives analytics and fills the transcript; typing is optional. Audio is sent to the local
                  Whisper model for transcription. Keep the text box as fallback.
                </p>
              </div>
            </div>
          </div>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div>
                <p className={styles.kicker}>Current prompt</p>
                <h3 className={styles.cardTitle}>{question || "Click Start to get your first question"}</h3>
              </div>
              <div className={styles.pill}>{STYLE_LABELS[style]}</div>
            </div>
            <textarea
              className={styles.answerBox}
              placeholder="Speak your answer (auto-filled). Type here only if you prefer."
              value={draft}
              disabled={status !== "active"}
              onChange={(e) => setDraft(e.target.value)}
              rows={5}
            />
            <div className={styles.listeningStatus}>
              <span className={listening ? styles.dotLive : styles.dotIdle} />
              {listening
                ? sttPending
                  ? "Transcribing…"
                  : "Recording — just speak naturally."
                : "Not recording. Tap to answer or wait for auto-listen."}
            </div>
            {sttError && <div className={styles.errorText}>{sttError}</div>}
            <div className={styles.answerActions}>
              <button className={styles.primary} onClick={handleSend} disabled={!draft.trim() || status !== "active"}>
                Send answer
              </button>
              <div className={styles.helperText}>
                Live answers auto-send when you stop talking. Voice latency is logged for fairness audits.
              </div>
            </div>
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <p className={styles.kicker}>Real-time signals</p>
              <h3 className={styles.cardTitle}>Glanceable HUD</h3>
            </div>
            <div className={styles.metricsGrid}>
              <Metric label="Speaking rate" value={`${analytics.speakingRate || "—"} wpm`} target="120-160" />
              <Metric label="Pause ratio" value={`${analytics.pauseRatio}`} target="<0.12" />
              <Metric label="Gaze on camera" value={`${analytics.gaze}%`} target="70%+" />
              <Metric label="Fillers (last answer)" value={`${analytics.fillers}`} target="keep ≤1" />
            </div>
            <div className={styles.videoRow}>
              <div className={styles.videoContainer}>
                <video ref={videoRef} className={styles.video} muted playsInline />
              </div>
              <div className={styles.metricHint}>
                Live camera/audio power adaptive coaching. Face detection (if supported) approximates on-camera gaze; audio
                RMS drives pause ratio and rate. {streamError || "Grant mic/cam permissions to unlock live telemetry."}
              </div>
            </div>
            <div className={styles.metricHint}>
              Feedback mixes modalities: color band for rate, subtle chime for filler spikes, compact cards after each
              turn.
            </div>
          </div>
        </div>

        <div className={styles.rightColumn}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <p className={styles.kicker}>Conversation</p>
              <h3 className={styles.cardTitle}>Transcript & moves</h3>
            </div>
            <div className={styles.feed}>
              {messages.length === 0 && <p className={styles.placeholder}>Answers and follow-ups will appear here.</p>}
              {messages.map((m, idx) => (
                <div key={`${m.role}-${idx}`} className={styles.message}>
                  <div className={styles.messageMeta}>
                    <span className={styles.messageRole}>{m.role === "user" ? "You" : "Interviewer"}</span>
                    {m.style && <span className={styles.messageStyle}>{STYLE_LABELS[m.style]}</span>}
                    <span className={styles.messageTurn}>Turn {m.turn + 1}</span>
                  </div>
                  <p className={styles.messageText}>{m.content}</p>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <p className={styles.kicker}>Actionable tips</p>
              <h3 className={styles.cardTitle}>Style-aware coaching</h3>
            </div>
            {tips.length === 0 ? (
              <p className={styles.placeholder}>Send an answer to see concise recommendations.</p>
            ) : (
              <ul className={styles.tipList}>
                {tips.map((tip, i) => (
                  <li key={`${tip.summary}-${i}`} className={styles.tipItem}>
                    <div className={styles.tipSummary}>{tip.summary}</div>
                    <div className={styles.tipDetail}>{tip.detail}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <p className={styles.kicker}>Experiment flow</p>
              <h3 className={styles.cardTitle}>Check-ins (pre / post)</h3>
            </div>
            <div className={styles.chipRow}>
              {["control", "treatment"].map((g) => (
                <button
                  key={g}
                  className={`${styles.chip} ${group === g ? styles.chipActive : ""}`}
                  onClick={() => setGroup(g as "control" | "treatment")}
                >
                  {g === "control" ? "Control: neutral only" : "Treatment: multi-style"}
                </button>
              ))}
            </div>
            <div className={styles.sliderRow}>
              <label>
                Confidence {confidence}/100
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                />
              </label>
              <label>
                Stress {stress}/100
                <input type="range" min={0} max={100} value={stress} onChange={(e) => setStress(Number(e.target.value))} />
              </label>
            </div>
            <button className={styles.secondary} onClick={logCheckIn}>
              Log check-in
            </button>
            <p className={styles.helperText}>
              Use this during pre-test, training, and post-tests to track adaptability and perceived readiness.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

function Avatar({
  style,
  talking,
  userTalking,
  listening,
}: {
  style: Style;
  talking: boolean;
  userTalking: boolean;
  listening: boolean;
}) {
  const moodClass = {
    supportive: styles.avatarSupportive,
    neutral: styles.avatarNeutral,
    cold: styles.avatarCold,
  }[style];
  const mouthClass = talking ? styles.mouthTalking : userTalking ? styles.mouthUser : styles.mouthIdle;
  const status = talking ? "Interviewer speaking" : listening ? "Listening to you" : userTalking ? "You’re speaking" : "Idle";

  return (
    <div className={`${styles.avatarShell} ${moodClass}`}>
      <div className={styles.avatarFace}>
        <div className={`${styles.eye} ${styles.eyeLeft}`} />
        <div className={`${styles.eye} ${styles.eyeRight}`} />
        <div className={`${styles.brow} ${styles.browLeft}`} />
        <div className={`${styles.brow} ${styles.browRight}`} />
        <div className={`${styles.mouth} ${mouthClass}`} />
        {listening && <div className={styles.listenGlow} />}
      </div>
      <div className={styles.avatarStatusText}>{status}</div>
    </div>
  );
}

function Metric({ label, value, target }: { label: string; value: string | number; target?: string }) {
  return (
    <div className={styles.metric}>
      <p className={styles.metricLabel}>{label}</p>
      <div className={styles.metricValue}>{value}</div>
      {target && <div className={styles.metricTarget}>Target: {target}</div>}
    </div>
  );
}
