"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export type Style = "supportive" | "neutral" | "cold";

export type ChatMessage = {
  role: "interviewer" | "user";
  content: string;
  turn: number;
  style?: Style;
};

export type Tip = { summary: string; detail: string };

export type SessionStatus = "idle" | "connecting" | "connected" | "active" | "error" | "closed";

export type Analytics = {
  speakingRate: number; // words per minute (estimate)
  pauseRatio: number; // fraction 0-1
  gaze: number; // percent
  fillers: number; // count in last answer
  speechSeconds?: number; // rolling speech duration used for rate calculations
  volume?: number; // normalized RMS (0-1)
};

export const initialAnalytics: Analytics = {
  speakingRate: 0,
  pauseRatio: 0,
  gaze: 0,
  fillers: 0,
  speechSeconds: 0,
  volume: 0,
};

export const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/interview";
export const HTTP_BASE = (WS_URL.startsWith("ws") ? WS_URL.replace(/^ws/, "http") : WS_URL).replace(/\/ws\/interview$/, "");

export const STYLE_LABELS: Record<Style, string> = {
  supportive: "Supportive",
  neutral: "Neutral",
  cold: "Cold / Pressuring",
};

export const STYLE_HELP: Record<Style, string> = {
  supportive: "Warm tone, short pauses, light encouragement.",
  neutral: "Balanced prompts, steady pacing.",
  cold: "Direct challenges, strategic silence, blunt follow-ups.",
};

export const SILENCE_TIMEOUT_MS = 2600;
export const MIN_RECORDING_MS = 1200;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function useMediaSensors(active: boolean) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [metrics, setMetrics] = useState<Analytics>(initialAnalytics);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
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
        const dataArray = new Uint8Array(analyser.fftSize) as Uint8Array<ArrayBuffer>;
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

export function useInterview() {
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
  const pendingStart = useRef<{
    style: Style;
    group: "control" | "treatment";
    consent: boolean;
    accent?: string;
    notes?: string;
    pack?: string;
    difficulty?: string;
  } | null>(null);
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
        const { style: s, group: g, consent, accent, notes, pack, difficulty } = pendingStart.current;
        ws.send(JSON.stringify({ type: "start_session", style: s, group: g, consent, accent, notes, pack, difficulty }));
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
          if (typeof data.turn === "number") setTurn(data.turn);
          if (data.source !== "follow_up" && data.question) {
            setMessages((prev) => {
              const nextTurn = typeof data.turn === "number" ? data.turn : prev.length;
              const last = prev.length > 0 ? prev[prev.length - 1] : null;
              if (last && last.role === "interviewer" && last.content === data.question && last.turn === nextTurn) return prev;
              return [
                ...prev,
                {
                  role: "interviewer",
                  content: data.question,
                  turn: nextTurn,
                  style: data.style ?? style,
                },
              ];
            });
          }
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
    (
      initialStyle: Style,
      initialGroup: "control" | "treatment",
      consent: boolean,
      accent?: string,
      notes?: string,
      pack?: string,
      difficulty?: string,
    ) => {
      setStyle(initialStyle);
      setGroup(initialGroup);
      setTips([]);
      setMessages([]);
      setAnalytics(initialAnalytics);
      setQuestion("");
      setTurn(0);
      setSessionId(null);
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "start_session", style: initialStyle, group: initialGroup, consent, accent, notes, pack, difficulty }));
        setStatus("active");
        return;
      }
      pendingStart.current = { style: initialStyle, group: initialGroup, consent, accent, notes, pack, difficulty };
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
      setMessages((prev) => [...prev, { role: "user", content: trimmed, turn, style }]);
      if (metricsOverride) {
        setAnalytics(metricsOverride);
      }
      const answerTurn = turn + 1;
      latencyRef.current[answerTurn] = Date.now();
      socket.send(JSON.stringify({ type: "user_answer", answer: trimmed, metrics: metricsOverride }));
    },
    [socket, style, turn],
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
