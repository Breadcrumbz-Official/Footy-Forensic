import { useState, useRef, useCallback, useEffect, useLayoutEffect } from "react";
import {
  Upload, Camera, ChevronRight, ChevronLeft, CheckCircle2, RotateCcw,
  Play, Pause, Zap, AlertCircle, TrendingUp, AlertTriangle,
  Info, SwitchCamera, Square, X, Sparkles
} from "lucide-react";

import * as api from "./api";
import type { AnalyseSpec, PhaseKey } from "./api";
import { PHASE_KEYS } from "./api";
import {
  toDisplay, scoreColor, PHASE_COLOR, PHASE_SHORT, PHASE_HINT,
} from "./results";
import type { DisplayMetric, DisplayResults } from "./results";
import { Recorder, ensureFiniteDuration, REC_LIMIT_MS, REC_MIN_MS } from "./recorder";
import { ReelScrubber } from "./ReelScrubber";
import type { ReelMarker } from "./ReelScrubber";

type Step = "upload" | "select" | "analyzing" | "results";


const WINDOW_DEFAULT = 0.2;
const WINDOW_MIN = 0.08;
const WINDOW_MAX = 1.0;
const WINDOW_STEP = 0.02;

const SEGMENT_CONFIG: { key: PhaseKey; label: string; color: string; shortLabel: string }[] = [
  { key: "plant", label: "Plant Foot", shortLabel: PHASE_SHORT.plant, color: PHASE_COLOR.plant },
  { key: "contact", label: "Ball Contact", shortLabel: PHASE_SHORT.contact, color: PHASE_COLOR.contact },
  { key: "followThrough", label: "Follow Through", shortLabel: PHASE_SHORT.followThrough, color: PHASE_COLOR.followThrough },
];

function ScoreRing({ score, size = 96, stroke = 7, color }: { score: number; size?: number; stroke?: number; color: string }) {
  const r = (size - stroke * 2) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / 100);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke={color} strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        style={{ transition: "stroke-dashoffset 1s ease-out" }}
      />
    </svg>
  );
}

function getStatusDot(status: DisplayMetric["status"]) {
  if (status === "good") return "bg-[#00df54]";
  if (status === "warn") return "bg-[#ffb800]";
  if (status === "bad") return "bg-[#e03c3c]";
  return "bg-muted-foreground/40";
}

const PX_PER_SECOND = 130;

export default function App() {
  const [step, setStep] = useState<Step>("upload");
  const [videoUrl, setVideoUrl] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(8);
  const [footedness, setFootedness] = useState<AnalyseSpec["footedness"]>("auto");
  const [sourceFps, setSourceFps] = useState(30);

  const [uploadPct, setUploadPct] = useState(0);
  const [stage, setStage] = useState<"upload" | "analyze">("upload");
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<DisplayResults | null>(null);

  const [selectPage, setSelectPage] = useState(0);
  const [segments, setSegments] = useState<Record<PhaseKey, number>>({
    plant: 1.4,
    contact: 3.2,
    followThrough: 5.6,
  });
  const [expandedPhase, setExpandedPhase] = useState<PhaseKey | null>(null);
  const [thumbnails, setThumbnails] = useState<string[]>([]);
  const [thumbsLoading, setThumbsLoading] = useState(false);
  
  
  const [windows, setWindows] = useState<Record<PhaseKey, number>>({
    plant: WINDOW_DEFAULT, contact: WINDOW_DEFAULT, followThrough: WINDOW_DEFAULT,
  });
  const [touched, setTouched] = useState<Record<PhaseKey, boolean>>({
    plant: false, contact: false, followThrough: false,
  });
  const [reelWidth, setReelWidth] = useState(0);

  
  const [health, setHealth] = useState<api.Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  
  const [camOpen, setCamOpen] = useState(false);
  const [camReady, setCamReady] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [facing, setFacing] = useState<"environment" | "user">("environment");

  
  const videoRef = useRef<HTMLVideoElement>(null);
  const thumbVideoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const thumbsExtractedFor = useRef<string>("");
  const camPreviewRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<Recorder | null>(null);
  const seekRaf = useRef<number | null>(null);
  const pendingSeek = useRef<number | null>(null);
  
  
  const cancelledRef = useRef(false);

  useEffect(() => {
    api.checkHealth()
      .then(h => { setHealth(h); setHealthError(null); })
      .catch(err => setHealthError(err.message));
  }, []);

  
  const handleFile = useCallback((file: File, fps = 30) => {
    if (!file.type.startsWith("video/")) return;
    setSourceFps(fps);
    
    
    cancelledRef.current = true;
    recorderRef.current?.release();
    recorderRef.current = null;
    setCamOpen(false);
    setCamReady(false);

    const url = URL.createObjectURL(file);
    setVideoFile(file);
    setVideoUrl(url);
    setError(null);
    setStep("select");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  
  const releaseCamera = useCallback(() => {
    recorderRef.current?.release();
    recorderRef.current = null;
    if (camPreviewRef.current) camPreviewRef.current.srcObject = null;
    setCamReady(false);
    setIsRecording(false);
    setElapsed(0);
  }, []);

  
  useEffect(() => () => {
    cancelledRef.current = true;
    releaseCamera();
  }, [releaseCamera]);

  const openCamera = useCallback(async () => {
    setCamError(null);
    setCamOpen(true);
    if (!Recorder.supported()) {
      setCamError("This browser cannot record video. Upload a file instead.");
      return;
    }
    
    
    if (!window.isSecureContext) {
      setCamError(
        "Recording needs a secure connection. Open this page over https (or on localhost) to use the camera.");
      return;
    }
    try {
      const rec = recorderRef.current ?? new Recorder();
      recorderRef.current = rec;
      
      
      await new Promise(requestAnimationFrame);
      if (!camPreviewRef.current) return;
      await rec.enableCamera(camPreviewRef.current);
      setCamReady(true);
    } catch (err) {
      const e = err as Error;
      setCamError(
        e.name === "NotAllowedError"
          ? "Camera permission was denied. Allow access in your browser's site settings and try again."
          : e.name === "NotFoundError"
            ? "No camera found on this device."
            : `Could not start the camera. ${e.message}`);
    }
  }, []);

  const closeCamera = useCallback(() => {
    cancelledRef.current = true;   
    releaseCamera();
    setCamOpen(false);
    setCamError(null);
  }, [releaseCamera]);

  const flipCamera = useCallback(async () => {
    if (!recorderRef.current || !camPreviewRef.current || isRecording) return;
    try {
      setFacing(await recorderRef.current.switchCamera(camPreviewRef.current));
      setCamError(null);
    } catch {
      setCamError("This device has only one camera.");
    }
  }, [isRecording]);

  const startRecording = useCallback(async () => {
    const rec = recorderRef.current;
    if (!rec || isRecording) return;
    setCamError(null);
    setIsRecording(true);
    setElapsed(0);
    cancelledRef.current = false;
    
    const trackFps = rec.stream?.getVideoTracks()[0]?.getSettings().frameRate;
    const fps = Number.isFinite(trackFps) && (trackFps as number) > 0 ? (trackFps as number) : 30;
    try {
      
      const blob = await rec.start(setElapsed);
      setIsRecording(false);
      if (cancelledRef.current) return;   
      releaseCamera();
      setCamOpen(false);
      
      
      const ext = blob.type.includes("mp4") ? "mp4" : "webm";
      handleFile(new File([blob], `kick.${ext}`, { type: blob.type || "video/webm" }), fps);
    } catch (err) {
      setIsRecording(false);
      setCamError((err as Error).message);
    }
  }, [isRecording, releaseCamera, handleFile]);

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop();
  }, []);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) videoRef.current.pause();
    else videoRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const activeKey = SEGMENT_CONFIG[selectPage].key;
  const activeWindow = windows[activeKey];

  
  const reelMarkers: ReelMarker[] = SEGMENT_CONFIG.map(seg => ({
    key: seg.key,
    label: seg.shortLabel,
    color: seg.color,
    start: segments[seg.key],
    end: segments[seg.key] + windows[seg.key],
    active: seg.key === activeKey,
    set: touched[seg.key],
  }));

  
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        setCurrentTime(v.currentTime);
        
        
        const t = Math.min(v.currentTime, Math.max(0, duration - activeWindow));
        setSegments(prev => (Math.abs(prev[activeKey] - t) < 1e-3 ? prev : { ...prev, [activeKey]: t }));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, activeKey, activeWindow, duration]);

  const handleTimeUpdate = () => {
    if (videoRef.current && !isPlaying) setCurrentTime(videoRef.current.currentTime);
  };

  
  const handleLoadedMetadata = async () => {
    if (!videoRef.current) return;
    const d = await ensureFiniteDuration(videoRef.current);
    if (!Number.isFinite(d) || d <= 0) return;
    setDuration(d);
    setSegments({
      plant: d * 0.18,
      contact: d * 0.42,
      followThrough: d * 0.70,
    });
  };

  const handleThumbVideoLoaded = async () => {
    const el = thumbVideoRef.current;
    if (!el) return;
    
    
    const own = await ensureFiniteDuration(el);
    const d = Number.isFinite(own) && own > 0 ? own : duration;
    extractThumbnails(d);
  };

  
  const seekTo = useCallback((t: number) => {
    setCurrentTime(t);
    pendingSeek.current = t;
    if (seekRaf.current !== null) return;
    const pump = () => {
      const v = videoRef.current;
      const target = pendingSeek.current;
      if (!v || target === null) { seekRaf.current = null; return; }
      if (v.seeking) { seekRaf.current = requestAnimationFrame(pump); return; }
      pendingSeek.current = null;
      seekRaf.current = null;
      
      
      if (typeof v.fastSeek === "function") v.fastSeek(target);
      else v.currentTime = target;
    };
    seekRaf.current = requestAnimationFrame(pump);
  }, []);

  useEffect(() => () => {
    if (seekRaf.current !== null) cancelAnimationFrame(seekRaf.current);
  }, []);

  
  const seekAndWait = (video: HTMLVideoElement, t: number) =>
    new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      video.addEventListener("seeked", onSeeked);
      video.currentTime = t;
    });

  
  const extractThumbnails = useCallback(async (d: number) => {
    const video = thumbVideoRef.current;
    if (!video || !Number.isFinite(d) || d <= 0) return;
    if (thumbsExtractedFor.current === videoUrl) return;
    thumbsExtractedFor.current = videoUrl;

    setThumbsLoading(true);
    const count = Math.min(90, Math.max(24, Math.round(d * 6)));
    const aspect = (video.videoWidth && video.videoHeight) ? video.videoWidth / video.videoHeight : 16 / 9;
    const canvas = document.createElement("canvas");
    canvas.height = 160;
    canvas.width = Math.round(160 * aspect);
    const ctx = canvas.getContext("2d");
    const frames: string[] = new Array(count).fill("");
    setThumbnails([...frames]);

    try {
      for (let i = 0; i < count; i++) {
        const t = (d * i) / (count - 1);
        await seekAndWait(video, Math.min(t, Math.max(0, d - 0.01)));
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          frames[i] = canvas.toDataURL("image/jpeg", 0.6);
        }
        setThumbnails([...frames]);
      }
    } finally {
      setThumbsLoading(false);
    }
  }, [videoUrl]);

  
  const startAnalysis = async () => {
    if (!videoFile) return;
    setStep("analyzing");
    setStage("upload");
    setUploadPct(0);
    setError(null);

    const spec: AnalyseSpec = {
      phases: {
        plant: { time: 0, clip: { start: 0, end: 0 } },
        contact: { time: 0, clip: { start: 0, end: 0 } },
        followThrough: { time: 0, clip: { start: 0, end: 0 } },
      },
      fps: sourceFps,
      footedness,
      duration,
    };
    for (const key of PHASE_KEYS) {
      const start = Math.max(0, Math.min(segments[key], Math.max(0, duration - windows[key])));
      const end = Math.min(start + windows[key], duration);
      spec.phases[key] = { time: (start + end) / 2, clip: { start, end } };
    }

    try {
      const raw = await api.analyze(videoFile, spec, (pct, which) => {
        setStage(which);
        if (which === "upload") setUploadPct(pct);
      });
      setResults(toDisplay(raw));
      setStep("results");
    } catch (err) {
      setError((err as Error).message);
      setStep("select");
    }
  };

  const reset = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl("");
    setVideoFile(null);
    setStep("upload");
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(8);
    setSelectPage(0);
    setExpandedPhase(null);
    setThumbnails([]);
    setThumbsLoading(false);
    setTouched({ plant: false, contact: false, followThrough: false });
    setWindows({ plant: WINDOW_DEFAULT, contact: WINDOW_DEFAULT, followThrough: WINDOW_DEFAULT });
    setResults(null);
    setError(null);
    setUploadPct(0);
    setCamOpen(false);
    setCamError(null);
    releaseCamera();
    thumbsExtractedFor.current = "";
  };

  const fmt = (t: number) => {
    const s = Math.floor(t);
    const cs = Math.floor((t - s) * 100);
    return `${s}.${cs.toString().padStart(2, "0")}s`;
  };

  return (
    <div
      className="min-h-screen bg-background text-foreground"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      
      {step === "upload" && (
        <div className="max-w-5xl mx-auto px-6">
          
          <div className="pt-20 pb-16 text-center">
            <h1
              className="text-foreground leading-none mb-6"
              style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: "clamp(3rem, 10vw, 7rem)", fontWeight: 900, letterSpacing: "-0.01em" }}
            >
              PERFECT<br />
              <span className="text-primary">YOUR SHOT</span>
            </h1>
            <p className="text-muted-foreground text-lg max-w-lg mx-auto leading-relaxed">
              Upload a short video, mark your plant, contact, and follow-through moments.
              Your kick is measured against a biomechanical rule set and scored.
            </p>
          </div>

          
          <div className="mb-6 flex items-center justify-center">
            {health ? (
              <span
                className="inline-flex items-center gap-2 text-[11px] tracking-widest text-muted-foreground border border-border px-3 py-1.5 rounded-full"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-[#00df54]" />
                ANALYSIS SERVER READY · {health.workers} WORKERS · MAX {health.maxUploadMb}MB
              </span>
            ) : (
              <span
                className="inline-flex items-center gap-2 text-[11px] tracking-widest text-[#e03c3c] border border-[#e03c3c]/30 px-3 py-1.5 rounded-full"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                <AlertTriangle className="w-3 h-3" />
                {healthError ? "ANALYSIS SERVER UNREACHABLE" : "CHECKING SERVER…"}
              </span>
            )}
          </div>

          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div
              className={`relative border-2 border-dashed rounded-xl p-10 flex flex-col items-center gap-5 cursor-pointer transition-all duration-200 ${
                isDragging
                  ? "border-primary bg-primary/8"
                  : "border-border hover:border-primary/40 hover:bg-card"
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className={`w-14 h-14 rounded-full border flex items-center justify-center transition-colors ${
                isDragging ? "border-primary bg-primary/10" : "border-border bg-muted"
              }`}>
                <Upload className={`w-6 h-6 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <div className="text-center">
                <p className="font-semibold text-foreground mb-1">Drop video file</p>
                <p className="text-sm text-muted-foreground">or click to browse your files</p>
              </div>
              <span
                className="text-[11px] text-muted-foreground bg-muted px-3 py-1 rounded-full tracking-widest"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                MP4 · MOV · AVI · MAX {health?.maxUploadMb ?? 200}MB
              </span>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </div>

            <div
              className="border-2 border-dashed border-border rounded-xl p-10 flex flex-col items-center gap-5 cursor-pointer transition-all duration-200 hover:border-primary/40 hover:bg-card"
              onClick={openCamera}
            >
              <div className="w-14 h-14 rounded-full border border-border bg-muted flex items-center justify-center">
                <Camera className="w-6 h-6 text-muted-foreground" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-foreground mb-1">Record live</p>
                <p className="text-sm text-muted-foreground">Use your device camera directly</p>
              </div>
              <span
                className="text-[11px] text-muted-foreground bg-muted px-3 py-1 rounded-full tracking-widest"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                AUTO-STOP {REC_LIMIT_MS / 1000}s
              </span>
            </div>
          </div>

          
          {camOpen && (
            <div className="bg-card border border-border rounded-xl p-5 mb-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className={`w-2.5 h-2.5 rounded-full ${isRecording ? "bg-[#e03c3c] animate-pulse" : "bg-muted-foreground/40"}`} />
                  <span
                    className="font-black text-sm tracking-wider text-foreground"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    {isRecording ? "RECORDING" : "CAMERA"}
                  </span>
                </div>
                <button
                  onClick={closeCamera}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                  aria-label="Close camera"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {camError && (
                <div className="flex items-start gap-2.5 border border-[#e03c3c]/40 bg-[#e03c3c]/8 rounded-lg p-3 mb-4">
                  <AlertTriangle className="w-4 h-4 text-[#e03c3c] mt-0.5 shrink-0" />
                  <p className="text-xs text-muted-foreground leading-relaxed">{camError}</p>
                </div>
              )}

              <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4">
                <video
                  ref={camPreviewRef}
                  className="w-full h-full object-contain"
                  style={{ transform: facing === "user" ? "scaleX(-1)" : undefined }}
                  playsInline
                  muted
                  autoPlay
                />
                {isRecording && (
                  <>
                    <div
                      className="absolute top-3 left-3 flex items-center gap-2 bg-black/70 px-2.5 py-1 rounded text-white text-[11px]"
                      style={{ fontFamily: "'JetBrains Mono', monospace" }}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-[#e03c3c] animate-pulse" />
                      {elapsed.toFixed(1)}s / {REC_LIMIT_MS / 1000}s
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-white/10">
                      <div
                        className="h-1 bg-[#e03c3c] transition-all duration-100"
                        style={{ width: `${Math.min(100, (elapsed * 1000 / REC_LIMIT_MS) * 100)}%` }}
                      />
                    </div>
                  </>
                )}
                {!camReady && !camError && (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                    Starting camera…
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={!camReady}
                    className="flex-1 bg-primary text-primary-foreground font-black text-base tracking-widest py-3.5 rounded-xl flex items-center justify-center gap-2.5 hover:bg-primary/90 active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none transition-all"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    <Camera className="w-5 h-5" />
                    START RECORDING
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    disabled={elapsed * 1000 < REC_MIN_MS}
                    className="flex-1 bg-[#e03c3c] text-white font-black text-base tracking-widest py-3.5 rounded-xl flex items-center justify-center gap-2.5 hover:bg-[#e03c3c]/90 active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none transition-all"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    <Square className="w-4 h-4" />
                    {elapsed * 1000 < REC_MIN_MS
                      ? `KEEP GOING… ${((REC_MIN_MS - elapsed * 1000) / 1000).toFixed(1)}s`
                      : "STOP"}
                  </button>
                )}
                <button
                  onClick={flipCamera}
                  disabled={!camReady || isRecording}
                  className="border border-border text-muted-foreground hover:text-foreground px-4 py-3.5 rounded-xl disabled:opacity-30 disabled:pointer-events-none transition-colors"
                  aria-label="Flip camera"
                  title="Flip camera"
                >
                  <SwitchCamera className="w-5 h-5" />
                </button>
              </div>

              <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                Prop the phone side-on to the ball, level with it, whole body in frame. Recording stops
                automatically at {REC_LIMIT_MS / 1000} seconds — you only need the run-up and the strike.
              </p>
              <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                <span className="text-foreground font-semibold">For a clean ball read:</span> keep the whole
                ball in frame and unobstructed at your plant and contact moments, film with even light (no
                strong backlight or deep shadow over it), and stand close enough that the ball reads as
                roughly a fifth to a full torso-width across — a speck in a wide shot won't register.
                None of this is required — the score still runs on body position alone if the ball
                isn't found — but a clean read adds the plant-foot-vs-ball metric back in.
              </p>
            </div>
          )}

          
          <div className="bg-card border border-border rounded-xl p-5 mb-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="font-semibold text-foreground text-sm mb-1">Which foot do you shoot with?</p>
                <p className="text-xs text-muted-foreground max-w-md leading-relaxed">
                  Telling us beats guessing. From a side-on camera the near and far leg overlap
                  constantly, and picking the wrong one mirrors every result.
                </p>
              </div>
              <div className="flex gap-2">
                {([["auto", "Auto-detect"], ["right", "Right"], ["left", "Left"]] as const).map(([val, label]) => (
                  <button
                    key={val}
                    onClick={() => setFootedness(val)}
                    className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                      footedness === val
                        ? "border-primary text-primary bg-primary/10"
                        : "border-border text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-36">
            <div className="flex items-start gap-2.5 border border-border rounded-xl p-4">
              <Info className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-xs text-muted-foreground leading-relaxed">
                <span className="text-foreground font-semibold">Film side-on.</span> Stand the camera
                level with the ball, square to the line of the shot, with the whole body in frame.
                Plant placement, backswing reach and torso lean are all read across the image, so a
                face-on angle flattens them and they will not be scored.
              </p>
            </div>
            <div className="flex items-start gap-2.5 border border-border rounded-xl p-4">
              <AlertCircle className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
              <p className="text-xs text-muted-foreground leading-relaxed">
                <span className="text-foreground font-semibold">Your video is uploaded</span> to the
                analysis server when you press Run Pose Analysis. The server reads the frames it needs
                and deletes the file immediately after — nothing is retained.
              </p>
            </div>
          </div>
        </div>
      )}

      
      {step === "select" && (
        <div className="max-w-3xl mx-auto px-6 py-8">
          
          {videoUrl && (
            <video
              ref={thumbVideoRef}
              src={videoUrl}
              
              
              style={{ position: "fixed", left: -9999, top: 0, width: 2, height: 2, opacity: 0, pointerEvents: "none" }}
              aria-hidden="true"
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={handleThumbVideoLoaded}
            />
          )}

          <div className="mb-6">
            <h2
              className="text-2xl font-black text-foreground tracking-wide"
              style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
            >
              MARK YOUR SHOT PHASES
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Scroll the reel to slide the {activeWindow.toFixed(2)}s window over your{" "}
              {SEGMENT_CONFIG[selectPage].label.toLowerCase()} moment. Press play and the reel follows.
            </p>
            {thumbsLoading && (
              <p
                className="text-[11px] text-primary mt-1.5 flex items-center gap-1.5"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                Generating preview frames…
              </p>
            )}
          </div>

          {error && (
            <div className="flex items-start gap-2.5 border border-[#e03c3c]/40 bg-[#e03c3c]/8 rounded-xl p-4 mb-5">
              <AlertTriangle className="w-4 h-4 text-[#e03c3c] mt-0.5 shrink-0" />
              <div>
                <p className="text-sm text-foreground font-semibold mb-0.5">Analysis failed</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{error}</p>
              </div>
            </div>
          )}

          
          <div className="flex items-center justify-center gap-2 mb-5">
            {SEGMENT_CONFIG.map((seg, i) => (
              <button
                key={seg.key}
                onClick={() => setSelectPage(i)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all"
                style={{
                  borderColor: i === selectPage ? seg.color : "var(--border)",
                  backgroundColor: i === selectPage ? `${seg.color}1a` : "transparent",
                }}
              >
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: i <= selectPage ? seg.color : "var(--muted-foreground)" }}
                />
                <span
                  className="text-[11px] font-black tracking-widest"
                  style={{
                    fontFamily: "'Barlow Condensed', sans-serif",
                    color: i === selectPage ? seg.color : "var(--muted-foreground)",
                  }}
                >
                  {i + 1}. {seg.shortLabel}
                </span>
              </button>
            ))}
          </div>

          
          <div className="bg-card border border-border rounded-xl overflow-hidden mb-5">
            <div className="relative aspect-video bg-black flex items-center justify-center">
              {videoUrl ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-full object-contain"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onEnded={() => setIsPlaying(false)}
                />
              ) : (
                <div className="text-muted-foreground text-sm">No video loaded</div>
              )}
              <button
                onClick={togglePlay}
                className="absolute inset-0 flex items-center justify-center bg-black/0 hover:bg-black/25 transition-colors group"
              >
                <div className="w-14 h-14 rounded-full bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  {isPlaying
                    ? <Pause className="w-6 h-6 text-white" />
                    : <Play className="w-6 h-6 text-white ml-0.5" />
                  }
                </div>
              </button>
              <div
                className="absolute bottom-3 left-3 text-[11px] bg-black/70 px-2 py-1 rounded text-white"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                {fmt(currentTime)} / {fmt(duration)}
              </div>
            </div>
          </div>

          
          {(() => {
            const seg = SEGMENT_CONFIG[selectPage];
            return (
              <div
                className="bg-card border rounded-xl p-5 mb-5"
                style={{ borderColor: `${seg.color}55` }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: seg.color }} />
                    <span
                      className="font-black text-sm tracking-wider text-foreground"
                      style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                    >
                      {seg.label.toUpperCase()}
                    </span>
                  </div>
                  <div
                    className="text-[11px] text-muted-foreground"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {fmt(segments[seg.key])} <span className="text-foreground/30">→</span> {fmt(segments[seg.key] + windows[seg.key])}
                    <span className="ml-2 text-muted-foreground/50">· {windows[seg.key].toFixed(2)}s</span>
                  </div>
                </div>

                <ReelScrubber
                  thumbnails={thumbnails}
                  duration={duration}
                  pxPerSecond={PX_PER_SECOND}
                  value={segments[seg.key]}
                  color={seg.color}
                  windowSeconds={windows[seg.key]}
                  markers={reelMarkers}
                  playheadTime={isPlaying ? currentTime : null}
                  containerWidth={reelWidth}
                  onMeasure={setReelWidth}
                  onChange={(t) => {
                    setSegments(prev => ({ ...prev, [seg.key]: t }));
                    setTouched(prev => (prev[seg.key] ? prev : { ...prev, [seg.key]: true }));
                    seekTo(t);
                  }}
                />

                
                <div className="flex items-center gap-3 mt-4">
                  <span
                    className="text-[10px] font-black tracking-widest text-muted-foreground shrink-0"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    WINDOW
                  </span>
                  <input
                    type="range"
                    min={WINDOW_MIN}
                    max={WINDOW_MAX}
                    step={WINDOW_STEP}
                    value={windows[seg.key]}
                    aria-label="Capture window length"
                    onChange={e => {
                      const w = Number(e.target.value);
                      setWindows(prev => ({ ...prev, [seg.key]: w }));
                      
                      setSegments(prev => ({
                        ...prev,
                        [seg.key]: Math.min(prev[seg.key], Math.max(0, duration - w)),
                      }));
                      setTouched(prev => ({ ...prev, [seg.key]: true }));
                    }}
                    className="flex-1 accent-primary"
                    style={{ accentColor: seg.color }}
                  />
                  <span
                    className="text-[11px] text-muted-foreground shrink-0 tabular-nums"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {windows[seg.key].toFixed(2)}s · ~{Math.max(1, Math.round(windows[seg.key] * sourceFps))} frames
                  </span>
                </div>

                <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                  {PHASE_HINT[seg.key]}
                </p>
              </div>
            );
          })()}

          
          <div className="flex items-start gap-2 px-1 mb-6">
            <Info className="w-3.5 h-3.5 text-muted-foreground mt-0.5 shrink-0" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              Best results come from a side-angle shot at 45–90° to the ball's path. Scroll or drag the
              reel like a video filmstrip, or press play and let it follow. Phases you have already set
              stay marked on the reel in their own colour. Mark them in order: plant, then contact,
              then follow-through.
            </p>
          </div>

          
          <div className="flex items-center justify-between gap-3">
            <button
              onClick={() => setSelectPage(p => Math.max(0, p - 1))}
              disabled={selectPage === 0}
              className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:pointer-events-none transition-colors border border-border px-4 py-3 rounded-xl"
            >
              <ChevronLeft className="w-4 h-4" />
              Back
            </button>

            {selectPage < SEGMENT_CONFIG.length - 1 ? (
              <button
                onClick={() => {
                  
                  
                  setTouched(prev => ({ ...prev, [activeKey]: true }));
                  setSelectPage(p => Math.min(SEGMENT_CONFIG.length - 1, p + 1));
                }}
                className="flex-1 bg-primary text-primary-foreground font-black text-base tracking-widest py-4 rounded-xl flex items-center justify-center gap-2.5 hover:bg-primary/90 active:scale-[0.98] transition-all"
                style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                NEXT: {SEGMENT_CONFIG[selectPage + 1].label.toUpperCase()}
                <ChevronRight className="w-5 h-5" />
              </button>
            ) : (
              <button
                onClick={startAnalysis}
                disabled={!videoFile}
                className="flex-1 bg-primary text-primary-foreground font-black text-base tracking-widest py-4 rounded-xl flex items-center justify-center gap-2.5 hover:bg-primary/90 active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none transition-all"
                style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                <Zap className="w-5 h-5" />
                RUN POSE ANALYSIS
              </button>
            )}
          </div>

          <p className="text-xs text-center text-muted-foreground mt-4">
            Your video is uploaded to the analysis server, then deleted once its frames have been read.
            {health?.aiFeedback && " The three measured frames are also sent to Google's Gemini API for AI coaching notes."}
          </p>
        </div>
      )}

      
      {step === "analyzing" && (
        <div className="min-h-[85vh] flex flex-col items-center justify-center px-6">
          <div className="text-center max-w-sm w-full">
            
            <div className="relative w-24 h-24 mx-auto mb-10">
              <div className="absolute inset-0 rounded-full border border-primary/15 animate-ping" style={{ animationDuration: "2s" }} />
              <div className="absolute inset-3 rounded-full border border-primary/25 animate-ping" style={{ animationDuration: "2s", animationDelay: "0.4s" }} />
              <div className="absolute inset-6 rounded-full border border-primary/40 animate-ping" style={{ animationDuration: "2s", animationDelay: "0.8s" }} />
              <div className="absolute inset-0 flex items-center justify-center">
                <Zap className="w-8 h-8 text-primary" />
              </div>
            </div>

            <h2
              className="text-4xl font-black text-foreground tracking-wide mb-2"
              style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
            >
              ANALYZING
            </h2>
            <p className="text-muted-foreground text-sm mb-8">
              {stage === "upload"
                ? `Uploading video… ${Math.round(uploadPct * 100)}%`
                : "Decoding, detecting pose and tracking the ball on the server…"}
            </p>

            
            <div className="w-full bg-muted rounded-full h-1.5 mb-2 overflow-hidden">
              {stage === "upload" ? (
                <div
                  className="bg-primary h-1.5 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${Math.round(uploadPct * 100)}%` }}
                />
              ) : (
                <div className="h-1.5 w-1/3 rounded-full bg-primary animate-pulse" />
              )}
            </div>
            <div
              className="text-right text-xs text-muted-foreground mb-8"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              {stage === "upload" ? `${Math.round(uploadPct * 100)}%` : "working…"}
            </div>

            
            <div className="space-y-2.5 text-left">
              {[
                { key: "upload", label: "Upload video to analysis server" },
                { key: "analyze", label: "Decode · pose · ball · scoring" },
              ].map(({ key, label }) => {
                const done = key === "upload" && stage === "analyze";
                const active = key === stage;
                return (
                  <div
                    key={label}
                    className={`flex items-center gap-3 text-sm transition-all duration-500 ${
                      done || active ? "text-foreground" : "text-muted-foreground/25"
                    }`}
                  >
                    <div className={`w-1.5 h-1.5 rounded-full shrink-0 transition-colors ${
                      done ? "bg-primary" : active ? "bg-primary animate-pulse" : "bg-muted-foreground/20"
                    }`} />
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "11px" }}>
                      {done && "✓ "}{label}
                    </span>
                  </div>
                );
              })}
            </div>

            <p className="text-[11px] text-muted-foreground/60 mt-8 leading-relaxed">
              The heavy pose model runs three phases in parallel. A cold worker takes longer than a warm one.
            </p>
          </div>
        </div>
      )}

      
      {step === "results" && results && (
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
          
          <div className="flex items-start justify-between">
            <div>
              <h2
                className="text-3xl font-black text-foreground tracking-wide"
                style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                SHOT ANALYSIS REPORT
              </h2>
            </div>
            <button
              onClick={reset}
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors border border-border px-4 py-2 rounded-lg hover:bg-card"
            >
              <RotateCcw className="w-4 h-4" />
              New clip
            </button>
          </div>

          
          {results.warnings.length > 0 && (
            <div className="border border-[#ffb800]/40 bg-[#ffb800]/8 rounded-xl p-5">
              <div className="flex items-center gap-2.5 mb-3">
                <AlertTriangle className="w-4 h-4 text-[#ffb800]" />
                <h3
                  className="text-sm font-black tracking-widest text-foreground"
                  style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                >
                  READ THIS FIRST
                </h3>
              </div>
              <ul className="space-y-2.5">
                {results.warnings.map((w, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#ffb800] mt-2 shrink-0" />
                    <span className="text-sm text-muted-foreground leading-relaxed">{w}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <div className="bg-card border border-border rounded-xl p-6 flex flex-col items-center justify-center gap-3">
              <div className="relative w-28 h-28">
                <ScoreRing score={results.overall ?? 0} size={112} color={scoreColor(results.overall)} />
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span
                    className="text-foreground font-black leading-none"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: "2.2rem" }}
                  >
                    {results.overall ?? "–"}
                  </span>
                  <span className="text-muted-foreground text-xs" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                    {results.overall === null ? "not scored" : "/100"}
                  </span>
                </div>
              </div>
              <span
                className="text-xs font-bold tracking-widest text-muted-foreground"
                style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                OVERALL SCORE
              </span>
            </div>

            
            {results.phases.map((phase) => (
              <div
                key={phase.key}
                className={`bg-card border rounded-xl p-5 cursor-pointer transition-all duration-150 ${
                  expandedPhase === phase.key ? "border-primary/30" : "border-border hover:border-border/60"
                }`}
                onClick={() => setExpandedPhase(expandedPhase === phase.key ? null : phase.key)}
              >
                <div className="flex items-center justify-between mb-4">
                  <span
                    className="text-xs font-black tracking-wider text-foreground"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    {phase.label.toUpperCase()}
                  </span>
                  <span
                    className="font-black text-xl"
                    style={{ color: scoreColor(phase.score), fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    {phase.score ?? "–"}
                  </span>
                </div>

                <div className="w-full bg-muted rounded-full h-1 mb-4">
                  <div
                    className="h-1 rounded-full"
                    style={{ width: `${phase.score ?? 0}%`, backgroundColor: scoreColor(phase.score) }}
                  />
                </div>

                {phase.insufficient && (
                  <p className="text-[11px] text-muted-foreground leading-relaxed mb-3">
                    Too few metrics could be measured here to publish a score.
                  </p>
                )}

                <div className="space-y-2">
                  {phase.metrics.map((m) => (
                    <div key={m.id} className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground truncate pr-2">{m.name}</span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <div className={`w-1.5 h-1.5 rounded-full ${getStatusDot(m.status)}`} />
                        <span
                          className="text-xs font-medium text-foreground"
                          style={{ fontFamily: "'JetBrains Mono', monospace" }}
                        >
                          {m.value ?? "—"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {expandedPhase === phase.key && (
                  <div className="mt-3 pt-3 border-t border-border space-y-2.5">
                    {phase.metrics.map((m) => (
                      <div key={m.id}>
                        <p className="text-[11px] text-foreground font-semibold">{m.name}</p>
                        <p className="text-[11px] text-muted-foreground leading-relaxed">
                          Ideal {m.ideal}
                          {m.value ? ` · yours ${m.value}` : ""}
                          {m.status === "unknown" && m.reason ? ` · ${m.reason}` : ""}
                        </p>
                        {m.caveat && (
                          <p className="text-[10px] text-muted-foreground/70 leading-relaxed mt-0.5">{m.caveat}</p>
                        )}
                      </div>
                    ))}
                    <p className="text-[10px] text-muted-foreground/60 pt-1">
                      {phase.counted} metric{phase.counted === 1 ? "" : "s"} scored, {phase.skipped} not measurable.
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>

          
          <div className="bg-card border border-border rounded-xl p-6">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
              <h3
                className="text-sm font-black tracking-widest text-foreground"
                style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                MEASURED FRAMES
              </h3>
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                <span>JOINT ANGLES:</span>
                {([["#00df54", "80+"], ["#ffb800", "60–79"], ["#e03c3c", "under 60"]] as const).map(([c, t]) => (
                  <span key={t} className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: c }} />
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {results.phases.map((p) => (
                <div key={p.key} className="border border-border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span
                      className="text-xs font-black tracking-wider"
                      style={{ fontFamily: "'Barlow Condensed', sans-serif", color: p.color }}
                    >
                      {p.label.toUpperCase()}
                    </span>
                    <span
                      className="text-sm font-bold"
                      style={{ color: scoreColor(p.score), fontFamily: "'JetBrains Mono', monospace" }}
                    >
                      {p.score === null ? "not scored" : `${p.score}/100`}
                    </span>
                  </div>

                  <div className="rounded-md overflow-hidden bg-black mb-3">
                    {p.image ? (
                      <img src={p.image} alt={`${p.label} frame with detected pose`} className="w-full object-contain" />
                    ) : (
                      <div className="h-40 flex items-center justify-center text-xs text-muted-foreground">
                        No frame returned
                      </div>
                    )}
                  </div>

                  <div
                    className="text-[10px] text-muted-foreground space-y-0.5"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {p.time !== null && <div>t = {p.time.toFixed(3)}s</div>}
                    {p.shiftedMs !== 0 && (
                      <div className="text-[#ffb800]">
                        nearest clean frame used ({p.shiftedMs > 0 ? "+" : ""}{p.shiftedMs}ms)
                      </div>
                    )}
                    <div className={p.ballFound ? "" : "text-muted-foreground/60"}>
                      {p.ballFound ? "ball tracked" : "no ball tracked"}
                    </div>
                    {p.reposed && (
                      <div className="text-[#ffb800]">skeleton re-detected (backup model)</div>
                    )}
                  </div>

                  {p.aiNote && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <div className="flex items-center gap-1.5 mb-1.5">
                        <Sparkles className="w-3 h-3 text-primary" />
                        <span
                          className="text-[10px] font-bold tracking-widest text-muted-foreground"
                          style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                        >
                          AI COACH
                        </span>
                      </div>
                      <p className="text-xs text-foreground leading-relaxed">{p.aiNote}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {results.feedback.map((f) => (
              <div key={f.key} className="bg-card border border-border rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: f.color }} />
                  <h3
                    className="text-xs font-black tracking-widest text-foreground"
                    style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
                  >
                    {f.label.toUpperCase()}
                  </h3>
                </div>

                {f.strengths.length === 0 && f.improvements.length === 0 ? (
                  <p className="text-xs text-muted-foreground leading-relaxed">Nothing scorable here.</p>
                ) : (
                  <div className="space-y-3">
                    {f.strengths.length > 0 && (
                      <ul className="space-y-2">
                        {f.strengths.map((s, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <CheckCircle2 className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
                            <span className="text-xs text-foreground leading-snug">{s}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                    {f.improvements.length > 0 && (
                      <ul className="space-y-2">
                        {f.improvements.map((s, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <AlertCircle className="w-3.5 h-3.5 text-accent mt-0.5 shrink-0" />
                            <span className="text-xs text-foreground leading-snug">{s}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          
          <details className="bg-card border border-border rounded-xl p-6">
            <summary className="text-sm font-black tracking-widest text-foreground cursor-pointer"
              style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
              WHAT THIS SYSTEM CAN AND CANNOT MEASURE
            </summary>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-5">
              <div>
                <p className="text-xs font-bold text-foreground mb-2">Reasonably detectable</p>
                <ul className="space-y-1.5 text-xs text-muted-foreground leading-relaxed">
                  <li>Joint angles (knee, hip, ankle)</li>
                  <li>Torso lean relative to vertical</li>
                  <li>Limb positions relative to the hips, normalized by torso length</li>
                  <li>Whether body mass is stacked over the support foot</li>
                  <li>Ball position, and the plant foot's fore/aft position relative to it</li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-bold text-foreground mb-2">Unreliable / not measured</p>
                <ul className="space-y-1.5 text-xs text-muted-foreground leading-relaxed">
                  <li>The exact point on the boot that met the ball</li>
                  <li>Ball speed, spin, or where the shot went</li>
                  <li>True 3D hip rotation from one camera — only a rough 2D proxy</li>
                  <li>Lateral plant-foot distance from the ball</li>
                  <li>Anything depending on depth toward or away from the camera</li>
                </ul>
              </div>
            </div>
            <p className="text-[11px] text-muted-foreground/70 mt-4">
              Metrics shown with a grey dot could not be measured from this footage and are excluded
              from every score above.
            </p>
          </details>

          
          <div className="flex items-center justify-between pt-2 pb-8">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <TrendingUp className="w-3.5 h-3.5" />
              Camera read as {results.context?.view?.label ?? "unknown"} · kicking leg:{" "}
              {results.context?.kickSide ?? "unknown"}
            </div>
            <button
              onClick={reset}
              className="flex items-center gap-2 bg-primary text-primary-foreground font-black text-sm tracking-widest px-6 py-3 rounded-xl hover:bg-primary/90 active:scale-[0.98] transition-all"
              style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
            >
              <RotateCcw className="w-4 h-4" />
              ANALYZE ANOTHER SHOT
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
