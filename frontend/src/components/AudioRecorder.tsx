import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, RotateCcw, Check, Volume2 } from "lucide-react";

interface AudioRecorderProps {
  onRecordingComplete: (file: File) => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const TARGET_DURATION = 8.0; // 8.0s clinical window

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
    };
  }, []);

  const drawVisualizer = () => {
    if (!analyserRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "#F8FAFC";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#2563EB";
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };

    render();
  };

  const startRecording = async () => {
    try {
      setAudioUrl(null);
      setRecordedFile(null);
      audioChunksRef.current = [];
      setRecordingSeconds(0);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      // Initialize Web Audio API for visualizer
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioContextRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      drawVisualizer();

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/ogg;codecs=opus";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const ext = mimeType.includes("ogg") ? "ogg" : "webm";
        const file = new File([audioBlob], `phonation_take_${Date.now()}.${ext}`, { type: mimeType });
        const url = URL.createObjectURL(audioBlob);

        setAudioUrl(url);
        setRecordedFile(file);

        // Stop all audio tracks
        stream.getTracks().forEach((track) => track.stop());
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      };

      mediaRecorder.start(100);
      setIsRecording(true);

      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        setRecordingSeconds(elapsed);
        if (elapsed >= 12.0) {
          stopRecording();
        }
      }, 100);
    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Microphone access was denied or is not supported by your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const handleConfirmTake = () => {
    if (recordedFile) {
      onRecordingComplete(recordedFile);
      setAudioUrl(null);
      setRecordedFile(null);
      setRecordingSeconds(0);
    }
  };

  const progressPercent = Math.min((recordingSeconds / TARGET_DURATION) * 100, 100);

  return (
    <div className="flex flex-col items-center w-full">
      {/* Visualizer Canvas during recording */}
      <div
        className={`w-full h-20 bg-slate-50 border border-slate-200/80 rounded-xl mb-4 overflow-hidden relative transition-all ${
          isRecording ? "block" : "hidden"
        }`}
      >
        <canvas ref={canvasRef} width={600} height={80} className="w-full h-full" />
        <div className="absolute top-2 right-3 text-xs font-mono font-medium text-blue-700 bg-blue-50/90 px-2.5 py-1 rounded-md border border-blue-200/50 shadow-xs">
          {recordingSeconds.toFixed(1)}s / {TARGET_DURATION}s
        </div>
      </div>

      {/* Primary Interaction State */}
      {!audioUrl ? (
        <div className="flex flex-col items-center py-2 sm:py-3 w-full">
          {/* Circular Button with Progress Ring */}
          <div className="relative flex items-center justify-center">
            {isRecording && (
              <svg className="absolute w-24 h-24 -rotate-90">
                <circle
                  cx="48"
                  cy="48"
                  r="44"
                  stroke="#E2E8F0"
                  strokeWidth="4"
                  fill="none"
                />
                <circle
                  cx="48"
                  cy="48"
                  r="44"
                  stroke="#2563EB"
                  strokeWidth="4"
                  fill="none"
                  strokeDasharray="276"
                  strokeDashoffset={276 - (276 * progressPercent) / 100}
                  className="transition-all duration-100 ease-linear"
                />
              </svg>
            )}

            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-200 shadow-md touch-manipulation ${
                isRecording
                  ? "bg-rose-600 hover:bg-rose-700 active:bg-rose-800 text-white animate-pulse"
                  : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white hover:scale-105 active:scale-95"
              }`}
              title={isRecording ? "Stop Recording" : "Start 8s Voice Recording"}
            >
              {isRecording ? <Square className="w-7 h-7 fill-white" /> : <Mic className="w-8 h-8" />}
            </button>
          </div>

          <div className="mt-4 text-center px-2">
            <div className="text-sm font-semibold text-slate-800">
              {isRecording ? "Sustain steady 'Aaah'..." : "Tap to Record Phonation"}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              {isRecording ? "Automatic stop after 8–10 seconds" : "Hold at natural pitch and steady loudness"}
            </div>
          </div>
        </div>
      ) : (
        /* Recording Complete Preview & Confirmation */
        <div className="w-full bg-slate-50/80 border border-slate-200/90 rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center shrink-0">
                <Volume2 className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-800 truncate">Recording Captured</div>
                <div className="text-[11px] text-slate-500">{recordingSeconds.toFixed(1)}s phonation window</div>
              </div>
            </div>

            <span className="text-[11px] font-semibold font-mono bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full border border-blue-200/50">
              Valid Take
            </span>
          </div>

          {/* Full-width responsive audio player */}
          <div className="w-full pt-1">
            <audio controls src={audioUrl} className="w-full h-10 rounded-lg focus:outline-none" />
          </div>

          {/* Touch-friendly Action Buttons: stack on mobile, row on tablet/desktop */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-end gap-2.5 pt-2 border-t border-slate-200/70">
            <button
              onClick={() => {
                setAudioUrl(null);
                setRecordedFile(null);
              }}
              className="w-full sm:w-auto px-4 py-2.5 min-h-[44px] text-xs font-semibold text-slate-700 hover:text-slate-900 bg-white hover:bg-slate-100 active:bg-slate-200 border border-slate-200 rounded-xl flex items-center justify-center gap-1.5 transition-colors touch-manipulation"
            >
              <RotateCcw className="w-3.5 h-3.5 shrink-0" />
              <span>Re-record</span>
            </button>

            <button
              onClick={handleConfirmTake}
              className="w-full sm:w-auto px-5 py-2.5 min-h-[44px] text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-xl flex items-center justify-center gap-1.5 shadow-xs transition-colors touch-manipulation"
            >
              <Check className="w-4 h-4 shrink-0" />
              <span>Add to Assessment Queue</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

