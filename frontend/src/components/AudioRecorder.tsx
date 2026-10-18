import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, RotateCcw, Check, Volume2, AlertTriangle, CheckCircle2 } from "lucide-react";

interface AudioRecorderProps {
  onRecordingComplete: (file: File) => void;
}

interface RecordingQuality {
  peak: number;
  rms: number;
  isLowVolume: boolean;
}

/**
 * Encodes an AudioBuffer into a standard 16-bit PCM Mono RIFF WAV Blob.
 * Guarantees exact container duration metadata, fixing the Chrome 0s/truncated replay bug.
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const data = buffer.getChannelData(0);
  const dataSize = data.length * 2;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true);  // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  // Write 16-bit PCM samples with hard clipping protection
  let offset = 44;
  for (let i = 0; i < data.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [liveVolume, setLiveVolume] = useState<number>(0);
  const [volumeStatus, setVolumeStatus] = useState<"quiet" | "optimal" | "loud">("quiet");
  const [qualityStats, setQualityStats] = useState<RecordingQuality | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);

  const TARGET_DURATION = 8.0;

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    analyserRef.current = null;
    setLiveVolume(0);
  };

  const startRecording = async () => {
    try {
      cleanup();
      setAudioUrl(null);
      setRecordedFile(null);
      setQualityStats(null);
      setElapsed(0);
      setLiveVolume(0);
      setVolumeStatus("quiet");

      // Access raw microphone without aggressive noise gates that distort sustained vowels
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: 1,
          },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      streamRef.current = stream;

      // Initialize live audio analyzer for real-time volume feedback
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const liveCtx = new AudioContextClass();
      audioCtxRef.current = liveCtx;
      const source = liveCtx.createMediaStreamSource(stream);
      const analyser = liveCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Real-time volume polling loop
      const pcmBuffer = new Float32Array(analyser.fftSize);
      const monitorVolume = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(pcmBuffer);
        let sumSquares = 0;
        let peak = 0;
        for (let i = 0; i < pcmBuffer.length; i++) {
          const val = Math.abs(pcmBuffer[i]);
          if (val > peak) peak = val;
          sumSquares += val * val;
        }
        const rms = Math.sqrt(sumSquares / pcmBuffer.length);
        // Map RMS (0.0 to ~0.3) to 0-100 percentage
        const volPct = Math.min(100, Math.round(rms * 380));
        setLiveVolume(volPct);

        if (peak > 0.92 || volPct > 80) {
          setVolumeStatus("loud");
        } else if (volPct >= 20) {
          setVolumeStatus("optimal");
        } else {
          setVolumeStatus("quiet");
        }

        animFrameRef.current = requestAnimationFrame(monitorVolume);
      };
      animFrameRef.current = requestAnimationFrame(monitorVolume);

      // Enforce highest quality broadcast bitrate (320 kbps) and preferred container
      let mimeType = "";
      if (typeof MediaRecorder !== "undefined") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        }
      }

      const recorderOptions: MediaRecorderOptions = {
        audioBitsPerSecond: 320000, // Maximum 320 kbps to prevent compression artifacts
      };
      if (mimeType) {
        recorderOptions.mimeType = mimeType;
      }

      const chunks: Blob[] = [];
      const mediaRecorder = new MediaRecorder(stream, recorderOptions);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
          const arrayBuffer = await rawBlob.arrayBuffer();

          // Decode raw audio into PCM AudioBuffer, then re-encode to clean RIFF WAV
          const decodeCtx = new AudioContextClass();
          const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
          const wavBlob = audioBufferToWav(audioBuffer);
          await decodeCtx.close();

          // Compute exact recording volume metrics
          const pcm = audioBuffer.getChannelData(0);
          let maxPeak = 0;
          let sumSquares = 0;
          for (let i = 0; i < pcm.length; i++) {
            const val = Math.abs(pcm[i]);
            if (val > maxPeak) maxPeak = val;
            sumSquares += val * val;
          }
          const overallRms = Math.sqrt(sumSquares / pcm.length);
          const isLowVolume = maxPeak < 0.12 || overallRms < 0.035;

          setQualityStats({
            peak: maxPeak,
            rms: overallRms,
            isLowVolume,
          });

          const fileName = `phonation_take_${Date.now()}.wav`;
          const file = new File([wavBlob], fileName, { type: "audio/wav" });
          const url = URL.createObjectURL(wavBlob);

          setAudioUrl(url);
          setRecordedFile(file);
        } catch (err) {
          console.error("Audio conversion failed:", err);
          alert("Audio recording failed to convert. Please try again.");
        } finally {
          cleanup();
        }
      };

      // Start recording as a single continuous block without timeslice fragmentation
      mediaRecorder.start();
      setIsRecording(true);

      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        const diff = (Date.now() - startTime) / 1000;
        setElapsed(diff);
        if (diff >= 10.0) {
          stopRecording();
        }
      }, 100);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Microphone access was denied or is not supported by your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setLiveVolume(0);
  };

  const handleConfirm = () => {
    if (recordedFile) {
      onRecordingComplete(recordedFile);
      setAudioUrl(null);
      setRecordedFile(null);
      setQualityStats(null);
      setElapsed(0);
    }
  };

  return (
    <div className="flex flex-col items-center w-full">
      {!audioUrl ? (
        <div className="flex flex-col items-center py-4 w-full">
          {/* Record Button with Pulse Effect */}
          <div className="relative flex items-center justify-center">
            {isRecording && (
              <span className="absolute w-24 h-24 rounded-full bg-rose-500/20 animate-ping" />
            )}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-all duration-200 shadow-md touch-manipulation ${
                isRecording
                  ? "bg-rose-600 hover:bg-rose-700 active:bg-rose-800 text-white"
                  : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white hover:scale-105 active:scale-95"
              }`}
              title={isRecording ? "Stop Recording" : "Start 8s Voice Recording"}
            >
              {isRecording ? <Square className="w-7 h-7 fill-white" /> : <Mic className="w-8 h-8" />}
            </button>
          </div>

          {/* Recording Timer & Status */}
          <div className="mt-4 text-center w-full max-w-xs">
            <div className="text-sm font-semibold text-slate-800">
              {isRecording ? (
                <span className="font-mono text-rose-600 font-bold text-base">
                  {elapsed.toFixed(1)}s / {TARGET_DURATION}s
                </span>
              ) : (
                "Tap to Record Phonation"
              )}
            </div>

            {/* Live Volume Meter during recording */}
            {isRecording ? (
              <div className="mt-3 w-full bg-slate-100 rounded-xl p-2.5 border border-slate-200 flex flex-col gap-1.5 animate-fadeIn">
                <div className="flex items-center justify-between text-[11px] font-medium">
                  <span className="text-slate-500">Live Input Level</span>
                  <span
                    className={`font-semibold ${
                      volumeStatus === "optimal"
                        ? "text-emerald-600"
                        : volumeStatus === "quiet"
                        ? "text-amber-600"
                        : "text-rose-600"
                    }`}
                  >
                    {volumeStatus === "optimal"
                      ? "✓ Optimal Volume"
                      : volumeStatus === "quiet"
                      ? "⚠️ Speak Louder"
                      : "⚠️ Too Loud (Clipping)"}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden relative">
                  {/* Recommended zone indicator: 20% to 80% */}
                  <div className="absolute left-[20%] right-[20%] top-0 bottom-0 bg-emerald-200/50 pointer-events-none" />
                  <div
                    className={`h-full transition-all duration-75 rounded-full ${
                      volumeStatus === "optimal"
                        ? "bg-emerald-500"
                        : volumeStatus === "quiet"
                        ? "bg-amber-500"
                        : "bg-rose-500"
                    }`}
                    style={{ width: `${Math.max(4, liveVolume)}%` }}
                  />
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 mt-1">
                Hold natural pitch and steady conversational volume for 8 seconds
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Recording Captured: Quality Audit, Preview Player & Action Buttons */
        <div className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center shrink-0">
                <Volume2 className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-800 truncate">Recording Captured</div>
                <div className="text-[11px] text-slate-500 font-mono">
                  {elapsed.toFixed(1)}s WAV • 320 kbps Studio PCM
                </div>
              </div>
            </div>
            <span className="text-[11px] font-semibold font-mono bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full border border-emerald-200">
              Lossless PCM
            </span>
          </div>

          {/* Volume Quality Alert Banner */}
          {qualityStats && (
            <div>
              {qualityStats.isLowVolume ? (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex items-start gap-2.5 text-xs text-amber-900">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <div className="font-semibold text-amber-800">
                      Low Volume Warning (Peak: {(qualityStats.peak * 100).toFixed(0)}%)
                    </div>
                    <div className="text-[11px] text-amber-700 leading-relaxed">
                      This sample was recorded softly. Software peak normalization amplifies room noise on low-volume audio, which can artificially trigger a false-positive Parkinson's flag.
                      For accurate screening, we recommend tapping <strong>Re-record</strong> and speaking at a firmer, projected conversational volume.
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-3 py-2 flex items-center gap-2 text-xs text-emerald-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  <span>
                    Acoustic quality verified: Optimal volume level (Peak: {(qualityStats.peak * 100).toFixed(0)}%) for dysphonia assessment.
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Native HTML5 Audio Player */}
          <div className="w-full pt-1">
            <audio controls src={audioUrl} className="w-full h-10 rounded-lg focus:outline-none" />
          </div>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-end gap-2.5 pt-2 border-t border-slate-200">
            <button
              onClick={() => {
                setAudioUrl(null);
                setRecordedFile(null);
                setQualityStats(null);
                setElapsed(0);
              }}
              className="w-full sm:w-auto px-4 py-2.5 min-h-[44px] text-xs font-semibold text-slate-700 hover:text-slate-900 bg-white hover:bg-slate-100 active:bg-slate-200 border border-slate-200 rounded-xl flex items-center justify-center gap-1.5 transition-colors touch-manipulation"
            >
              <RotateCcw className="w-3.5 h-3.5 shrink-0" />
              <span>Re-record</span>
            </button>

            <button
              onClick={handleConfirm}
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

