import React from "react";
import { Trash2, Mic, FileAudio, ArrowRight, Loader2, Music2 } from "lucide-react";
import { AudioTake } from "../types/api";

interface TakeQueueProps {
  takes: AudioTake[];
  onRemoveTake: (id: string) => void;
  onClearAll: () => void;
  onRunAssessment: () => void;
  isAnalyzing: boolean;
}

export const TakeQueue: React.FC<TakeQueueProps> = ({
  takes,
  onRemoveTake,
  onClearAll,
  onRunAssessment,
  isAnalyzing,
}) => {
  return (
    <div className="bg-white border border-slate-200/80 rounded-2xl p-4 sm:p-5 shadow-xs flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-3 px-0.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
              Assessment Queue
            </span>
            <span
              className={`text-[11px] font-semibold px-2 py-0.5 rounded-full transition-colors ${
                takes.length > 0
                  ? "bg-blue-100 text-blue-700 border border-blue-200/60"
                  : "bg-slate-100 text-slate-500"
              }`}
            >
              {takes.length} {takes.length === 1 ? "take" : "takes"}
            </span>
          </div>

          {takes.length > 0 && (
            <button
              onClick={onClearAll}
              disabled={isAnalyzing}
              className="text-xs font-medium text-slate-400 hover:text-rose-600 active:text-rose-700 transition-colors disabled:opacity-50 touch-manipulation px-1 py-0.5"
            >
              Clear All
            </button>
          )}
        </div>

        {takes.length === 0 ? (
          <div className="p-4 rounded-xl border border-dashed border-slate-200 bg-slate-50/60 text-center my-2">
            <Music2 className="w-6 h-6 text-slate-400 mx-auto mb-1.5 opacity-60" />
            <div className="text-xs font-semibold text-slate-700">Awaiting Voice Takes</div>
            <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">
              Record 8-second sustained vowel phonation or upload audio files to queue assessments.
            </p>
          </div>
        ) : (
          <div className="space-y-1.5 mb-3 max-h-[190px] overflow-y-auto pr-0.5 scrollbar-thin">
            {takes.map((take, idx) => (
              <div
                key={take.id}
                className="flex items-center justify-between p-2 rounded-xl bg-slate-50/90 border border-slate-200/70 text-xs hover:border-slate-300 transition-all"
              >
                <div className="flex items-center gap-2.5 min-w-0 flex-1 mr-2">
                  <div className="w-6 h-6 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-slate-500 shrink-0">
                    {take.source === "mic" ? (
                      <Mic className="w-3.5 h-3.5 text-blue-600" />
                    ) : (
                      <FileAudio className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                  <div className="min-w-0 truncate">
                    <span className="font-semibold text-slate-800 mr-1.5">Take {idx + 1}:</span>
                    <span className="text-slate-700 truncate font-mono text-[11px]">{take.name}</span>
                    <span className="text-slate-400 ml-1.5 text-[10px]">
                      ({(take.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => onRemoveTake(take.id)}
                  disabled={isAnalyzing}
                  className="p-1.5 text-slate-400 hover:text-rose-600 active:text-rose-700 rounded-lg hover:bg-white active:bg-rose-50 transition-colors shrink-0 touch-manipulation"
                  title="Remove take"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <button
        onClick={onRunAssessment}
        disabled={isAnalyzing || takes.length === 0}
        className={`w-full py-3 px-4 min-h-[46px] rounded-xl flex items-center justify-center gap-2 font-semibold text-sm transition-all touch-manipulation shadow-xs ${
          takes.length > 0
            ? "bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white shadow-blue-500/10 cursor-pointer"
            : "bg-slate-100 text-slate-400 border border-slate-200/80 cursor-not-allowed"
        }`}
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-white" />
            <span>Evaluating Acoustic Biomarkers...</span>
          </>
        ) : (
          <>
            <span>
              {takes.length === 0
                ? "Analyze Voice Biomarkers"
                : `Analyze ${takes.length} ${takes.length === 1 ? "Recording" : "Recordings"}`}
            </span>
            <ArrowRight className="w-4 h-4 shrink-0" />
          </>
        )}
      </button>
    </div>
  );
};

