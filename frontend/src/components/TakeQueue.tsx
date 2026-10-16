import React from "react";
import { Trash2, Mic, FileAudio, ArrowRight, Loader2 } from "lucide-react";
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
  if (takes.length === 0) return null;

  return (
    <div className="mt-5 pt-5 border-t border-slate-200/80">
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
            Queued Assessment Takes
          </span>
          <span className="text-xs font-medium bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
            {takes.length} {takes.length === 1 ? "take" : "takes"}
          </span>
        </div>

        <button
          onClick={onClearAll}
          disabled={isAnalyzing}
          className="text-xs text-slate-400 hover:text-rose-600 transition-colors disabled:opacity-50"
        >
          Clear All
        </button>
      </div>

      <div className="space-y-2 mb-4">
        {takes.map((take, idx) => (
          <div
            key={take.id}
            className="flex items-center justify-between p-2.5 rounded-lg bg-slate-50 border border-slate-200/70 text-xs"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-6 h-6 rounded-md bg-white border border-slate-200 flex items-center justify-center text-slate-500 shrink-0">
                {take.source === "mic" ? <Mic className="w-3.5 h-3.5 text-blue-600" /> : <FileAudio className="w-3.5 h-3.5 text-slate-600" />}
              </div>
              <div className="min-w-0 truncate">
                <span className="font-semibold text-slate-700 mr-2">Take {idx + 1}:</span>
                <span className="text-slate-900 truncate">{take.name}</span>
                <span className="text-slate-400 ml-2">({(take.size / 1024).toFixed(1)} KB)</span>
              </div>
            </div>

            <button
              onClick={() => onRemoveTake(take.id)}
              disabled={isAnalyzing}
              className="p-1 text-slate-400 hover:text-rose-600 rounded-md hover:bg-white transition-colors"
              title="Remove take"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>

      <button
        onClick={onRunAssessment}
        disabled={isAnalyzing || takes.length === 0}
        className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-50 text-white font-medium text-sm rounded-xl flex items-center justify-center gap-2 shadow-xs transition-all hover:shadow-sm"
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Evaluating Acoustic Biomarkers (Wav2Vec2)...</span>
          </>
        ) : (
          <>
            <span>Analyze {takes.length} {takes.length === 1 ? "Recording" : "Recordings"}</span>
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </button>
    </div>
  );
};
