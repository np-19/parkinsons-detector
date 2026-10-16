import React from "react";
import { VolumeX, Mic, Clock } from "lucide-react";

export const ProtocolPills: React.FC = () => {
  return (
    <div className="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-xs">
      <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-3 px-0.5">
        <span className="font-semibold text-slate-700 text-[11px] tracking-wide uppercase">Clinical SOP Guidelines</span>
        <span className="text-[10px] text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full font-semibold border border-blue-100">mPower Protocol</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 gap-2 sm:gap-2.5">
        <div className="flex items-center gap-2.5 p-2 rounded-xl bg-slate-50/80 border border-slate-100/90">
          <div className="p-1.5 rounded-lg bg-blue-50 text-blue-600 shrink-0">
            <VolumeX className="w-3.5 h-3.5" />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-slate-800 text-xs">Quiet Room Required</div>
            <div className="text-[11px] text-slate-500 truncate">Suppress fans & room echo</div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 p-2 rounded-xl bg-slate-50/80 border border-slate-100/90">
          <div className="p-1.5 rounded-lg bg-blue-50 text-blue-600 shrink-0">
            <Mic className="w-3.5 h-3.5" />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-slate-800 text-xs">Sustained Vowel "Aaah"</div>
            <div className="text-[11px] text-slate-500 truncate">Steady pitch for 8–10 seconds</div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 p-2 rounded-xl bg-slate-50/80 border border-slate-100/90">
          <div className="p-1.5 rounded-lg bg-blue-50 text-blue-600 shrink-0">
            <Clock className="w-3.5 h-3.5" />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-slate-800 text-xs">Multi-Take Ideal (2–3 Takes)</div>
            <div className="text-[11px] text-slate-500 truncate">Captures diurnal motor variance</div>
          </div>
        </div>
      </div>
    </div>
  );
};

