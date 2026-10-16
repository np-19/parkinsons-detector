import React from "react";
import { VolumeX, Mic, Clock } from "lucide-react";

export const ProtocolPills: React.FC = () => {
  return (
    <div className="bg-white border border-slate-200/80 rounded-xl p-3.5 shadow-xs mb-6">
      <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-2 px-1">
        <span>STANDARD OPERATING PROCEDURE</span>
        <span className="text-[11px] text-blue-600 bg-blue-50 px-2 py-0.5 rounded-md font-semibold">mPower Protocol</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        <div className="flex items-center gap-2.5 p-2 rounded-lg bg-slate-50 border border-slate-100">
          <div className="p-1.5 rounded-md bg-blue-50 text-blue-600">
            <VolumeX className="w-4 h-4" />
          </div>
          <div>
            <div className="font-medium text-slate-800 text-xs">Quiet Room</div>
            <div className="text-[11px] text-slate-500">Fans & noise turned off</div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 p-2 rounded-lg bg-slate-50 border border-slate-100">
          <div className="p-1.5 rounded-md bg-blue-50 text-blue-600">
            <Mic className="w-4 h-4" />
          </div>
          <div>
            <div className="font-medium text-slate-800 text-xs">Sustained "Aaah"</div>
            <div className="text-[11px] text-slate-500">Steady pitch for 8–10s</div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 p-2 rounded-lg bg-slate-50 border border-slate-100">
          <div className="p-1.5 rounded-md bg-blue-50 text-blue-600">
            <Clock className="w-4 h-4" />
          </div>
          <div>
            <div className="font-medium text-slate-800 text-xs">2–3 Takes Ideal</div>
            <div className="text-[11px] text-slate-500">Catches diurnal variance</div>
          </div>
        </div>
      </div>
    </div>
  );
};
