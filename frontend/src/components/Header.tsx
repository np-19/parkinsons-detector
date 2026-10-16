import React from "react";
import { Activity, Cpu, ShieldCheck } from "lucide-react";
import { HealthResponse } from "../types/api";

interface HeaderProps {
  health: HealthResponse | null;
  loading: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, loading }) => {
  return (
    <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-xs">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-xs">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12h3l2-6 4 12 3-9 2 6h6" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-semibold text-slate-900 tracking-tight text-lg">NeuroVocal</h1>
              <span className="text-[11px] font-medium bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full border border-blue-200/60">
                v2.0
              </span>
            </div>
            <p className="text-xs text-slate-500">Acoustic Parkinson's Screening Engine</p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-2 sm:gap-3">
          {loading ? (
            <div className="flex items-center gap-1.5 text-xs text-slate-400 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-200">
              <Activity className="w-3.5 h-3.5 animate-pulse text-blue-500" />
              <span>Connecting...</span>
            </div>
          ) : health && health.model_loaded ? (
            <div className="flex items-center gap-2">
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-600 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-200">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span className="capitalize">{health.device.replace(":0", "")}</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-medium">Model Ready (91.2%)</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-xs text-amber-700 bg-amber-50 px-2.5 py-1 rounded-full border border-amber-200">
              <ShieldCheck className="w-3.5 h-3.5 text-amber-500" />
              <span>Backend Offline</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
