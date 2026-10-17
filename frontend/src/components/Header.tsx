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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Logo */}
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-xs shrink-0">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="sm:w-6 sm:h-6">
              <path d="M2 12h3l2-6 4 12 3-9 2 6h6" />
            </svg>
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <h1 className="font-semibold text-slate-900 tracking-tight text-base sm:text-lg truncate">NeuroVocal</h1>
            </div>
            <p className="text-[11px] sm:text-xs text-slate-500 truncate hidden xs:block">Acoustic Parkinson's Screening Engine</p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          {loading ? (
            <div className="flex items-center gap-1.5 text-xs text-slate-400 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-200">
              <Activity className="w-3.5 h-3.5 animate-pulse text-blue-500" />
              <span>Connecting...</span>
            </div>
          ) : health && health.model_loaded ? (
            <div className="flex items-center gap-2">
              <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-600 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-200">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span className="capitalize">{health.device.replace(":0", "")}</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
                <span className="font-medium hidden sm:inline">Model Ready</span>
                <span className="font-medium sm:hidden">Ready</span>
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

