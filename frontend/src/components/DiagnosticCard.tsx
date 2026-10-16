import React from "react";
import { CheckCircle2, AlertTriangle, Printer, RotateCcw, ShieldCheck } from "lucide-react";
import { PredictionResponse } from "../types/api";

interface DiagnosticCardProps {
  report: PredictionResponse;
  onReset: () => void;
}

export const DiagnosticCard: React.FC<DiagnosticCardProps> = ({ report, onReset }) => {
  const isPositive = report.is_pd;
  const scorePercent = Math.min(Math.max(report.meta_classifier_score * 100, 0), 100);
  const thresholdPercent = report.threshold * 100;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 sm:p-7 shadow-xs mb-8">
      {/* Top Clinical Header */}
      <div className="flex items-center justify-between pb-5 border-b border-slate-100">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Acoustic Biomarker Assessment Report
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            Analyzed {report.recordings_analyzed} {report.recordings_analyzed === 1 ? "take" : "takes"} at {new Date(report.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePrint}
            className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg border border-slate-200 text-xs flex items-center gap-1.5 transition-colors"
            title="Print or Save PDF"
          >
            <Printer className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Export PDF</span>
          </button>

          <button
            onClick={onReset}
            className="p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg border border-blue-200 text-xs flex items-center gap-1.5 transition-colors"
            title="Start New Assessment"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">New Patient</span>
          </button>
        </div>
      </div>

      {/* Hero Diagnostic Badge */}
      <div
        className={`my-6 p-5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
          isPositive
            ? "bg-rose-50/70 border-rose-200 text-rose-950"
            : "bg-emerald-50/70 border-emerald-200 text-emerald-950"
        }`}
      >
        <div className="flex items-start sm:items-center gap-3.5">
          <div
            className={`w-11 h-11 rounded-full flex items-center justify-center shrink-0 ${
              isPositive ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"
            }`}
          >
            {isPositive ? <AlertTriangle className="w-6 h-6" /> : <CheckCircle2 className="w-6 h-6" />}
          </div>
          <div>
            <div className="text-base sm:text-lg font-bold tracking-tight">
              {report.diagnosis}
            </div>
            <div className="text-xs text-slate-600 mt-0.5">
              Confidence Tier: <span className="font-semibold">{report.confidence_level}</span>
            </div>
          </div>
        </div>

        <div className="text-right sm:border-l sm:border-slate-200/60 sm:pl-5 shrink-0">
          <div className="text-[11px] uppercase tracking-wider text-slate-500 font-medium">Meta-Classifier Score</div>
          <div className="text-2xl font-bold font-mono text-slate-900">
            {report.meta_classifier_score.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Calibrated Continuous Linear Gauge */}
      <div className="my-6 p-4 rounded-xl bg-slate-50 border border-slate-200/70">
        <div className="flex items-center justify-between text-xs text-slate-600 mb-2 font-medium">
          <span>Diagnostic Probability Scale</span>
          <span>Decision Boundary: <strong>{report.threshold}</strong></span>
        </div>

        {/* The Track */}
        <div className="relative w-full h-3 bg-slate-200 rounded-full overflow-visible">
          {/* Fill Bar */}
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isPositive ? "bg-rose-500" : "bg-blue-600"
            }`}
            style={{ width: `${scorePercent}%` }}
          />

          {/* Notch for Threshold (0.29) */}
          <div
            className="absolute top-[-4px] bottom-[-4px] w-1 bg-slate-900 rounded-full z-10"
            style={{ left: `${thresholdPercent}%` }}
            title={`Clinical Cutoff Threshold: ${report.threshold}`}
          />
        </div>

        {/* Labels below bar */}
        <div className="flex justify-between text-[11px] text-slate-400 mt-2 font-mono">
          <span>0.00 (Healthy Control)</span>
          <span className="text-slate-700 font-semibold pl-6">▲ Cutoff (0.29)</span>
          <span>1.00 (Severe Dysphonia)</span>
        </div>
      </div>

      {/* 3 Key Statistical Moments Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-6">
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <div className="text-[11px] font-medium text-slate-500 uppercase">Mean Severity</div>
          <div className="text-xl font-bold font-mono text-slate-900 mt-1">
            {report.distribution.mean_probability.toFixed(4)}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Average across takes</div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <div className="text-[11px] font-medium text-slate-500 uppercase">Peak (Top-2)</div>
          <div className="text-xl font-bold font-mono text-slate-900 mt-1">
            {report.distribution.top2_probability.toFixed(4)}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Worst symptomatic takes</div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
          <div className="text-[11px] font-medium text-slate-500 uppercase">Diurnal Fluctuation (σ)</div>
          <div className="text-xl font-bold font-mono text-slate-900 mt-1">
            {report.distribution.std_deviation.toFixed(4)}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Motor variance / ON-OFF cycle</div>
        </div>
      </div>

      {/* Detailed Per-Take Breakdown Table */}
      <div className="my-6">
        <div className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2 px-1">
          Acoustic Session Analysis Breakdown
        </div>

        <div className="border border-slate-200/80 rounded-xl overflow-hidden">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-semibold">Take #</th>
                <th className="py-2.5 px-3 font-semibold">Filename</th>
                <th className="py-2.5 px-3 font-semibold">Duration</th>
                <th className="py-2.5 px-3 font-semibold">Raw P(PD)</th>
                <th className="py-2.5 px-3 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {report.recordings.map((rec, i) => (
                <tr key={i} className="hover:bg-slate-50/50">
                  <td className="py-2.5 px-3 font-medium text-slate-700">Take {i + 1}</td>
                  <td className="py-2.5 px-3 text-slate-900 font-mono truncate max-w-[160px]">{rec.filename}</td>
                  <td className="py-2.5 px-3 text-slate-500 font-mono">{rec.duration_seconds.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 font-mono font-medium text-slate-800">
                    {rec.probability_pd.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium ${
                        rec.recording_status.includes("Tremor")
                          ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                          : "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                      }`}
                    >
                      {rec.recording_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Clinical Observations & Pharmacological Guardrails */}
      {report.clinical_warnings && report.clinical_warnings.length > 0 ? (
        <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200/80 text-amber-900 text-xs">
          <div className="flex items-center gap-1.5 font-semibold text-amber-950 mb-1">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Clinical Advisory Notice</span>
          </div>
          <ul className="list-disc list-inside space-y-1 text-slate-700 pl-1">
            {report.clinical_warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/70 text-xs text-slate-600 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>Multi-session protocol satisfied. Diurnal motor fluctuation successfully accounted for.</span>
        </div>
      )}
    </div>
  );
};
