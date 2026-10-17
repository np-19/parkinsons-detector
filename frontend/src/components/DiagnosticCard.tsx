import React from "react";
import { CheckCircle2, AlertTriangle, AlertCircle, Printer, RotateCcw, ShieldCheck } from "lucide-react";
import { PredictionResponse } from "../types/api";

interface DiagnosticCardProps {
  report: PredictionResponse;
  onReset: () => void;
}

export const DiagnosticCard: React.FC<DiagnosticCardProps> = ({ report, onReset }) => {
  const participantScore = report.participant_mean_score ?? report.meta_classifier_score;
  const scorePercent = Math.min(Math.max(participantScore * 100, 0), 100);

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-4 sm:p-7 shadow-xs mb-8 max-w-5xl mx-auto w-full">
      {/* Top Clinical Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 sm:pb-5 border-b border-slate-100">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Acoustic Biomarker Assessment Report
          </div>
          <div className="text-xs text-slate-500 mt-0.5">
            Analyzed {report.recordings_analyzed} {report.recordings_analyzed === 1 ? "take" : "takes"} at {new Date(report.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </div>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <button
            onClick={handlePrint}
            className="flex-1 sm:flex-initial p-2.5 sm:px-3 sm:py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 active:bg-slate-200 rounded-xl border border-slate-200 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors touch-manipulation"
            title="Print or Save PDF"
          >
            <Printer className="w-3.5 h-3.5 shrink-0" />
            <span>Export PDF</span>
          </button>

          <button
            onClick={onReset}
            className="flex-1 sm:flex-initial p-2.5 sm:px-3 sm:py-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 active:bg-blue-100 rounded-xl border border-blue-200 text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors touch-manipulation"
            title="Start New Assessment"
          >
            <RotateCcw className="w-3.5 h-3.5 shrink-0" />
            <span>New Patient</span>
          </button>
        </div>
      </div>

      {/* Hero Diagnostic Badge with 3-Tier Clinical Triage */}
      {(() => {
        const screeningCutoff = report.screening_cutoff ?? report.threshold - 0.10;
        const triage = report.triage_level || (
          participantScore < screeningCutoff
            ? "LOW_RISK"
            : participantScore < report.threshold
              ? "BORDERLINE"
              : "HIGH_RISK"
        );
        
        let badgeBg = "bg-emerald-50/70 border-emerald-200 text-emerald-950";
        let iconBg = "bg-emerald-100 text-emerald-700";
        let IconComponent = CheckCircle2;
        let triageTag = "Low Clinical Risk (Healthy Voice)";
        let triageTagColor = "bg-emerald-100 text-emerald-800 border-emerald-200";

        if (triage === "BORDERLINE") {
          badgeBg = "bg-amber-50/80 border-amber-300 text-amber-950";
          iconBg = "bg-amber-100 text-amber-700";
          IconComponent = AlertCircle;
          triageTag = "Borderline Triage (Review Recommended)";
          triageTagColor = "bg-amber-100 text-amber-900 border-amber-300";
        } else if (triage === "HIGH_RISK") {
          badgeBg = "bg-rose-50/70 border-rose-200 text-rose-950";
          iconBg = "bg-rose-100 text-rose-700";
          IconComponent = AlertTriangle;
          triageTag = "High Clinical Risk (Acoustic Alert)";
          triageTagColor = "bg-rose-100 text-rose-800 border-rose-200";
        }

        return (
          <div className={`my-5 p-4 sm:p-5 rounded-2xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${badgeBg}`}>
            <div className="flex items-start sm:items-center gap-3.5">
              <div className={`w-11 h-11 rounded-2xl flex items-center justify-center shrink-0 shadow-2xs ${iconBg}`}>
                <IconComponent className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="text-base sm:text-lg font-bold tracking-tight">
                    {report.diagnosis}
                  </div>
                  <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${triageTagColor}`}>
                    {triageTag}
                  </span>
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  Confidence Tier: <span className="font-semibold text-slate-900">{report.confidence_level}</span>
                </div>
              </div>
            </div>

            <div className="text-right sm:border-l sm:border-slate-200/60 sm:pl-5 shrink-0">
              <div className="text-[11px] uppercase tracking-wider text-slate-500 font-medium">Acoustic Biomarker Score</div>
              <div className="text-2xl font-bold font-mono text-slate-900">
                {participantScore.toFixed(4)}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Calibrated 3-Zone Clinical Probability Scale */}
      {(() => {
        const screeningCutoff = report.screening_cutoff ?? report.threshold - 0.10;
        const balancedCutoff = report.threshold ?? 0.36;
        const diagnosticCutoff = report.diagnostic_cutoff ?? report.threshold + 0.10;

        const zone1Width = Math.round(screeningCutoff * 100);
        const zone2Width = Math.round((balancedCutoff - screeningCutoff) * 100);
        const zone3Width = Math.max(100 - zone1Width - zone2Width, 0);

        return (
          <div className="my-5 p-4 sm:p-5 rounded-2xl bg-slate-50/90 border border-slate-200/80">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-600 mb-3 font-medium gap-1">
              <span className="font-semibold text-slate-800">3-Tier Clinical Triage Scale</span>
              <div className="flex items-center gap-3 text-[11px] text-slate-500">
                <span>Screening Cutoff: <strong>{screeningCutoff.toFixed(2)}</strong></span>
                <span>Balanced: <strong>{balancedCutoff.toFixed(2)}</strong></span>
                <span>Diagnostic: <strong>{diagnosticCutoff.toFixed(2)}</strong></span>
              </div>
            </div>

            {/* 3-Zone Segmented Track */}
            <div className="relative w-full h-4 rounded-full overflow-hidden flex bg-slate-200 border border-slate-300/60 shadow-inner">
              {/* Zone 1: Green */}
              <div style={{ width: `${zone1Width}%` }} className="h-full bg-emerald-300/80" title={`Low Risk Zone (<${screeningCutoff.toFixed(2)})`} />
              {/* Zone 2: Amber */}
              <div style={{ width: `${zone2Width}%` }} className="h-full bg-amber-300/90" title={`Borderline Screening Alert Zone (${screeningCutoff.toFixed(2)}–${balancedCutoff.toFixed(2)})`} />
              {/* Zone 3: Red */}
              <div style={{ width: `${zone3Width}%` }} className="h-full bg-rose-300/80" title={`High Risk Zone (≥${diagnosticCutoff.toFixed(2)})`} />
            </div>

            {/* Needle Pin / Marker for Patient Score */}
            <div className="relative w-full h-3">
              <div
                className="absolute top-[-16px] -translate-x-1/2 flex flex-col items-center transition-all duration-500 z-20 pointer-events-none"
                style={{ left: `${scorePercent}%` }}
              >
                <div className="w-2.5 h-2.5 rounded-full bg-slate-900 ring-2 ring-white shadow-sm" />
                <div className="text-[10px] font-bold font-mono text-slate-900 bg-white px-1.5 py-0.5 rounded shadow-xs border border-slate-200 -mt-0.5 whitespace-nowrap">
                  {participantScore.toFixed(3)}
                </div>
              </div>
            </div>

            {/* Zone Labels Below Track */}
            <div className="grid grid-cols-3 text-[10px] sm:text-[11px] font-medium pt-2 text-center text-slate-500">
              <div className="text-left text-emerald-800">
                <span className="font-semibold">0.00 – {screeningCutoff.toFixed(2)}</span>
                <div className="text-[10px] text-slate-400">Low Risk (Healthy)</div>
              </div>
              <div className="text-center text-amber-800">
                <span className="font-semibold">{screeningCutoff.toFixed(2)} – {balancedCutoff.toFixed(2)}</span>
                <div className="text-[10px] text-slate-400">Borderline / ON-State Alert</div>
              </div>
              <div className="text-right text-rose-800">
                <span className="font-semibold">{balancedCutoff.toFixed(2)} – 1.00</span>
                <div className="text-[10px] text-slate-400">High Risk (Dysphonic)</div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Actionable Clinical Recommendation & Next Steps */}
      {report.triage_recommendation && (
        <div className="my-5 p-4 rounded-xl bg-blue-50/70 border border-blue-200/80 text-xs">
          <div className="flex items-center gap-2 font-semibold text-blue-950 mb-1.5">
            <ShieldCheck className="w-4 h-4 text-blue-600 shrink-0" />
            <span>Clinical Triage Recommendation & Decision Support</span>
          </div>
          <p className="text-slate-700 leading-relaxed">
            {report.triage_recommendation}
          </p>
        </div>
      )}

      {report.failed_recordings?.length > 0 && (
        <div className="my-5 p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs">
          <div className="font-semibold text-amber-950 mb-1.5">
            {report.failed_recordings.length} recording(s) were not analyzed
          </div>
          <ul className="list-disc pl-5 text-amber-900 space-y-1">
            {report.failed_recordings.map((failure) => (
              <li key={`${failure.filename}:${failure.error}`}>
                <span className="font-medium">{failure.filename}:</span> {failure.error}
              </li>
            ))}
          </ul>
        </div>
      )}

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

        <div className="border border-slate-200/80 rounded-xl overflow-x-auto scrollbar-thin">
          <table className="w-full text-xs text-left min-w-[500px]">
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
