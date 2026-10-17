import { useState, useEffect } from "react";
import { Mic, UploadCloud, AlertCircle } from "lucide-react";
import { Header } from "./components/Header";
import { ProtocolPills } from "./components/ProtocolPills";
import { AudioRecorder } from "./components/AudioRecorder";
import { Dropzone } from "./components/Dropzone";
import { TakeQueue } from "./components/TakeQueue";
import { DiagnosticCard } from "./components/DiagnosticCard";
import { checkBackendHealth, predictPatientRecordings } from "./services/api";
import { AudioTake, HealthResponse, PredictionResponse } from "./types/api";

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");
  const [takes, setTakes] = useState<AudioTake[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [report, setReport] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    async function loadHealth() {
      try {
        setHealthLoading(true);
        const data = await checkBackendHealth();
        setHealth(data);
      } catch (err) {
        console.warn("Could not reach backend API:", err);
      } finally {
        setHealthLoading(false);
      }
    }
    loadHealth();
  }, []);

  const handleAddMicTake = (file: File) => {
    const newTake: AudioTake = {
      id: `take_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      file,
      name: file.name,
      size: file.size,
      source: "mic",
    };
    setTakes((prev) => [...prev, newTake]);
    setErrorMessage(null);
  };

  const handleAddUploadedFiles = (files: File[]) => {
    const newTakes: AudioTake[] = files.map((file) => ({
      id: `take_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      file,
      name: file.name,
      size: file.size,
      source: "upload",
    }));
    setTakes((prev) => [...prev, ...newTakes]);
    setErrorMessage(null);
  };

  const handleRemoveTake = (id: string) => {
    setTakes((prev) => prev.filter((t) => t.id !== id));
  };

  const handleClearAll = () => {
    setTakes([]);
    setErrorMessage(null);
  };

  const handleRunAssessment = async () => {
    if (takes.length === 0) return;

    try {
      setIsAnalyzing(true);
      setErrorMessage(null);
      const files = takes.map((t) => t.file);
      const result = await predictPatientRecordings(files);
      setReport(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Inference service encountered an error.";
      setErrorMessage(msg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setReport(null);
    setTakes([]);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Header health={health} loading={healthLoading} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-5 lg:py-6 flex flex-col justify-center">
        {/* If report is ready, show diagnostic card */}
        {report ? (
          <DiagnosticCard report={report} onReset={handleReset} />
        ) : (
          <div className="w-full">
            {/* Error Notification */}
            {errorMessage && (
              <div className="mb-4 p-3.5 sm:p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2.5 shadow-2xs">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Responsive Main Layout: Desktop 2-column side-by-side, Mobile single-column stacked */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-6 items-start">
              {/* Left Column on Desktop (5 cols): Protocol & Assessment Queue */}
              <div className="lg:col-span-5 flex flex-col gap-4 order-2 lg:order-1">
                {/* Protocol Pills: Shown here on desktop */}
                <div className="hidden lg:block">
                  <ProtocolPills />
                </div>

                {/* Take Queue */}
                <TakeQueue
                  takes={takes}
                  onRemoveTake={handleRemoveTake}
                  onClearAll={handleClearAll}
                  onRunAssessment={handleRunAssessment}
                  isAnalyzing={isAnalyzing}
                />
              </div>

              {/* Right Column on Desktop (7 cols): Voice Biomarker Acquisition */}
              <div className="lg:col-span-7 flex flex-col gap-4 order-1 lg:order-2">
                {/* Protocol Pills: Shown on mobile at the top */}
                <div className="block lg:hidden">
                  <ProtocolPills />
                </div>

                {/* Primary Voice Biomarker Acquisition Card */}
                <div className="bg-white border border-slate-200/90 rounded-2xl p-5 sm:p-6 shadow-xs flex flex-col justify-between">
                  <div className="mb-4">
                    <h2 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                      Voice Biomarker Acquisition
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Capture sustained vowel phonation (/a/) or upload patient acoustic takes.
                    </p>
                  </div>

                  {/* Mode Toggle Tabs */}
                  <div className="flex rounded-xl bg-slate-100/90 p-1 mb-5 border border-slate-200/70 max-w-md mx-auto w-full">
                    <button
                      onClick={() => setActiveTab("mic")}
                      className={`flex-1 py-2 px-3 text-xs font-medium rounded-lg flex items-center justify-center gap-2 transition-all touch-manipulation ${
                        activeTab === "mic"
                          ? "bg-white text-blue-700 shadow-xs font-semibold"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      <Mic className="w-3.5 h-3.5 shrink-0" />
                      <span>Microphone Phonation</span>
                    </button>

                    <button
                      onClick={() => setActiveTab("upload")}
                      className={`flex-1 py-2 px-3 text-xs font-medium rounded-lg flex items-center justify-center gap-2 transition-all touch-manipulation ${
                        activeTab === "upload"
                          ? "bg-white text-blue-700 shadow-xs font-semibold"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      <UploadCloud className="w-3.5 h-3.5 shrink-0" />
                      <span>Upload Audio File(s)</span>
                    </button>
                  </div>

                  {/* Active Tab Panel */}
                  <div className="min-h-[160px] flex flex-col justify-center">
                    {activeTab === "mic" ? (
                      <AudioRecorder onRecordingComplete={handleAddMicTake} />
                    ) : (
                      <Dropzone onFilesSelected={handleAddUploadedFiles} />
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200/70 bg-white py-3.5 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-[11px] text-slate-400">
          NeuroVocal Clinical Decision Support • Model: Fine-Tuned WaveLM (mPower Cohort)
        </div>
      </footer>
    </div>
  );
}

export default App;
