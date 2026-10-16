export interface RecordingResult {
  filename: string;
  duration_seconds: number;
  probability_pd: number;
  recording_status: string;
}

export interface StatisticalDistribution {
  mean_probability: number;
  top2_probability: number;
  max_probability: number;
  min_probability: number;
  std_deviation: number;
  recordings_analyzed: number;
}

export interface PredictionResponse {
  diagnosis: string;
  is_pd: boolean;
  triage_level?: "LOW_RISK" | "BORDERLINE" | "HIGH_RISK";
  triage_recommendation?: string;
  confidence_level: string;
  meta_classifier_score: number;
  threshold: number;
  screening_cutoff?: number;
  diagnostic_cutoff?: number;
  recordings_analyzed: number;
  distribution: StatisticalDistribution;
  recordings: RecordingResult[];
  clinical_warnings: string[];
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  device: string;
  torch_version: string;
  supported_formats: string[];
}

export interface AudioTake {
  id: string;
  file: File;
  name: string;
  size: number;
  duration?: number;
  previewUrl?: string;
  source: "mic" | "upload";
}
