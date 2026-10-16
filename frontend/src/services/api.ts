import { HealthResponse, PredictionResponse } from "../types/api";

const API_BASE = "";

export async function checkBackendHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Healthcheck failed with status ${res.status}`);
  }
  return res.json();
}

export async function predictPatientRecordings(files: File[]): Promise<PredictionResponse> {
  if (files.length === 0) {
    throw new Error("At least one audio file is required.");
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let errorMsg = `Server error (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorMsg = errJson.detail;
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorMsg);
  }

  return res.json();
}

