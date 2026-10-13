import type { DashboardSummary, ModelInfo, PredictResponse, PredictionHistoryItem } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail: unknown = `Request failed (${res.status})`
    try {
      const data = await res.json()
      detail = data.detail ?? data
    } catch {
      // ignore
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res
}

export async function fetchDashboard(): Promise<DashboardSummary> {
  const res = await ensureOk(await fetch(`${API_BASE}/api/dashboard/summary`))
  return res.json()
}

export async function fetchActiveModel(): Promise<ModelInfo> {
  const res = await ensureOk(await fetch(`${API_BASE}/api/models/active`))
  return res.json()
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await ensureOk(await fetch(`${API_BASE}/api/models`))
  return res.json()
}

export async function fetchPredictions(limit = 50): Promise<PredictionHistoryItem[]> {
  const res = await ensureOk(await fetch(`${API_BASE}/api/predictions?limit=${limit}`))
  return res.json()
}

export async function predictFingerprint(file: File): Promise<PredictResponse> {
  const body = new FormData()
  body.append('file', file)
  const res = await ensureOk(
    await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body,
    }),
  )
  return res.json()
}
