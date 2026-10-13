import { useEffect, useState } from 'react'
import { Disclaimer } from '../components/Disclaimer'
import { fetchPredictions } from '../services/api'
import type { PredictionHistoryItem } from '../types'

export function PredictionsPage() {
  const [rows, setRows] = useState<PredictionHistoryItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPredictions()
      .then(setRows)
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-3xl font-semibold">Prediction history</h2>
        <p className="mt-2 text-[var(--muted)]">
          Fingerprint images are not stored by default. Only hashes, predictions, and model metadata are retained.
        </p>
      </div>
      <Disclaimer />
      {error ? <p className="text-[var(--danger)]">{error}</p> : null}
      <div className="overflow-x-auto rounded-2xl border border-[var(--line)] bg-[var(--card)]">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-[#f3efe7] text-xs uppercase tracking-wide text-[var(--muted)]">
            <tr>
              <th className="px-4 py-3">Timestamp</th>
              <th className="px-4 py-3">Prediction</th>
              <th className="px-4 py-3">Confidence</th>
              <th className="px-4 py-3">Model</th>
              <th className="px-4 py-3">Request</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td className="px-4 py-6 text-[var(--muted)]" colSpan={5}>
                  No predictions logged yet.
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr key={row.request_id} className="border-t border-[var(--line)]">
                  <td className="px-4 py-3">{new Date(row.timestamp).toLocaleString()}</td>
                  <td className="px-4 py-3 font-medium">{row.prediction}</td>
                  <td className="px-4 py-3">{(row.confidence * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3">{row.model_id}</td>
                  <td className="px-4 py-3 font-mono text-xs">{row.request_id.slice(0, 10)}…</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
