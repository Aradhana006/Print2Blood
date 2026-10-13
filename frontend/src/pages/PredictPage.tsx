import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Disclaimer } from '../components/Disclaimer'
import { predictFingerprint } from '../services/api'
import type { PredictResponse } from '../types'

const ACCEPT = '.bmp,.png,.jpg,.jpeg,image/bmp,image/png,image/jpeg'

export function PredictPage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const chartData = useMemo(() => {
    if (!result) return []
    return Object.entries(result.probabilities)
      .map(([label, probability]) => ({
        label,
        probability: Number((probability * 100).toFixed(2)),
      }))
      .sort((a, b) => b.probability - a.probability)
  }, [result])

  function onFile(next: File | null) {
    setFile(next)
    setResult(null)
    setError(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(next ? URL.createObjectURL(next) : null)
  }

  async function onAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const data = await predictFingerprint(file)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 motion-rise">
      <section className="relative overflow-hidden rounded-[28px] border border-[var(--line)] bg-[linear-gradient(135deg,#0a3d36_0%,#0f6a5c_55%,#1d7a55_100%)] px-6 py-8 text-white shadow-xl shadow-[var(--glow)] lg:px-10 lg:py-10">
        <div className="pointer-events-none absolute -right-10 top-0 h-56 w-56 rounded-full bg-[#c45c26]/30 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 left-20 h-40 w-40 rounded-full bg-white/10 blur-3xl" />
        <p className="text-[11px] uppercase tracking-[0.22em] text-[#b7e0d5]">Primary workflow</p>
        <h2 className="brand-display mt-3 max-w-2xl text-4xl font-semibold leading-tight lg:text-5xl">
          Goodbye to Blood Test
        </h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-[#d7efe8]">
          Drop in a new fingerprint image from your phone, scanner, or folder. The app sends it to the AI model and
          returns an experimental blood-group prediction with confidence scores.
        </p>
      </section>

      <Disclaimer />

      <div className="grid gap-5 xl:grid-cols-2">
        <section className="glass-card rounded-[28px] p-5 lg:p-6">
          <label
            htmlFor="fingerprint-upload"
            className={[
              'flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-[24px] border border-dashed px-4 py-8 text-center transition duration-200',
              dragOver
                ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                : 'border-[#b7c7c0] bg-[#f7fbf9]',
            ].join(' ')}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragOver(false)
              const dropped = e.dataTransfer.files?.[0]
              if (dropped) onFile(dropped)
            }}
          >
            {preview ? (
              <img
                src={preview}
                alt="Fingerprint preview"
                className="max-h-60 rounded-2xl object-contain shadow-lg shadow-black/10"
              />
            ) : (
              <>
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
                  <span className="text-2xl">+</span>
                </div>
                <p className="brand-display text-2xl font-semibold">Upload your fingerprint</p>
                <p className="mt-2 max-w-sm text-sm text-[var(--muted)]">
                  Drag & drop or browse. Supports BMP, PNG, JPG, JPEG. Use your own new images — labels are not required
                  for prediction.
                </p>
              </>
            )}
            <input
              id="fingerprint-upload"
              type="file"
              accept={ACCEPT}
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={!file || loading}
              onClick={onAnalyze}
              className="rounded-2xl bg-[var(--accent)] px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-[var(--glow)] transition hover:translate-y-[-1px] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? 'Analyzing…' : 'Analyze fingerprint'}
            </button>
            {file ? <span className="text-sm text-[var(--muted)]">{file.name}</span> : null}
          </div>
          {error ? <p className="mt-3 text-sm text-[var(--danger)]">{error}</p> : null}
        </section>

        <section className="glass-card rounded-[28px] p-5 lg:p-6">
          <h3 className="brand-display text-2xl font-semibold">Prediction result</h3>
          {!result ? (
            <div className="mt-10 rounded-2xl border border-dashed border-[var(--line)] bg-[#f8fbfa] px-5 py-10 text-center text-sm text-[var(--muted)]">
              Your result will appear here after analysis — predicted group, confidence, and all 8 class scores.
            </div>
          ) : (
            <div className="mt-4 space-y-4 motion-rise">
              <div className="rounded-[24px] bg-[linear-gradient(145deg,#122722,#0f6a5c)] px-5 py-7 text-white">
                <p className="text-[11px] uppercase tracking-[0.16em] text-[#9fd0c4]">Model prediction</p>
                <p className="brand-display mt-2 text-6xl font-semibold">{result.prediction}</p>
                <p className="mt-3 text-[#d2ebe4]">
                  Confidence {(result.confidence * 100).toFixed(2)}% · {result.model_name} {result.model_version}
                </p>
              </div>

              <div>
                <p className="text-sm font-semibold">Top predictions</p>
                <ul className="mt-2 space-y-1 text-sm">
                  {result.top3.map((item) => (
                    <li
                      key={item.label}
                      className="flex justify-between rounded-xl bg-[#f4f8f6] px-3 py-2"
                    >
                      <span>{item.label}</span>
                      <span className="font-medium">{(item.probability * 100).toFixed(2)}%</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e1ebe6" />
                    <XAxis type="number" domain={[0, 100]} unit="%" />
                    <YAxis type="category" dataKey="label" width={40} />
                    <Tooltip />
                    <Bar dataKey="probability" fill="#0f6a5c" radius={[0, 8, 8, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-1 text-xs text-[var(--muted)]">
                <p>Request ID: {result.request_id}</p>
                <p>Image hash stored (image itself is not kept by default)</p>
              </div>
              <Disclaimer />
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
