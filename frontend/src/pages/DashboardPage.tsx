import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Disclaimer } from '../components/Disclaimer'
import { MetricCard } from '../components/MetricCard'
import { fetchDashboard } from '../services/api'
import type { DashboardSummary } from '../types'

function fmtPct(value: number | null | undefined) {
  if (value == null) return 'Not evaluated yet'
  return `${(value * 100).toFixed(2)}%`
}

export function DashboardPage() {
  const [data, setData] = useState<DashboardSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDashboard()
      .then(setData)
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) {
    return (
      <div className="space-y-4">
        <h2 className="brand-display text-4xl font-semibold">Goodbye to Blood Test</h2>
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-red-800">
          Could not load dashboard. Start the API on port 8000 first.
          <div className="mt-2 text-sm">{error}</div>
        </div>
      </div>
    )
  }

  if (!data) {
    return <p className="text-[var(--muted)]">Loading…</p>
  }

  const chartData = Object.entries(data.class_distribution).map(([label, count]) => ({
    label,
    count,
  }))

  return (
    <div className="space-y-6 motion-rise">
      <section className="relative overflow-hidden rounded-[28px] border border-[var(--line)] bg-white/70 px-6 py-8 shadow-sm lg:px-10">
        <div className="pointer-events-none absolute -right-8 -top-10 h-44 w-44 rounded-full bg-[var(--accent-soft)] blur-2xl" />
        <p className="text-[11px] uppercase tracking-[0.2em] text-[var(--muted)]">Welcome</p>
        <h2 className="brand-display mt-2 max-w-3xl text-4xl font-semibold leading-tight lg:text-5xl">
          Goodbye to Blood Test
        </h2>
        <p className="mt-4 max-w-2xl text-[var(--muted)]">
          An experimental AI research app that estimates ABO/Rh blood group from a fingerprint image. Fast to try with
          your own photos. Not a replacement for lab testing.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            to="/predict"
            className="rounded-2xl bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-[var(--glow)]"
          >
            Scan a fingerprint
          </Link>
          <Link
            to="/research"
            className="rounded-2xl border border-[var(--line)] bg-white px-5 py-3 text-sm font-semibold text-[var(--ink)]"
          >
            How it works
          </Link>
        </div>
      </section>

      <Disclaimer />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Training images on disk" value={data.total_dataset_images.toLocaleString()} />
        <MetricCard
          label="Active AI model"
          value={`${data.deployed_model_name ?? '—'} ${data.deployed_model_version ?? ''}`}
          hint={
            data.deployed_model_accuracy != null
              ? `Old notebook baseline ${fmtPct(data.deployed_model_accuracy)}`
              : undefined
          }
        />
        <MetricCard
          label="Macro F1 (notebook)"
          value={data.deployed_model_macro_f1?.toFixed(2) ?? 'Not evaluated yet'}
        />
        <MetricCard label="Predictions so far" value={String(data.total_predictions)} />
      </div>

      <section className="glass-card rounded-[28px] p-5">
        <h3 className="brand-display text-2xl font-semibold">Built-in dataset mix</h3>
        <p className="mt-1 text-sm text-[var(--muted)]">
          These bars show the original 6,000-image research set. Your new personal images are tested on the Predict page
          and do not need to be added here first.
        </p>
        <div className="mt-4 h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e1ebe6" />
              <XAxis dataKey="label" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#0f6a5c" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}
