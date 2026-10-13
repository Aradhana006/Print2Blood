type Props = {
  label: string
  value: string
  hint?: string
}

export function MetricCard({ label, value, hint }: Props) {
  return (
    <div className="glass-card motion-rise rounded-2xl p-4">
      <p className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">{label}</p>
      <p className="brand-display mt-2 text-[1.7rem] font-semibold text-[var(--ink)]">{value}</p>
      {hint ? <p className="mt-1 text-xs text-[var(--muted)]">{hint}</p> : null}
    </div>
  )
}
