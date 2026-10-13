import { Disclaimer } from '../components/Disclaimer'

export function ComingSoonPage({ title, phase }: { title: string; phase: string }) {
  return (
    <div className="space-y-5">
      <h2 className="text-3xl font-semibold">{title}</h2>
      <Disclaimer />
      <div className="rounded-2xl border border-[var(--line)] bg-[var(--card)] p-6 shadow-sm">
        <p className="text-lg font-medium">{phase} is planned next.</p>
        <p className="mt-2 text-sm text-[var(--muted)]">
          This page is intentionally empty in Phase 1 so the prediction workflow stays focused and honest about what
          is implemented.
        </p>
      </div>
    </div>
  )
}
