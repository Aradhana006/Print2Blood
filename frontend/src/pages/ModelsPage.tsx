import { useEffect, useState } from 'react'
import { Disclaimer } from '../components/Disclaimer'
import { fetchModels } from '../services/api'
import type { ModelInfo } from '../types'

export function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div className="space-y-5">
      <div>
        <h2 className="brand-display text-4xl font-semibold">AI model</h2>
        <p className="mt-2 text-[var(--muted)]">
          Goodbye to Blood Test currently uses the existing ResNet50 weights as <strong>ResNet50 v1</strong> without
          modifying them.
        </p>
      </div>
      <Disclaimer />
      {error ? <p className="text-[var(--danger)]">{error}</p> : null}
      <div className="grid gap-4">
        {models.map((model) => (
          <article key={model.id} className="rounded-2xl border border-[var(--line)] bg-[var(--card)] p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xl font-semibold">
                {model.model_name} {model.version}
              </h3>
              <span className="rounded-full bg-[#e7f3f7] px-3 py-1 text-xs font-semibold uppercase text-[var(--accent)]">
                {model.is_active ? 'Deployed' : model.status}
              </span>
            </div>
            <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-[var(--muted)]">Architecture</dt>
                <dd>{model.architecture}</dd>
              </div>
              <div>
                <dt className="text-[var(--muted)]">Image size</dt>
                <dd>{model.image_size}×{model.image_size}</dd>
              </div>
              <div>
                <dt className="text-[var(--muted)]">Historical notebook accuracy</dt>
                <dd>
                  {model.accuracy != null
                    ? `${(model.accuracy * 100).toFixed(2)}% (random 80/20 split, not master split)`
                    : 'Not evaluated yet'}
                </dd>
              </div>
              <div>
                <dt className="text-[var(--muted)]">Macro F1</dt>
                <dd>{model.macro_f1 != null ? model.macro_f1.toFixed(2) : 'Not evaluated yet'}</dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-[var(--muted)]">Split type</dt>
                <dd>{model.split_type ?? 'Not evaluated yet'}</dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-[var(--muted)]">Notes</dt>
                <dd>{model.notes ?? '—'}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </div>
  )
}
