import { useEffect, useRef, useState } from 'react'
import { analyzeFingerprint } from './services/api'
import type { AnalyzeResponse } from './types'

const ACCEPT = '.bmp,.png,.jpg,.jpeg,image/bmp,image/png,image/jpeg'
type Mode = 'upload' | 'camera'

export default function App() {
  const [mode, setMode] = useState<Mode>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [showScience, setShowScience] = useState(false)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  function clearPreview() {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
  }

  function setImageFile(next: File | null) {
    setFile(next)
    setResult(null)
    setError(null)
    clearPreview()
    if (next) setPreview(URL.createObjectURL(next))
  }

  async function startCamera() {
    setError(null)
    setResult(null)
    try {
      stopCamera()
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraOn(true)
    } catch {
      setError('Camera blocked. Allow permission, or use Upload.')
      setCameraOn(false)
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
  }

  function captureFrame() {
    const video = videoRef.current
    if (!video || !video.videoWidth) {
      setError('Camera is not ready yet.')
      return
    }
    const canvas = document.createElement('canvas')
    const side = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - side) / 2
    const sy = (video.videoHeight - side) / 2
    canvas.width = 512
    canvas.height = 512
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, sx, sy, side, side, 0, 0, 512, 512)
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError('Could not capture image.')
          return
        }
        setImageFile(new File([blob], `live-scan-${Date.now()}.jpg`, { type: 'image/jpeg' }))
        stopCamera()
      },
      'image/jpeg',
      0.92,
    )
  }

  useEffect(() => {
    return () => {
      stopCamera()
      if (preview) URL.revokeObjectURL(preview)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (mode === 'camera') void startCamera()
    else stopCamera()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  async function onAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyzeFingerprint(file))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const scores = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="min-h-screen bg-black px-4 py-8 text-white">
      <div className="mx-auto max-w-xl">
        <header className="mb-6 border border-white/15 bg-neutral-950 px-5 py-6">
          <p className="text-[11px] uppercase tracking-[0.22em] text-neutral-400">AI Research Prototype</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">Goodbye to Blood Test</h1>
          <p className="mt-2 text-sm text-neutral-400">
            Scan or upload a fingerprint. Get quality checks, pattern cues, blood-group prediction, and an AI attention map.
          </p>
        </header>

        <div className="mb-4 grid grid-cols-2 gap-1 border border-white/15 bg-neutral-950 p-1">
          {(['upload', 'camera'] as Mode[]).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={`px-3 py-2.5 text-sm font-medium transition ${
                mode === item ? 'bg-white text-black' : 'text-neutral-400 hover:text-white'
              }`}
            >
              {item === 'upload' ? 'Upload image' : 'Live scanner'}
            </button>
          ))}
        </div>

        <main className="space-y-4">
          <section className="overflow-hidden border border-white/15 bg-neutral-950">
            {mode === 'upload' ? (
              <label className="flex min-h-52 cursor-pointer flex-col items-center justify-center px-4 py-8 text-center">
                {preview ? (
                  <img src={preview} alt="Preview" className="max-h-48 object-contain" />
                ) : (
                  <>
                    <div className="mb-3 flex h-12 w-12 items-center justify-center border border-white/25 text-xl text-white">
                      +
                    </div>
                    <span className="text-sm font-medium text-neutral-200">Drop or choose fingerprint</span>
                    <span className="mt-1 text-xs text-neutral-500">BMP, PNG, JPG</span>
                  </>
                )}
                <input
                  type="file"
                  accept={ACCEPT}
                  className="hidden"
                  onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
                />
              </label>
            ) : (
              <div className="bg-black">
                {cameraOn ? (
                  <div className="relative">
                    <video ref={videoRef} playsInline muted className="h-64 w-full object-cover grayscale" />
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                      <div className="h-40 w-40 rounded-full border border-white/80" />
                    </div>
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center px-4 text-center text-sm text-neutral-400">
                    {preview ? (
                      <img src={preview} alt="Captured" className="max-h-60 object-contain grayscale" />
                    ) : (
                      'Starting camera…'
                    )}
                  </div>
                )}
                <div className="flex gap-2 border-t border-white/10 p-3">
                  {cameraOn ? (
                    <>
                      <button
                        type="button"
                        onClick={captureFrame}
                        className="flex-1 bg-white px-3 py-2.5 text-sm font-semibold text-black"
                      >
                        Capture fingerprint
                      </button>
                      <button
                        type="button"
                        onClick={stopCamera}
                        className="border border-white/20 px-3 py-2.5 text-sm text-white"
                      >
                        Stop
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => void startCamera()}
                      className="flex-1 bg-white px-3 py-2.5 text-sm font-semibold text-black"
                    >
                      {preview ? 'Scan again' : 'Start camera'}
                    </button>
                  )}
                </div>
              </div>
            )}
          </section>

          {file ? <p className="truncate text-xs text-neutral-500">{file.name}</p> : null}

          <button
            type="button"
            disabled={!file || loading}
            onClick={onAnalyze}
            className="w-full bg-white px-4 py-3 text-sm font-semibold text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-35"
          >
            {loading ? 'Analyzing fingerprint…' : 'Analyze with AI'}
          </button>

          {error ? (
            <div className="border border-white/20 bg-neutral-950 px-4 py-3 text-sm text-neutral-200">
              {error}
            </div>
          ) : null}

          {result ? (
            <section className="space-y-4 border border-white/15 bg-neutral-950 p-5">
              <div className="border border-white/15 bg-black px-4 py-5">
                <p className="text-[11px] uppercase tracking-[0.16em] text-neutral-400">Model prediction</p>
                <p className="mt-1 text-5xl font-semibold text-white">{result.prediction}</p>
                <p className="mt-2 text-sm text-neutral-300">
                  {(result.confidence * 100).toFixed(1)}% softmax score · {result.confidence_band.level} band
                </p>
                <p className="mt-1 text-xs text-neutral-500">{result.confidence_band.meaning}</p>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="border border-white/10 bg-black p-3">
                  <p className="text-neutral-500">Quality</p>
                  <p className="mt-1 text-xl font-semibold text-white">{result.quality.scores.overall}</p>
                  <p className="text-neutral-400">{result.quality.verdict}</p>
                </div>
                <div className="border border-white/10 bg-black p-3">
                  <p className="text-neutral-500">Pattern</p>
                  <p className="mt-1 text-xl font-semibold text-white">
                    {result.quality.pattern.pattern_estimate.replace('-like', '')}
                  </p>
                  <p className="text-neutral-400">cue</p>
                </div>
                <div className="border border-white/10 bg-black p-3">
                  <p className="text-neutral-500">Model</p>
                  <p className="mt-1 text-xl font-semibold text-white">{result.model_name}</p>
                  <p className="text-neutral-400">{result.model_version}</p>
                </div>
              </div>

              <div>
                <p className="mb-2 text-sm font-medium text-neutral-200">AI attention map</p>
                <img
                  src={`data:image/jpeg;base64,${result.explainability.heatmap_overlay_jpeg_base64}`}
                  alt="AI attention heatmap"
                  className="w-full border border-white/10 grayscale"
                />
                <p className="mt-2 text-[11px] leading-relaxed text-neutral-500">
                  {result.explainability.interpretation}
                </p>
              </div>

              <div>
                <p className="mb-2 text-sm font-medium text-neutral-200">Class scores (softmax, uncalibrated)</p>
                <ul className="space-y-2">
                  {scores.map(([label, prob]) => (
                    <li key={label}>
                      <div className="mb-1 flex justify-between text-xs text-neutral-400">
                        <span>{label}</span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 overflow-hidden bg-neutral-800">
                        <div
                          className="h-full bg-white"
                          style={{ width: `${Math.max(prob * 100, 1)}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="border border-white/10 bg-black p-3 text-xs leading-relaxed text-neutral-400">
                <p className="font-semibold text-neutral-200">AI insights</p>
                <ul className="mt-2 list-disc space-y-1 pl-4">
                  {result.ai_insights.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                  <li>{result.quality.tip}</li>
                </ul>
              </div>
            </section>
          ) : null}

          <button
            type="button"
            onClick={() => setShowScience((v) => !v)}
            className="text-xs font-medium text-neutral-300 underline"
          >
            {showScience ? 'Hide' : 'How does this AI work?'}
          </button>

          {showScience ? (
            <div className="space-y-2 border border-white/15 bg-neutral-950 p-4 text-xs leading-relaxed text-neutral-400">
              <p>
                The model studies ridge patterns in the fingerprint image (loops, whorls, arches, texture). It does not
                read blood antigens.
              </p>
              <p>
                Pipeline: capture → quality check → resize 256×256 → ResNet50 → 8 blood-group probabilities →
                explainability map.
              </p>
              <p>This is experimental research, not a laboratory blood test.</p>
            </div>
          ) : null}

          <p className="text-xs leading-relaxed text-neutral-600">
            Research prototype only. Not a clinical diagnostic tool.
          </p>
        </main>
      </div>
    </div>
  )
}
