import { Disclaimer } from '../components/Disclaimer'

export function ResearchPage() {
  return (
    <div className="space-y-5 motion-rise">
      <h2 className="brand-display text-4xl font-semibold">How Goodbye to Blood Test works</h2>
      <Disclaimer />

      <div className="space-y-4 text-sm leading-relaxed">
        <section className="glass-card rounded-[24px] p-5">
          <h3 className="brand-display text-xl font-semibold">What the app does today</h3>
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-[var(--muted)]">
            <li>You upload a fingerprint image (BMP/PNG/JPG) on the Scan page.</li>
            <li>The backend checks the file is a real readable image.</li>
            <li>It resizes the image to 256×256 and applies ResNet preprocessing.</li>
            <li>The approved AI model (ResNet50 v1) outputs 8 softmax class scores: A+, A-, AB+, AB-, B+, B-, O+, O-.</li>
            <li>The highest score is shown as the model prediction (uncalibrated; not a lab probability).</li>
            <li>By default the raw image is deleted after analysis; only a hash + prediction are saved.</li>
          </ol>
        </section>

        <section className="glass-card rounded-[24px] p-5">
          <h3 className="brand-display text-xl font-semibold">What it does not do</h3>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-[var(--muted)]">
            <li>It does not read blood, antigens, or DNA.</li>
            <li>It does not prove a biological fingerprint↔blood-group law.</li>
            <li>High confidence is not medical certainty.</li>
            <li>Training/retraining from your private labelled dataset is the next product phase.</li>
          </ul>
        </section>

        <section className="glass-card rounded-[24px] p-5">
          <h3 className="brand-display text-xl font-semibold">Testing with your own images</h3>
          <p className="mt-3 text-[var(--muted)]">
            For prediction right now: just open <strong>Scan fingerprint</strong> and upload any new image. No labels
            needed.
          </p>
          <p className="mt-3 text-[var(--muted)]">
            For a future labelled dataset (retraining), put images into folders under <code>external_data/</code> —
            see that folder’s README.
          </p>
        </section>
      </div>
    </div>
  )
}
