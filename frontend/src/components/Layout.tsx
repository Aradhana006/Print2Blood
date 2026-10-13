import { NavLink, Outlet } from 'react-router-dom'

const nav = [
  { to: '/', label: 'Home' },
  { to: '/predict', label: 'Scan fingerprint' },
  { to: '/predictions', label: 'History' },
  { to: '/models', label: 'AI model' },
  { to: '/datasets', label: 'My datasets', soon: true },
  { to: '/experiments', label: 'Training', soon: true },
  { to: '/reports', label: 'Reports', soon: true },
  { to: '/research', label: 'How it works' },
]

export function Layout() {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[280px_1fr]">
      <aside className="relative overflow-hidden border-r border-white/10 bg-[var(--accent-deep)] px-5 py-7 text-[#eef7f4]">
        <div className="pointer-events-none absolute -left-16 top-10 h-48 w-48 rounded-full bg-[var(--accent)]/30 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 right-0 h-40 w-40 rounded-full bg-[var(--signal)]/20 blur-3xl" />

        <div className="relative mb-10">
          <p className="text-[11px] uppercase tracking-[0.22em] text-[#9fd0c4]">Experimental AI</p>
          <h1 className="brand-display mt-3 text-[1.85rem] font-semibold leading-tight text-white">
            Goodbye to
            <br />
            Blood Test
          </h1>
          <p className="mt-3 text-sm leading-relaxed text-[#b7d7cf]">
            Upload a fingerprint. Get an experimental blood-group model prediction in seconds.
          </p>
        </div>

        <nav className="relative space-y-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                [
                  'block rounded-xl px-3 py-2.5 text-sm transition duration-200',
                  isActive
                    ? 'bg-white text-[var(--accent-deep)] shadow-lg shadow-black/10'
                    : 'text-[#cfe3dd] hover:bg-white/8 hover:text-white',
                  item.soon ? 'opacity-55' : '',
                ].join(' ')
              }
            >
              <span className="flex items-center justify-between gap-2">
                <span>{item.label}</span>
                {item.soon ? <span className="text-[10px] uppercase tracking-wide">Soon</span> : null}
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="relative mt-10 rounded-2xl border border-white/10 bg-black/15 p-4 text-xs leading-relaxed text-[#c5ddd6]">
          Not a medical device. Model predictions are research outputs and must never replace a laboratory blood test.
        </div>
      </aside>

      <main className="px-5 py-6 lg:px-9 lg:py-8">
        <Outlet />
      </main>
    </div>
  )
}
