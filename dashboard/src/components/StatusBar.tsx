import { useRef, type ChangeEvent } from 'react'

export type HealthState =
  | { kind: 'checking' }
  | { kind: 'ready'; modelName: string; lastCheckedAt: number }
  | { kind: 'error'; message: string; lastCheckedAt: number }

interface StatusBarProps {
  health: HealthState
  inferenceMs: number | null
  steps: number
  currentStep?: number
  onStepsChange: (steps: number) => void
  hasScenario: boolean
  scenarioLabel?: string
  geographyLabel?: string | null
  onImportGeographyFile?: (file: File) => void
  onClearGeography?: () => void
  isSimulating: boolean
  onRefreshHealth: () => void
  onResetScenario: () => void
}

export function StatusBar({
  health,
  inferenceMs,
  steps,
  currentStep,
  onStepsChange,
  hasScenario,
  scenarioLabel,
  geographyLabel,
  onImportGeographyFile,
  onClearGeography,
  isSimulating,
  onRefreshHealth: _onRefreshHealth,
  onResetScenario,
}: StatusBarProps) {
  const geoInputRef = useRef<HTMLInputElement | null>(null)

  const handleGeoFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && onImportGeographyFile) {
      onImportGeographyFile(file)
    }
    e.target.value = ''
  }

  const handleStepInput = (raw: string) => {
    const parsed = Number.parseInt(raw, 10)
    if (!Number.isNaN(parsed)) {
      onStepsChange(Math.max(1, Math.min(60, parsed)))
    }
  }

  const pillClass =
    'inline-flex items-center rounded-full border border-white/15 bg-white/[0.06] backdrop-blur-md shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]'

  return (
    <header className="sticky top-0 z-30 h-16 border-b border-white/10 bg-[#0D1210]/85 backdrop-blur-md px-6">
      <div className="mx-auto flex h-full max-w-[1680px] items-center justify-between gap-4">
        {/* Wordmark left */}
        <div className="flex min-w-0 items-center gap-3">
          <h1 className="font-serif-display text-2xl leading-none tracking-tight text-[#ECE7DA]">
            Monsoon Twin
          </h1>
          {hasScenario && (
            <div className="hidden items-center gap-2.5 text-xs text-white/60 sm:flex">
              <span className="text-white/25">/</span>
              <span className="max-w-[180px] truncate font-mono text-[#ECE7DA]">
                {scenarioLabel ?? 'Loaded scenario'}
              </span>
              <button
                type="button"
                onClick={onResetScenario}
                disabled={isSimulating}
                className={`ui-transition ${pillClass} px-3.5 py-1 text-[13px] text-[#7FB5B0] hover:border-white/30 hover:bg-white/[0.11] disabled:opacity-40 cursor-pointer`}
              >
                Change
              </button>
            </div>
          )}
        </div>

        {/* Big glassy rounded pill controls & telemetry right */}
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-[13px] text-white/75 tabular-nums">
          {onImportGeographyFile && (
            <>
              <input
                ref={geoInputRef}
                type="file"
                accept=".json,application/json"
                onChange={handleGeoFileChange}
                className="hidden"
              />
              {geographyLabel ? (
                <span className={`${pillClass} gap-2 px-4 py-1.5 font-sans text-[13px]`}>
                  <span className="text-white/60">Mask:</span>
                  <span className="max-w-[110px] truncate font-mono text-[#ECE7DA]">
                    {geographyLabel}
                  </span>
                  {onClearGeography && (
                    <button
                      type="button"
                      onClick={onClearGeography}
                      className="ui-transition rounded-full border border-white/15 bg-white/[0.08] px-2.5 py-0.5 text-xs text-[#7FB5B0] hover:border-white/30 hover:bg-white/[0.14] cursor-pointer"
                    >
                      Clear
                    </button>
                  )}
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => geoInputRef.current?.click()}
                  className={`ui-transition ${pillClass} px-4 py-1.5 font-sans text-[13px] text-white/90 hover:border-white/30 hover:bg-white/[0.11] cursor-pointer`}
                >
                  Import geography
                </button>
              )}
            </>
          )}

          {health.kind === 'ready' ? (
            <span className={`${pillClass} px-4 py-1.5 text-[#ECE7DA]`}>
              <span className="max-w-[180px] truncate">{health.modelName}</span>
            </span>
          ) : health.kind === 'error' ? (
            <span className="inline-flex items-center rounded-full border border-[#D9593D]/45 bg-[#D9593D]/10 px-4 py-1.5 text-[#D9593D] backdrop-blur-md">
              offline
            </span>
          ) : null}

          <span className={`${pillClass} px-4 py-1.5 text-[#ECE7DA]`}>
            {isSimulating
              ? 'simulating…'
              : inferenceMs !== null
                ? `${Math.round(inferenceMs)} ms`
                : '— ms'}
          </span>

          <label
            htmlFor="status-steps-input"
            className={`${pillClass} gap-2 py-1 pr-1.5 pl-4`}
          >
            <span className="text-white/75">
              step {hasScenario && currentStep ? `${currentStep}/` : ''}
            </span>
            <input
              id="status-steps-input"
              type="number"
              min={1}
              max={60}
              value={steps}
              disabled={isSimulating}
              onChange={(e) => handleStepInput(e.target.value)}
              aria-label="Forecast rollout steps"
              className="ui-transition w-11 rounded-full border border-white/15 bg-black/30 px-2 py-0.5 text-center font-mono text-[13px] text-[#ECE7DA] focus:border-white/40 focus:outline-none disabled:opacity-40"
            />
          </label>

          {hasScenario && (
            <button
              type="button"
              onClick={onResetScenario}
              disabled={isSimulating}
              className={`ui-transition ${pillClass} px-3.5 py-1.5 text-[13px] font-sans text-[#7FB5B0] hover:border-white/30 hover:bg-white/[0.11] sm:hidden cursor-pointer`}
            >
              Change
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
