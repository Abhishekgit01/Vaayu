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
  onRefreshHealth,
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

  const dotColor =
    health.kind === 'ready'
      ? '#8DB580'
      : health.kind === 'checking'
        ? '#F0B84B'
        : '#D9593D'

  const statusWord =
    health.kind === 'ready'
      ? 'ready'
      : health.kind === 'checking'
        ? 'checking'
        : 'offline'

  return (
    <header className="sticky top-0 z-30 h-14 border-b border-[#26312B] bg-[#0D1210] px-6">
      <div className="mx-auto flex h-full max-w-[1680px] items-center justify-between gap-4">
        {/* Wordmark left */}
        <div className="flex min-w-0 items-baseline gap-4">
          <h1 className="font-serif-display text-2xl leading-none tracking-tight text-[#ECE7DA]">
            Monsoon Twin
          </h1>
          {hasScenario && (
            <div className="hidden items-baseline gap-2.5 text-xs text-[#8C958D] sm:flex">
              <span className="text-[#26312B]">/</span>
              <span className="max-w-[180px] truncate font-mono text-[#ECE7DA]">
                {scenarioLabel ?? 'Loaded scenario'}
              </span>
              <button
                type="button"
                onClick={onResetScenario}
                disabled={isSimulating}
                className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline disabled:opacity-40 cursor-pointer"
              >
                Change
              </button>
            </div>
          )}
        </div>

        {/* Plain-text telemetry right */}
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 font-mono text-xs text-[#8C958D] tabular-nums">
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
                <span className="inline-flex items-center gap-1.5 font-sans text-xs">
                  <span className="text-[#8C958D]">Mask:</span>
                  <span className="max-w-[110px] truncate font-mono text-[#ECE7DA]">
                    {geographyLabel}
                  </span>
                  {onClearGeography && (
                    <button
                      type="button"
                      onClick={onClearGeography}
                      className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
                    >
                      Clear
                    </button>
                  )}
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => geoInputRef.current?.click()}
                  className="ui-transition font-sans text-xs text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
                >
                  Import geography
                </button>
              )}
              <span className="text-[#5A645D]">·</span>
            </>
          )}

          <button
            type="button"
            onClick={onRefreshHealth}
            title="Poll POST /health"
            className="ui-transition inline-flex items-center gap-1.5 hover:text-[#ECE7DA] cursor-pointer"
          >
            <span className="font-sans text-[#8C958D]">Server</span>
            <span
              aria-hidden="true"
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: dotColor }}
            />
            <span className={health.kind === 'error' ? 'text-[#D9593D]' : 'text-[#ECE7DA]'}>
              {statusWord}
            </span>
          </button>

          {health.kind === 'ready' && (
            <>
              <span className="text-[#5A645D]">·</span>
              <span className="max-w-[180px] truncate text-[#8C958D]">{health.modelName}</span>
            </>
          )}

          <span className="text-[#5A645D]">·</span>
          <span className="text-[#ECE7DA]">
            {isSimulating
              ? 'simulating…'
              : inferenceMs !== null
                ? `${Math.round(inferenceMs)} ms`
                : '— ms'}
          </span>

          <span className="text-[#5A645D]">·</span>
          <label htmlFor="status-steps-input" className="inline-flex items-center gap-1">
            <span>
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
              className="ui-transition w-10 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1 py-0.5 text-center font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none disabled:opacity-40"
            />
          </label>

          {hasScenario && (
            <button
              type="button"
              onClick={onResetScenario}
              disabled={isSimulating}
              className="ui-transition ml-1 text-xs font-sans text-[#7FB5B0] underline-offset-4 hover:underline sm:hidden cursor-pointer"
            >
              Change
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
