import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { checkHealth, runSimulation } from './api/client'
import type {
  DisplayChannelId,
  ScenarioData,
  State3D,
  StateSequence,
} from './api/types'
import { ComparisonView } from './components/ComparisonView'
import { EmptyState } from './components/EmptyState'
import { ErrorBanner } from './components/ErrorBanner'
import { ImpactPanel } from './components/ImpactPanel'
import {
  ChannelSelector,
  MapControls,
  type CompareMode,
  type PlaybackSpeed,
} from './components/MapControls'
import { ShockPanel } from './components/ShockPanel'
import { SpreadChart } from './components/SpreadChart'
import { StatusBar, type HealthState } from './components/StatusBar'
import {
  generateSyntheticRollout,
  generateSyntheticScenario,
  isDevPreviewEnabled,
} from './dev/syntheticScenario'
import {
  parseAndValidateGeographyJson,
  type GeographyOverlayData,
} from './lib/geography'
import { DEFAULT_IMPACT_SETTINGS, type ImpactSettings } from './lib/impact'
import {
  applyDragToShock,
  buildPerturbation,
  buildZeroPerturbation,
  createEmptyShockConfig,
  getShockMask,
  moveShockCenter,
  summarizeShockChannels,
  type ShockConfig,
} from './lib/shock'

const HEALTH_POLL_INTERVAL_MS = 10_000
const BASE_PLAYBACK_STEP_MS = 600

export default function App() {
  // Server telemetry state
  const [health, setHealth] = useState<HealthState>({ kind: 'checking' })
  const [inferenceMs, setInferenceMs] = useState<number | null>(null)
  const [simulateError, setSimulateError] = useState<string | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulatingLabel, setSimulatingLabel] = useState('Running simulation…')

  // In-memory scenario & rollout state (never persisted to localStorage)
  const devPreviewEnabled = useMemo(() => isDevPreviewEnabled(), [])
  const [isSyntheticActive, setIsSyntheticActive] = useState<boolean>(false)
  const [scenario, setScenario] = useState<ScenarioData | null>(null)
  const [steps, setSteps] = useState<number>(10)
  const [primeRollout, setPrimeRollout] = useState<StateSequence | null>(null)
  const [primeSteps, setPrimeSteps] = useState<number | null>(null)
  const [branchRollout, setBranchRollout] = useState<StateSequence | null>(null)
  const [appliedShockMask, setAppliedShockMask] = useState<boolean[][]>(() => getShockMask(null))
  const [branchLabel, setBranchLabel] = useState<string>('No branch simulated yet')

  // Map & playback controls
  const [channel, setChannel] = useState<DisplayChannelId>('t2m')
  const [vorticityActive, setVorticityActive] = useState<boolean>(false)
  const [windFlowEnabled, setWindFlowEnabled] = useState<boolean>(false)
  const [stepIndex, setStepIndex] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>(1)
  const [compareMode, setCompareMode] = useState<CompareMode>('prime')
  const [smoothInterpolation, setSmoothInterpolation] = useState<boolean>(false)
  const [showWindArrows, setShowWindArrows] = useState<boolean>(false)
  const [showTpOutline, setShowTpOutline] = useState<boolean>(true)
  const [showShockOverlay, setShowShockOverlay] = useState<boolean>(true)

  // Optional in-memory 32x36 land_sea_mask geography overlay (never saved)
  const [geography, setGeography] = useState<GeographyOverlayData | null>(null)
  const [geographyError, setGeographyError] = useState<string | null>(null)

  // Shock configuration & visible impact settings
  const [shockConfig, setShockConfig] = useState<ShockConfig>(() => createEmptyShockConfig())
  const [hasTouchedShock, setHasTouchedShock] = useState<boolean>(false)
  const [impactSettings, setImpactSettings] = useState<ImpactSettings>(() => ({
    ...DEFAULT_IMPACT_SETTINGS,
  }))

  const simAbortRef = useRef<AbortController | null>(null)

  // Poll POST /health on load and every 10 seconds
  const pollHealth = useCallback(async () => {
    try {
      const res = await checkHealth()
      setHealth({
        kind: 'ready',
        modelName: res.model_name,
        lastCheckedAt: Date.now(),
      })
    } catch (err) {
      setHealth({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
        lastCheckedAt: Date.now(),
      })
    }
  }, [])

  useEffect(() => {
    void pollHealth()
    const id = window.setInterval(() => {
      void pollHealth()
    }, HEALTH_POLL_INTERVAL_MS)
    return () => window.clearInterval(id)
  }, [pollHealth])

  // Abort in-flight simulation on unmount
  useEffect(() => {
    return () => {
      simAbortRef.current?.abort()
    }
  }, [])

  const totalRolloutSteps = primeRollout?.length ?? branchRollout?.length ?? 0

  // Keep stepIndex in bounds when rollout length changes
  useEffect(() => {
    if (totalRolloutSteps === 0) {
      setStepIndex(0)
      setIsPlaying(false)
    } else if (stepIndex >= totalRolloutSteps) {
      setStepIndex(totalRolloutSteps - 1)
    }
  }, [totalRolloutSteps, stepIndex])

  // Smooth playback timer
  useEffect(() => {
    if (!isPlaying || totalRolloutSteps <= 1) return
    const intervalMs = Math.max(80, Math.round(BASE_PLAYBACK_STEP_MS / playbackSpeed))
    const id = window.setInterval(() => {
      setStepIndex((prev) => (prev + 1) % totalRolloutSteps)
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [isPlaying, totalRolloutSteps, playbackSpeed])

  // Shared scrubber keyboard shortcuts: Space = play/pause, ArrowLeft / ArrowRight = step
  useEffect(() => {
    if (!scenario || totalRolloutSteps <= 0) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const target = e.target as HTMLElement | null
      if (target) {
        if (target.isContentEditable) return
        if (target.dataset?.splitHandle === 'true') return
        const tag = target.tagName
        if (tag === 'TEXTAREA' || tag === 'SELECT') return
        if (tag === 'INPUT' && target.dataset?.dayScrubber !== 'true') return
      }

      if (e.code === 'Space' || e.key === ' ') {
        if (totalRolloutSteps > 1) {
          e.preventDefault()
          setIsPlaying((p) => !p)
        }
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setIsPlaying(false)
        setStepIndex((prev) => Math.max(0, prev - 1))
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        setIsPlaying(false)
        setStepIndex((prev) => Math.min(totalRolloutSteps - 1, prev + 1))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [scenario, totalRolloutSteps])

  // Run Timeline Prime (perturbation: null) immediately upon importing a scenario
  const simulatePrimeOnly = useCallback(
    async (targetScenario: ScenarioData, targetSteps: number) => {
      simAbortRef.current?.abort()
      const controller = new AbortController()
      simAbortRef.current = controller

      setIsSimulating(true)
      setSimulatingLabel('Simulating Timeline Prime…')
      setSimulateError(null)

      try {
        const primeRes = await runSimulation(
          {
            init_states: targetScenario.init_states,
            doy_sincos: targetScenario.doy_sincos,
            steps: targetSteps,
            perturbation: null,
          },
          controller.signal,
        )
        setPrimeRollout(primeRes.rollout)
        setPrimeSteps(targetSteps)
        setInferenceMs(primeRes.inference_ms)
        setStepIndex(0)
      } catch (err) {
        if (controller.signal.aborted) return
        setSimulateError(err instanceof Error ? err.message : String(err))
      } finally {
        if (simAbortRef.current === controller) {
          setIsSimulating(false)
        }
      }
    },
    [],
  )

  const handleImportScenario = useCallback(
    (imported: ScenarioData) => {
      setIsSyntheticActive(false)
      setScenario(imported)
      setPrimeRollout(null)
      setPrimeSteps(null)
      setBranchRollout(null)
      setAppliedShockMask(getShockMask(null))
      setBranchLabel('No branch simulated yet')
      setHasTouchedShock(false)
      setCompareMode('prime')
      setStepIndex(0)
      setIsPlaying(false)
      setSimulateError(null)
      void simulatePrimeOnly(imported, steps)
    },
    [simulatePrimeOnly, steps],
  )

  const handleResetScenario = useCallback(() => {
    simAbortRef.current?.abort()
    setIsSimulating(false)
    setIsPlaying(false)
    setIsSyntheticActive(false)
    setScenario(null)
    setPrimeRollout(null)
    setPrimeSteps(null)
    setBranchRollout(null)
    setAppliedShockMask(getShockMask(null))
    setBranchLabel('No branch simulated yet')
    setHasTouchedShock(false)
    setSimulateError(null)
    setStepIndex(0)
  }, [])

  const handleLoadSyntheticPreview = useCallback(() => {
    simAbortRef.current?.abort()
    setIsSimulating(false)
    setIsPlaying(false)
    setSimulateError(null)
    const synthetic = generateSyntheticScenario()
    const prime = generateSyntheticRollout(synthetic.init_states, steps, null)
    setIsSyntheticActive(true)
    setScenario(synthetic)
    setPrimeRollout(prime)
    setPrimeSteps(steps)
    setBranchRollout(null)
    setAppliedShockMask(getShockMask(null))
    setBranchLabel('No branch simulated yet')
    setHasTouchedShock(false)
    setCompareMode('prime')
    setStepIndex(0)
  }, [steps])

  const executeShockRun = useCallback(
    async (perturbation: State3D, label: string, configSnapshot: ShockConfig) => {
      if (!scenario) return
      if (isSyntheticActive) {
        setSimulateError(null)
        setIsPlaying(false)
        const needPrime = !primeRollout || primeSteps !== steps
        if (needPrime) {
          setPrimeRollout(generateSyntheticRollout(scenario.init_states, steps, null))
          setPrimeSteps(steps)
        }
        setBranchRollout(generateSyntheticRollout(scenario.init_states, steps, perturbation))
        setAppliedShockMask(getShockMask(perturbation, configSnapshot))
        setBranchLabel(label)
        return
      }
      simAbortRef.current?.abort()
      const controller = new AbortController()
      simAbortRef.current = controller

      setIsSimulating(true)
      setSimulateError(null)
      setIsPlaying(false)

      try {
        const needPrime = !primeRollout || primeSteps !== steps
        if (needPrime) {
          setSimulatingLabel('Simulating Timeline Prime (1/2)…')
          const primeRes = await runSimulation(
            {
              init_states: scenario.init_states,
              doy_sincos: scenario.doy_sincos,
              steps,
              perturbation: null,
            },
            controller.signal,
          )
          setPrimeRollout(primeRes.rollout)
          setPrimeSteps(steps)
          setInferenceMs(primeRes.inference_ms)
        }

        setSimulatingLabel(needPrime ? 'Simulating Branch (2/2)…' : 'Simulating Branch…')
        const branchRes = await runSimulation(
          {
            init_states: scenario.init_states,
            doy_sincos: scenario.doy_sincos,
            steps,
            perturbation,
          },
          controller.signal,
        )
        setBranchRollout(branchRes.rollout)
        setAppliedShockMask(getShockMask(perturbation, configSnapshot))
        setBranchLabel(label)
        setInferenceMs(branchRes.inference_ms)
      } catch (err) {
        if (controller.signal.aborted) return
        setSimulateError(err instanceof Error ? err.message : String(err))
      } finally {
        if (simAbortRef.current === controller) {
          setIsSimulating(false)
        }
      }
    },
    [scenario, isSyntheticActive, primeRollout, primeSteps, steps],
  )

  const handleRunShock = useCallback(() => {
    if (!scenario) return
    const lastInputDay = scenario.init_states[2]
    const perturbation = buildPerturbation(shockConfig, lastInputDay)
    const summary = shockConfig.presetName
      ? `${shockConfig.presetName} (${summarizeShockChannels(shockConfig)})`
      : summarizeShockChannels(shockConfig)
    setCompareMode((prev) => (prev === 'prime' ? 'branch' : prev))
    void executeShockRun(perturbation, summary, shockConfig)
  }, [scenario, shockConfig, executeShockRun])

  const handleApplyPresetAndRun = useCallback(
    (preset: ShockConfig) => {
      setShockConfig(preset)
      setHasTouchedShock(true)
      if (!scenario) return
      const lastInputDay = scenario.init_states[2]
      const perturbation = buildPerturbation(preset, lastInputDay)
      const summary = preset.presetName
        ? `${preset.presetName} (${summarizeShockChannels(preset)})`
        : summarizeShockChannels(preset)
      setCompareMode((prev) => (prev === 'prime' ? 'branch' : prev))
      void executeShockRun(perturbation, summary, preset)
    },
    [scenario, executeShockRun],
  )

  const handleRunZeroShock = useCallback(() => {
    if (!scenario) return
    const zeroPerturbation = buildZeroPerturbation()
    setCompareMode((prev) => (prev === 'prime' ? 'difference' : prev))
    void executeShockRun(
      zeroPerturbation,
      'Zero shock (all-zero [6][32][36] perturbation)',
      shockConfig,
    )
  }, [scenario, shockConfig, executeShockRun])

  const handleShockClick = useCallback((row: number, col: number) => {
    setHasTouchedShock(true)
    setShockConfig((prev) => moveShockCenter(prev, row, col))
  }, [])

  const handleShockDrag = useCallback(
    (startRow: number, startCol: number, endRow: number, endCol: number) => {
      setHasTouchedShock(true)
      setShockConfig((prev) => applyDragToShock(prev, startRow, startCol, endRow, endCol))
    },
    [],
  )

  const handleShockConfigChange = useCallback((next: ShockConfig) => {
    setHasTouchedShock(true)
    setShockConfig(next)
  }, [])

  const handleResetShockConfig = useCallback(() => {
    setHasTouchedShock(false)
    setShockConfig(createEmptyShockConfig())
  }, [])

  const handleImportGeographyFile = useCallback(async (file: File) => {
    setGeographyError(null)
    try {
      const text = await file.text()
      const res = parseAndValidateGeographyJson(text, file.name)
      if (!res.ok) {
        setGeographyError(res.error)
        return
      }
      setGeography(res.data)
    } catch (err) {
      setGeographyError(
        `Failed to read geography file "${file.name}": ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    }
  }, [])

  const handleClearGeography = useCallback(() => {
    setGeography(null)
    setGeographyError(null)
  }, [])

  const handleSelectPhysicalChannel = useCallback((nextChannel: DisplayChannelId) => {
    setVorticityActive(false)
    if (nextChannel === 'wind') {
      setWindFlowEnabled(false)
    }
    setChannel(nextChannel)
  }, [])

  const isShockSet = useMemo(() => {
    if (hasTouchedShock || shockConfig.presetName !== null) return true
    return Object.values(shockConfig.channels).some((ch) => ch.enabled && ch.value !== 0)
  }, [hasTouchedShock, shockConfig])

  return (
    <div className="flex min-h-screen flex-col bg-[#0D1210] text-[#ECE7DA]">
      <StatusBar
        health={health}
        inferenceMs={inferenceMs}
        steps={steps}
        currentStep={totalRolloutSteps > 0 ? stepIndex + 1 : undefined}
        onStepsChange={setSteps}
        hasScenario={scenario !== null}
        scenarioLabel={scenario?.sourceLabel}
        geographyLabel={geography?.sourceLabel ?? null}
        onImportGeographyFile={(file) => void handleImportGeographyFile(file)}
        onClearGeography={handleClearGeography}
        isSimulating={isSimulating}
        onRefreshHealth={() => void pollHealth()}
        onResetScenario={handleResetScenario}
      />

      {/* Flat full-width ember error strips */}
      {(health.kind === 'error' || simulateError || geographyError) && (
        <div className="flex w-full flex-col">
          {health.kind === 'error' && (
            <ErrorBanner
              title="Server unreachable (POST /health)"
              message={health.message}
              onRetry={() => void pollHealth()}
            />
          )}
          {simulateError && (
            <ErrorBanner
              title="Simulation failed (POST /simulate)"
              message={simulateError}
              onDismiss={() => setSimulateError(null)}
            />
          )}
          {geographyError && (
            <ErrorBanner
              title="Geography import failed"
              message={geographyError}
              onDismiss={() => setGeographyError(null)}
            />
          )}
        </div>
      )}

      {isSyntheticActive && (
        <div
          role="status"
          className="w-full border-b border-[#26312B] bg-[#D9593D] px-6 py-1.5 font-mono text-xs font-medium text-[#0D1210]"
        >
          <div className="mx-auto flex max-w-[1680px] items-center gap-2">
            SYNTHETIC - DEV ONLY
          </div>
        </div>
      )}

      {!scenario ? (
        <EmptyState
          steps={steps}
          onStepsChange={setSteps}
          onImport={handleImportScenario}
          showDevPreview={devPreviewEnabled}
          onLoadSyntheticPreview={handleLoadSyntheticPreview}
        />
      ) : (
        <main className="panel-enter flex w-full flex-1 flex-col">
          {/* Above-the-fold full-height Stage (~70%) + Right Instrument Rail (~30%) */}
          <div className="w-full border-b border-[#26312B]">
            <div className="mx-auto grid min-h-[calc(100vh-56px)] w-full max-w-[1680px] lg:grid-cols-[minmax(0,7fr)_minmax(300px,3fr)]">
              {/* Left ~70%: Single Map Stage + Sticky Bottom Film Timeline */}
              <div className="flex min-w-0 flex-col justify-start gap-2 px-6 pt-4 pb-4 lg:sticky lg:top-[56px] lg:self-start">
                <ComparisonView
                  primeRollout={primeRollout}
                  branchRollout={branchRollout}
                  appliedShockMask={appliedShockMask}
                  isShockSet={isShockSet}
                  channel={channel}
                  vorticityActive={vorticityActive}
                  onToggleVorticity={() => setVorticityActive((v) => !v)}
                  stepIndex={stepIndex}
                  compareMode={compareMode}
                  onCompareModeChange={setCompareMode}
                  smoothInterpolation={smoothInterpolation}
                  onToggleSmoothInterpolation={() => setSmoothInterpolation((v) => !v)}
                  windFlowEnabled={windFlowEnabled}
                  onToggleWindFlow={() => setWindFlowEnabled((v) => !v)}
                  geography={geography}
                  onImportGeographyFile={(file) => void handleImportGeographyFile(file)}
                  onClearGeography={handleClearGeography}
                  showWindArrows={showWindArrows}
                  onToggleWindArrows={() => setShowWindArrows((v) => !v)}
                  showTpOutline={showTpOutline}
                  onToggleTpOutline={() => setShowTpOutline((v) => !v)}
                  showShockOverlay={showShockOverlay}
                  onToggleShockOverlay={() => setShowShockOverlay((v) => !v)}
                  shockConfig={shockConfig}
                  onApplyPresetAndRun={handleApplyPresetAndRun}
                  onRunZeroShock={handleRunZeroShock}
                  branchLabel={branchLabel}
                  isSimulating={isSimulating}
                  simulatingLabel={simulatingLabel}
                  onShockClick={handleShockClick}
                  onShockDrag={handleShockDrag}
                />

                <MapControls
                  stepIndex={stepIndex}
                  totalSteps={totalRolloutSteps}
                  onStepIndexChange={(idx) => {
                    setIsPlaying(false)
                    setStepIndex(idx)
                  }}
                  isPlaying={isPlaying}
                  onTogglePlay={() => setIsPlaying((p) => !p)}
                  playbackSpeed={playbackSpeed}
                  onPlaybackSpeedChange={setPlaybackSpeed}
                />
              </div>

              {/* Right ~30% Rail: Channel selector, Shock panel, Presets */}
              <div className="flex flex-col gap-5 border-t border-[#26312B] px-6 py-5 lg:border-t-0 lg:border-l">
                <ChannelSelector
                  channel={channel}
                  onChannelChange={handleSelectPhysicalChannel}
                  vorticityActive={vorticityActive}
                  onToggleVorticity={() => setVorticityActive((v) => !v)}
                />

                <ShockPanel
                  config={shockConfig}
                  onChange={handleShockConfigChange}
                  onReset={handleResetShockConfig}
                  onRun={handleRunShock}
                  onRunZeroShock={handleRunZeroShock}
                  isSimulating={isSimulating}
                  hasPrime={primeRollout !== null && primeSteps === steps}
                />
              </div>
            </div>
          </div>

          {/* Below the fold: Spread chart, then Impact panel */}
          <div className="mx-auto flex w-full max-w-[1680px] flex-col gap-10 px-6">
            <SpreadChart
              primeRollout={primeRollout}
              branchRollout={branchRollout}
              shockMask={appliedShockMask}
              channel={channel}
              stepIndex={stepIndex}
            />

            <ImpactPanel
              primeRollout={primeRollout}
              branchRollout={branchRollout}
              stepIndex={stepIndex}
              settings={impactSettings}
              onSettingsChange={setImpactSettings}
              smoothInterpolation={smoothInterpolation}
            />
          </div>
        </main>
      )}
    </div>
  )
}
