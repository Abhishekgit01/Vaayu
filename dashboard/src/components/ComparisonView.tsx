import { useMemo, useRef, useState, type ChangeEvent } from 'react'
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import type { DisplayChannelId, StateSequence } from '../api/types'
import { paletteForChannel } from '../lib/colormaps'
import { GRID_COLS, GRID_ROWS, type Grid2D } from '../api/types'
import { computeWindSpeedGrid, getDisplayChannelMeta } from '../lib/constants'
import type { GeographyOverlayData } from '../lib/geography'
import { precomputeRolloutWorkspace } from '../lib/heatmap'
import {
  createWarmBayOfBengalPreset,
  createWeakerArabianSeaWindPreset,
  getShockMask,
  type ShockConfig,
} from '../lib/shock'
import {
  computeViewportForShockMask,
  DEFAULT_MAP_VIEWPORT,
  zoomViewportAtNorm,
  type MapViewport,
} from '../lib/viewport'
import { VORTICITY_META, type ViewFieldId } from '../lib/vorticity'
import { GridMap, type HoveredCell } from './GridMap'
import type { StageMode } from './MapControls'
import { RolloutSurface3D } from './RolloutSurface3D'
import { SplitCompareMap } from './SplitCompareMap'

interface ComparisonViewProps {
  primeRollout: StateSequence | null
  branchRollout: StateSequence | null
  appliedShockMask: boolean[][]
  isShockSet: boolean
  channel: DisplayChannelId
  vorticityActive: boolean
  onToggleVorticity: () => void
  stepIndex: number
  compareMode: StageMode
  onCompareModeChange: (mode: StageMode) => void
  smoothInterpolation: boolean
  onToggleSmoothInterpolation: () => void
  windFlowEnabled: boolean
  onToggleWindFlow: () => void
  geography: GeographyOverlayData | null
  onImportGeographyFile: (file: File) => void
  onClearGeography: () => void
  showWindArrows: boolean
  onToggleWindArrows: () => void
  showTpOutline: boolean
  onToggleTpOutline: () => void
  showShockOverlay: boolean
  onToggleShockOverlay: () => void
  shockConfig: ShockConfig
  onApplyPresetAndRun: (preset: ShockConfig) => void
  onRunZeroShock: () => void
  branchLabel: string
  isSimulating: boolean
  simulatingLabel: string
  onShockClick: (row: number, col: number) => void
  onShockDrag: (startRow: number, startCol: number, endRow: number, endCol: number) => void
}

interface NoShockStagePromptProps {
  modeTitle: string
  accentColor: string
  isSimulating: boolean
  onApplyPresetAndRun: (preset: ShockConfig) => void
  onRunZeroShock: () => void
}

function NoShockStagePrompt({
  modeTitle,
  accentColor,
  isSimulating,
  onApplyPresetAndRun,
  onRunZeroShock,
}: NoShockStagePromptProps) {
  return (
    <div className="panel-enter flex flex-col pt-2.5" style={{ borderTop: `1px solid ${accentColor}` }}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="font-serif-display text-xl italic leading-none text-[#ECE7DA]">
          {modeTitle}
        </h3>
        <span className="font-mono text-[11px] text-[#5A645D]">no shock simulated</span>
      </div>

      <div className="flex aspect-[36/32] w-full flex-col justify-center rounded-[2px] border border-[#26312B] bg-[#141B18] px-8 py-10 sm:px-14">
        <div className="max-w-lg">
          <p className="font-serif-display text-3xl leading-tight text-[#ECE7DA] sm:text-4xl">
            No atmospheric shock has run yet.
          </p>
          <p className="mt-2.5 text-sm leading-relaxed text-[#8C958D]">
            Configure a custom perturbation in the right rail, or run one of the presets below to
            branch from Timeline Prime.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              disabled={isSimulating}
              onClick={() => onApplyPresetAndRun(createWarmBayOfBengalPreset())}
              className="ui-transition rounded-[3px] border border-[#F0B84B] bg-[#F0B84B] px-3.5 py-2 text-xs font-medium text-[#0D1210] hover:bg-[#f4c668] disabled:opacity-40 cursor-pointer"
            >
              Warm Bay of Bengal (+3.0 K, +15% tcwv)
            </button>

            <button
              type="button"
              disabled={isSimulating}
              onClick={() => onApplyPresetAndRun(createWeakerArabianSeaWindPreset())}
              className="ui-transition rounded-[3px] border border-[#26312B] bg-[#1A231F] px-3.5 py-2 text-xs text-[#ECE7DA] hover:border-[#F0B84B] disabled:opacity-40 cursor-pointer"
            >
              Weaker Arabian Sea wind (−40% u10, v10)
            </button>

            <button
              type="button"
              disabled={isSimulating}
              onClick={onRunZeroShock}
              className="ui-transition ml-1 text-xs text-[#7FB5B0] underline-offset-4 hover:underline disabled:opacity-40 cursor-pointer"
            >
              Run zero shock
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ComparisonView({
  primeRollout,
  branchRollout,
  appliedShockMask,
  isShockSet,
  channel,
  vorticityActive,
  onToggleVorticity,
  stepIndex,
  compareMode,
  onCompareModeChange,
  smoothInterpolation,
  onToggleSmoothInterpolation,
  windFlowEnabled,
  onToggleWindFlow,
  geography,
  onImportGeographyFile,
  onClearGeography,
  showWindArrows,
  onToggleWindArrows,
  showTpOutline,
  onToggleTpOutline,
  showShockOverlay,
  onToggleShockOverlay,
  shockConfig,
  onApplyPresetAndRun,
  onRunZeroShock,
  branchLabel,
  isSimulating,
  simulatingLabel,
  onShockClick,
  onShockDrag,
}: ComparisonViewProps) {
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null)
  const [splitRatio, setSplitRatio] = useState<number>(0.5)
  const [dimensionMode, setDimensionMode] = useState<'2d' | '3d'>('2d')
  const [viewport, setViewport] = useState<MapViewport>(DEFAULT_MAP_VIEWPORT)
  const geoInputRef = useRef<HTMLInputElement | null>(null)

  const handleFocusShock = () => {
    const targetMask = isShockSet
      ? getShockMask(null, shockConfig, geography?.landSeaMask)
      : appliedShockMask
    setViewport(computeViewportForShockMask(targetMask))
  }

  const handleGeoInput = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onImportGeographyFile(file)
    }
    e.target.value = ''
  }

  // Recompute display grids, differences, scales, vectors, and contours ONLY when rollouts change
  const workspace = useMemo(
    () => precomputeRolloutWorkspace(primeRollout, branchRollout, appliedShockMask),
    [primeRollout, branchRollout, appliedShockMask],
  )

  const activeFieldId: ViewFieldId = vorticityActive ? 'vorticity' : channel
  const chMeta = vorticityActive ? VORTICITY_META : getDisplayChannelMeta(channel)
  const basePalette = vorticityActive ? 'diverging' : paletteForChannel(channel)
  const series = workspace.byChannel[activeFieldId]

  const primeState = primeRollout?.[stepIndex] ?? null
  const branchState = branchRollout?.[stepIndex] ?? null

  // Calculate Wind Speed statically on-the-fly for the active timeline step: Magnitude = Math.sqrt(u10**2 + v10**2)
  const onTheFlyWind = useMemo(() => {
    if (activeFieldId !== 'wind') return null
    const pGrid = primeState ? computeWindSpeedGrid(primeState) : null
    const bGrid = branchState ? computeWindSpeedGrid(branchState) : null
    let dGrid: Grid2D | null = null
    if (pGrid && bGrid) {
      dGrid = new Array(GRID_ROWS)
      for (let r = 0; r < GRID_ROWS; r++) {
        const row = new Array<number>(GRID_COLS)
        for (let c = 0; c < GRID_COLS; c++) {
          row[c] = bGrid[r][c] - pGrid[r][c]
        }
        dGrid[r] = row
      }
    }
    return { primeGrid: pGrid, branchGrid: bGrid, diffGrid: dGrid }
  }, [activeFieldId, primeState, branchState])

  const primeGrid =
    activeFieldId === 'wind' ? (onTheFlyWind?.primeGrid ?? null) : (series.primeGrids[stepIndex] ?? null)
  const branchGrid =
    activeFieldId === 'wind' ? (onTheFlyWind?.branchGrid ?? null) : (series.branchGrids[stepIndex] ?? null)
  const diffGrid =
    activeFieldId === 'wind' ? (onTheFlyWind?.diffGrid ?? null) : (series.diffGrids[stepIndex] ?? null)
  const stepMaxAbsDiff = series.stepMaxAbsDiff[stepIndex] ?? 0

  const primeWindArrows = workspace.primeWindArrows[stepIndex]
  const branchWindArrows = workspace.branchWindArrows[stepIndex]
  const diffWindArrows = workspace.diffWindArrows[stepIndex]
  const primeTpSegments = workspace.primeTpSegments[stepIndex]
  const branchTpSegments = workspace.branchTpSegments[stepIndex]

  const hasBranch = branchRollout !== null
  const hasNonZeroResponse = hasBranch && series.rolloutMaxAbsDiff > 0

  const stepDayNumber = stepIndex + 1
  const activeFrontContour =
    showShockOverlay && hasNonZeroResponse ? series.frontContoursByStep[stepIndex] : undefined
  const activeLocalityRing =
    showShockOverlay && hasNonZeroResponse ? workspace.localityRingSegments : undefined
  const activeRefRing =
    showShockOverlay && hasNonZeroResponse ? workspace.refRingsByStep[stepIndex] : undefined
  const showFootprintOutline = showShockOverlay && isShockSet

  const diffSummary = diffGrid
    ? stepMaxAbsDiff === 0
      ? `exact zero difference (0 ${chMeta.displayUnit})`
      : `max |Δ| ${stepMaxAbsDiff.toFixed(
          stepMaxAbsDiff < 0.1 ? 3 : chMeta.decimals,
        )} ${chMeta.displayUnit}`
    : 'run Branch or Zero shock to compute Branch − Prime'

  const fieldHeaderLabel = vorticityActive
    ? `${chMeta.label} [computed curl(u10, v10)] (${chMeta.displayUnit})`
    : chMeta.isComputed
      ? `${chMeta.label} [computed √(u10² + v10²)] (${chMeta.displayUnit})`
      : `${chMeta.label} (${chMeta.displayUnit})`

  const tpOutlineActive = !vorticityActive && channel === 'tp' && showTpOutline
  const windArrowsActive = !vorticityActive && channel === 'wind' && showWindArrows && !windFlowEnabled

  const landSeaMask = geography?.landSeaMask ?? null
  const coastlineSegments = geography?.coastlineSegments

  const compareGrids = useMemo(
    () => ({ prime: primeGrid, branch: branchGrid, diff: diffGrid }),
    [primeGrid, branchGrid, diffGrid],
  )

  return (
    <div className="relative flex flex-1 flex-col">
      {/* Stage mode switch bar: Prime | Branch | Difference | Split curtain */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#26312B] pb-2.5 text-xs">
        <div role="tablist" aria-label="Stage view mode" className="flex items-center gap-5">
          {(
            [
              { id: 'prime', label: 'Prime', underline: 'after:bg-[#7FB5B0]' },
              { id: 'branch', label: 'Branch', underline: 'after:bg-[#F0B84B]' },
              { id: 'difference', label: 'Difference', underline: 'after:bg-[#ECE7DA]' },
              { id: 'split', label: 'Split curtain', underline: 'after:bg-[#F0B84B]' },
            ] as const
          ).map((item) => {
            const active = compareMode === item.id
            return (
              <button
                key={item.id}
                role="tab"
                aria-selected={active}
                type="button"
                onClick={() => onCompareModeChange(item.id)}
                className={`ui-transition relative pb-1 cursor-pointer ${
                  active
                    ? `font-medium text-[#ECE7DA] after:absolute after:right-0 after:-bottom-[11px] after:left-0 after:h-[1px] ${item.underline}`
                    : 'text-[#8C958D] hover:text-[#ECE7DA]'
                }`}
              >
                {item.label}
              </button>
            )
          })}
        </div>

        <div className="flex flex-wrap items-center gap-3.5 text-xs text-[#8C958D]">
          {/* Wind flow toggle */}
          <button
            type="button"
            onClick={onToggleWindFlow}
            aria-pressed={windFlowEnabled}
            className={`ui-transition rounded-[2px] border px-2 py-0.5 cursor-pointer ${
              windFlowEnabled
                ? 'border-[#F0B84B] bg-[#1A231F] text-[#ECE7DA]'
                : 'border-[#26312B] bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
            }`}
          >
            Wind flow
          </button>

          {/* Rotation (relative vorticity) computed layer toggle */}
          <button
            type="button"
            onClick={onToggleVorticity}
            aria-pressed={vorticityActive}
            className={`ui-transition rounded-[2px] border px-2 py-0.5 cursor-pointer ${
              vorticityActive
                ? 'border-[#F0B84B] bg-[#1A231F] text-[#ECE7DA]'
                : 'border-[#26312B] bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
            }`}
          >
            Rotation (relative vorticity){' '}
            <span className="font-mono text-[10px] text-[#7FB5B0]">computed</span>
          </button>

          {/* In-memory GeoJSON / land_sea_mask importer */}
          <input
            ref={geoInputRef}
            type="file"
            onChange={handleGeoInput}
            className="hidden"
          />
          {geography ? (
            <span className="inline-flex items-center gap-1.5 rounded-[2px] border border-[#26312B] bg-[#141B18] px-2 py-0.5 text-[#ECE7DA]">
              <span>Coastline</span>
              <button
                type="button"
                onClick={onClearGeography}
                className="ui-transition font-mono text-[10px] text-[#7FB5B0] hover:underline cursor-pointer"
              >
                clear
              </button>
            </span>
          ) : (
            <button
              type="button"
              onClick={() => geoInputRef.current?.click()}
              className="ui-transition rounded-[2px] border border-[#26312B] bg-[#141B18] px-2 py-0.5 text-[#8C958D] hover:border-[#7FB5B0] hover:text-[#ECE7DA] cursor-pointer"
            >
              Import geography
            </button>
          )}

          {dimensionMode === '2d' && (
            <>
              {!vorticityActive && channel === 'wind' && compareMode !== 'difference' && !windFlowEnabled && (
                <button
                  type="button"
                  onClick={onToggleWindArrows}
                  aria-pressed={showWindArrows}
                  className={`ui-transition cursor-pointer ${
                    showWindArrows ? 'text-[#ECE7DA]' : 'text-[#5A645D] hover:text-[#8C958D]'
                  }`}
                >
                  Vectors {showWindArrows ? 'on' : 'off'}
                </button>
              )}
              {!vorticityActive && channel === 'tp' && compareMode !== 'difference' && (
                <button
                  type="button"
                  onClick={onToggleTpOutline}
                  aria-pressed={showTpOutline}
                  className={`ui-transition cursor-pointer ${
                    showTpOutline ? 'text-[#ECE7DA]' : 'text-[#5A645D] hover:text-[#8C958D]'
                  }`}
                >
                  &gt;10 mm/day {showTpOutline ? 'on' : 'off'}
                </button>
              )}
              <button
                type="button"
                onClick={onToggleShockOverlay}
                aria-pressed={showShockOverlay}
                className={`ui-transition cursor-pointer ${
                  showShockOverlay ? 'text-[#ECE7DA]' : 'text-[#5A645D] hover:text-[#8C958D]'
                }`}
              >
                Shock frame {showShockOverlay ? 'on' : 'off'}
              </button>
              <button
                type="button"
                onClick={onToggleSmoothInterpolation}
                aria-pressed={smoothInterpolation}
                className={`ui-transition cursor-pointer ${
                  smoothInterpolation ? 'text-[#ECE7DA]' : 'text-[#5A645D] hover:text-[#8C958D]'
                }`}
              >
                {smoothInterpolation ? 'Interpolated' : 'Crisp cells'}
              </button>
            </>
          )}

          {(dimensionMode === '2d' || compareMode === 'split') && (
            <div className="inline-flex items-center overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18] font-mono text-[11px]">
              <button
                type="button"
                disabled={viewport.zoom <= 1.001}
                onClick={() => setViewport(zoomViewportAtNorm(viewport, viewport.zoom / 1.4))}
                title="Zoom out"
                aria-label="Zoom out"
                className="ui-transition px-1.5 py-0.5 text-[#8C958D] hover:text-[#ECE7DA] disabled:opacity-35 cursor-pointer"
              >
                <ZoomOut className="h-3 w-3" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                onClick={() => setViewport(DEFAULT_MAP_VIEWPORT)}
                title="Reset zoom to 1×"
                className="tabular-nums ui-transition border-l border-[#26312B] px-1.5 py-0.5 text-[#ECE7DA] hover:text-[#7FB5B0] cursor-pointer"
              >
                {viewport.zoom.toFixed(1)}×
              </button>
              <button
                type="button"
                disabled={viewport.zoom >= 6}
                onClick={() => setViewport(zoomViewportAtNorm(viewport, viewport.zoom * 1.4))}
                title="Zoom in (scroll wheel on map; drag to pan)"
                aria-label="Zoom in"
                className="ui-transition border-l border-[#26312B] px-1.5 py-0.5 text-[#8C958D] hover:text-[#ECE7DA] disabled:opacity-35 cursor-pointer"
              >
                <ZoomIn className="h-3 w-3" strokeWidth={1.5} />
              </button>
              {(isShockSet || hasBranch) && (
                <button
                  type="button"
                  onClick={handleFocusShock}
                  title="Zoom to active shock footprint"
                  className="ui-transition inline-flex items-center gap-1 border-l border-[#26312B] px-2 py-0.5 text-[#7FB5B0] hover:text-[#ECE7DA] cursor-pointer"
                >
                  <RotateCcw className="h-2.5 w-2.5" strokeWidth={1.5} />
                  Focus shock
                </button>
              )}
            </div>
          )}

          {/* 2D / 3D toggle (for Prime, Branch, and Difference single-field views) */}
          {compareMode !== 'split' && (
            <div
              role="group"
              aria-label="Map projection dimension"
              className="inline-flex overflow-hidden rounded-[2px] border border-[#26312B] font-mono text-[11px]"
            >
              <button
                type="button"
                onClick={() => setDimensionMode('2d')}
                className={`ui-transition px-2 py-0.5 cursor-pointer ${
                  dimensionMode === '2d'
                    ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                    : 'bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
                }`}
              >
                2D
              </button>
              <button
                type="button"
                onClick={() => setDimensionMode('3d')}
                className={`ui-transition border-l border-[#26312B] px-2 py-0.5 cursor-pointer ${
                  dimensionMode === '3d'
                    ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                    : 'bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
                }`}
              >
                3D
              </button>
            </div>
          )}
        </div>
      </div>

      {isSimulating && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-[#0D1210]/70">
          <div className="rounded-[3px] border border-[#26312B] bg-[#141B18] px-5 py-3">
            <p className="text-xs font-medium text-[#ECE7DA]">{simulatingLabel}</p>
            <p className="mt-0.5 font-mono text-[11px] text-[#8C958D]">POST /simulate</p>
          </div>
        </div>
      )}

      {/* Single full-height Map Stage */}
      <div className="flex flex-col justify-start py-4">
        <div className="mx-auto w-full max-w-[min(100%,calc((100vh-228px)*1.125+42px))]">
          {compareMode === 'prime' &&
            (dimensionMode === '3d' ? (
              <RolloutSurface3D
                title="Timeline Prime"
                accentRule="rain"
                subtitle={`${fieldHeaderLabel} · perturbation: null`}
                grid={primeGrid}
                physicalState={primeState}
                compareGrids={compareGrids}
                channel={activeFieldId}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Timeline Prime…"
              />
            ) : (
              <GridMap
                title="Timeline Prime"
                badgeVariant="prime"
                subtitle={`${fieldHeaderLabel} · perturbation: null`}
                grid={primeGrid}
                physicalState={primeState}
                compareGrids={compareGrids}
                channel={activeFieldId}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                smoothInterpolation={smoothInterpolation}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                showWindArrows={windArrowsActive}
                windArrows={primeWindArrows}
                showTpThresholdOutline={tpOutlineActive}
                tpSegments={primeTpSegments}
                shockConfig={shockConfig}
                showShockOverlay={showFootprintOutline}
                viewport={viewport}
                onViewportChange={setViewport}
                allowShockPlacement
                onShockClick={onShockClick}
                onShockDrag={onShockDrag}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Timeline Prime…"
              />
            ))}

          {compareMode === 'branch' &&
            (!hasBranch ? (
              <NoShockStagePrompt
                modeTitle="Branch"
                accentColor="#F0B84B"
                isSimulating={isSimulating}
                onApplyPresetAndRun={onApplyPresetAndRun}
                onRunZeroShock={onRunZeroShock}
              />
            ) : dimensionMode === '3d' ? (
              <RolloutSurface3D
                title="Branch"
                accentRule="lightning"
                subtitle={`${fieldHeaderLabel} · ${branchLabel}`}
                grid={branchGrid}
                physicalState={branchState}
                compareGrids={compareGrids}
                channel={activeFieldId}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Branch…"
              />
            ) : (
              <GridMap
                title="Branch"
                badgeVariant="branch"
                subtitle={`${fieldHeaderLabel} · ${branchLabel}`}
                grid={branchGrid}
                physicalState={branchState}
                compareGrids={compareGrids}
                channel={activeFieldId}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                smoothInterpolation={smoothInterpolation}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                showWindArrows={windArrowsActive}
                windArrows={branchWindArrows}
                showTpThresholdOutline={tpOutlineActive}
                tpSegments={branchTpSegments}
                shockConfig={shockConfig}
                showShockOverlay={showFootprintOutline}
                frontContourSegments={activeFrontContour}
                localityRingSegments={activeLocalityRing}
                refRingSegments={activeRefRing}
                stepDayNumber={stepDayNumber}
                viewport={viewport}
                onViewportChange={setViewport}
                allowShockPlacement
                onShockClick={onShockClick}
                onShockDrag={onShockDrag}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Branch…"
              />
            ))}

          {compareMode === 'difference' &&
            (!hasBranch ? (
              <NoShockStagePrompt
                modeTitle="Difference (Branch − Prime)"
                accentColor="#26312B"
                isSimulating={isSimulating}
                onApplyPresetAndRun={onApplyPresetAndRun}
                onRunZeroShock={onRunZeroShock}
              />
            ) : dimensionMode === '3d' ? (
              <RolloutSurface3D
                title="Difference (Branch − Prime)"
                accentRule="hairline"
                subtitle={`${vorticityActive ? `${chMeta.label} [computed]` : chMeta.label} · ${diffSummary}`}
                grid={diffGrid}
                channel={activeFieldId}
                palette="diverging"
                scale={series.diffScale}
                unit={chMeta.displayUnit}
                decimals={
                  series.rolloutMaxAbsDiff > 0 && series.rolloutMaxAbsDiff < 0.1
                    ? 3
                    : chMeta.decimals
                }
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                diffWindArrows={diffWindArrows}
                isDifference
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Difference…"
              />
            ) : (
              <GridMap
                title="Difference (Branch − Prime)"
                badgeVariant="diff"
                subtitle={`${vorticityActive ? `${chMeta.label} [computed]` : chMeta.label} · ${diffSummary}`}
                grid={diffGrid}
                channel={activeFieldId}
                palette="diverging"
                scale={series.diffScale}
                unit={chMeta.displayUnit}
                decimals={
                  series.rolloutMaxAbsDiff > 0 && series.rolloutMaxAbsDiff < 0.1
                    ? 3
                    : chMeta.decimals
                }
                smoothInterpolation={smoothInterpolation}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                isDifference
                windFlowEnabled={windFlowEnabled}
                diffWindArrows={diffWindArrows}
                shockConfig={shockConfig}
                showShockOverlay={showFootprintOutline && series.rolloutMaxAbsDiff > 0}
                frontContourSegments={activeFrontContour}
                localityRingSegments={activeLocalityRing}
                refRingSegments={activeRefRing}
                stepDayNumber={stepDayNumber}
                viewport={viewport}
                onViewportChange={setViewport}
                allowShockPlacement
                onShockClick={onShockClick}
                onShockDrag={onShockDrag}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Simulating Difference…"
              />
            ))}

          {compareMode === 'split' && (
            <div className="relative">
              <SplitCompareMap
                primeGrid={primeGrid}
                branchGrid={branchGrid}
                diffGrid={diffGrid}
                primeState={primeState}
                branchState={branchState}
                channel={activeFieldId}
                channelLabel={vorticityActive ? `${chMeta.label} [computed]` : chMeta.label}
                branchLabel={branchLabel}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                smoothInterpolation={smoothInterpolation}
                landSeaMask={landSeaMask}
                coastlineSegments={coastlineSegments}
                windFlowEnabled={windFlowEnabled}
                splitRatio={splitRatio}
                onSplitRatioChange={setSplitRatio}
                showWindArrows={windArrowsActive}
                primeWindArrows={primeWindArrows}
                branchWindArrows={branchWindArrows}
                showTpThresholdOutline={tpOutlineActive}
                primeTpSegments={primeTpSegments}
                branchTpSegments={branchTpSegments}
                shockConfig={shockConfig}
                showShockOverlay={showFootprintOutline}
                frontContourSegments={activeFrontContour}
                localityRingSegments={activeLocalityRing}
                refRingSegments={activeRefRing}
                stepDayNumber={stepDayNumber}
                viewport={viewport}
                onViewportChange={setViewport}
                onShockClick={onShockClick}
                onShockDrag={onShockDrag}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
              />
              {!hasBranch && (
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-[#26312B] pt-2.5 text-xs text-[#8C958D]">
                  <span>Right curtain awaits a shock rollout:</span>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      disabled={isSimulating}
                      onClick={() => onApplyPresetAndRun(createWarmBayOfBengalPreset())}
                      className="ui-transition rounded-[2px] border border-[#F0B84B] bg-[#F0B84B] px-2.5 py-1 text-xs font-medium text-[#0D1210] hover:bg-[#f4c668] cursor-pointer"
                    >
                      Warm Bay of Bengal
                    </button>
                    <button
                      type="button"
                      disabled={isSimulating}
                      onClick={() => onApplyPresetAndRun(createWeakerArabianSeaWindPreset())}
                      className="ui-transition rounded-[2px] border border-[#26312B] bg-[#141B18] px-2.5 py-1 text-xs text-[#ECE7DA] hover:border-[#F0B84B] cursor-pointer"
                    >
                      Weaker Arabian Sea wind
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
