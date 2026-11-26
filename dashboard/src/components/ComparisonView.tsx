import { useMemo, useState } from 'react'
import type { DisplayChannelId, StateSequence } from '../api/types'
import { paletteForChannel } from '../lib/colormaps'
import { getDisplayChannelMeta } from '../lib/constants'
import { precomputeRolloutWorkspace } from '../lib/heatmap'
import type { ShockConfig } from '../lib/shock'
import { GridMap, type HoveredCell } from './GridMap'
import type { CompareMode } from './MapControls'
import { RolloutSurface3D } from './RolloutSurface3D'
import { SplitCompareMap } from './SplitCompareMap'

interface ComparisonViewProps {
  primeRollout: StateSequence | null
  branchRollout: StateSequence | null
  channel: DisplayChannelId
  stepIndex: number
  compareMode: CompareMode
  onCompareModeChange: (mode: CompareMode) => void
  smoothInterpolation: boolean
  onToggleSmoothInterpolation: () => void
  showWindArrows: boolean
  onToggleWindArrows: () => void
  showTpOutline: boolean
  onToggleTpOutline: () => void
  showShockOverlay: boolean
  onToggleShockOverlay: () => void
  shockConfig: ShockConfig
  branchLabel: string
  isSimulating: boolean
  simulatingLabel: string
  onShockClick: (row: number, col: number) => void
  onShockDrag: (startRow: number, startCol: number, endRow: number, endCol: number) => void
}

export function ComparisonView({
  primeRollout,
  branchRollout,
  channel,
  stepIndex,
  compareMode,
  onCompareModeChange,
  smoothInterpolation,
  onToggleSmoothInterpolation,
  showWindArrows,
  onToggleWindArrows,
  showTpOutline,
  onToggleTpOutline,
  showShockOverlay,
  onToggleShockOverlay,
  shockConfig,
  branchLabel,
  isSimulating,
  simulatingLabel,
  onShockClick,
  onShockDrag,
}: ComparisonViewProps) {
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null)
  const [splitRatio, setSplitRatio] = useState<number>(0.5)
  const [dimensionMode, setDimensionMode] = useState<'2d' | '3d'>('2d')

  // Recompute display grids, differences, scales, vectors, and contours ONLY when rollouts change
  const workspace = useMemo(
    () => precomputeRolloutWorkspace(primeRollout, branchRollout),
    [primeRollout, branchRollout],
  )

  const chMeta = getDisplayChannelMeta(channel)
  const basePalette = paletteForChannel(channel)
  const series = workspace.byChannel[channel]

  const primeState = primeRollout?.[stepIndex] ?? null
  const branchState = branchRollout?.[stepIndex] ?? null

  const primeGrid = series.primeGrids[stepIndex] ?? null
  const branchGrid = series.branchGrids[stepIndex] ?? null
  const diffGrid = series.diffGrids[stepIndex] ?? null
  const stepMaxAbsDiff = series.stepMaxAbsDiff[stepIndex] ?? 0

  const primeWindArrows = workspace.primeWindArrows[stepIndex]
  const branchWindArrows = workspace.branchWindArrows[stepIndex]
  const primeTpSegments = workspace.primeTpSegments[stepIndex]
  const branchTpSegments = workspace.branchTpSegments[stepIndex]

  const diffSummary = diffGrid
    ? stepMaxAbsDiff === 0
      ? `Exact zero difference (0 ${chMeta.displayUnit})`
      : `max |Δ| ${stepMaxAbsDiff.toFixed(
          stepMaxAbsDiff < 0.1 ? 3 : chMeta.decimals,
        )} ${chMeta.displayUnit}`
    : 'Run Branch or Zero shock to compute Branch − Prime'

  const tpOutlineActive = channel === 'tp' && showTpOutline
  const windArrowsActive = channel === 'wind' && showWindArrows

  const compareGrids = useMemo(
    () => ({ prime: primeGrid, branch: branchGrid, diff: diffGrid }),
    [primeGrid, branchGrid, diffGrid],
  )

  return (
    <div className="relative flex flex-col gap-5">
      {/* Slim instrument strip above the map */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#26312B] pb-2.5 text-xs">
        <div className="flex items-center gap-5">
          {(
            [
              { id: 'side-by-side', label: 'Multiverse' },
              { id: 'split', label: 'Split curtain' },
              { id: 'difference', label: 'Difference only' },
            ] as const
          ).map((item) => {
            const active = compareMode === item.id
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onCompareModeChange(item.id)}
                className={`ui-transition relative pb-1 cursor-pointer ${
                  active
                    ? 'font-medium text-[#ECE7DA] after:absolute after:right-0 after:-bottom-[11px] after:left-0 after:h-[1px] after:bg-[#F0B84B]'
                    : 'text-[#8C958D] hover:text-[#ECE7DA]'
                }`}
              >
                {item.label}
              </button>
            )
          })}
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs text-[#8C958D]">
          {dimensionMode === '2d' && (
            <>
              {channel === 'wind' && (
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
              {channel === 'tp' && (
                <button
                  type="button"
                  onClick={onToggleTpOutline}
                  aria-pressed={showTpOutline}
                  className={`ui-transition cursor-pointer ${
                    showTpOutline ? 'text-[#ECE7DA]' : 'text-[#5A645D] hover:text-[#8C958D]'
                  }`}
                >
                  &gt;10 mm/day contour {showTpOutline ? 'on' : 'off'}
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

          {/* 2D / 3D toggle */}
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

      {dimensionMode === '3d' ? (
        <div className="flex flex-col gap-6">
          {compareMode !== 'difference' && (
            <div className="grid gap-6 md:grid-cols-2">
              <RolloutSurface3D
                title="Timeline Prime"
                accentRule="rain"
                subtitle={`${chMeta.label} (${chMeta.displayUnit})`}
                grid={primeGrid}
                physicalState={primeState}
                compareGrids={compareGrids}
                channel={channel}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Awaiting Timeline Prime rollout"
              />
              <RolloutSurface3D
                title="Branch"
                accentRule="lightning"
                subtitle={branchLabel}
                grid={branchGrid}
                physicalState={branchState}
                compareGrids={compareGrids}
                channel={channel}
                palette={basePalette}
                scale={series.sharedScale}
                unit={chMeta.displayUnit}
                decimals={chMeta.decimals}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Run Branch or Zero shock to extrude Branch"
              />
            </div>
          )}
          {(compareMode === 'side-by-side' || compareMode === 'difference') && (
            <RolloutSurface3D
              title="Difference (Branch − Prime)"
              accentRule="hairline"
              subtitle={diffSummary}
              grid={diffGrid}
              channel={channel}
              palette="diverging"
              scale={series.diffScale}
              unit={chMeta.displayUnit}
              decimals={
                series.rolloutMaxAbsDiff > 0 && series.rolloutMaxAbsDiff < 0.1 ? 3 : chMeta.decimals
              }
              isDifference
              hoveredCell={hoveredCell}
              onHoverCell={setHoveredCell}
              emptyMessage="Run Branch or Zero shock to extrude Difference"
            />
          )}
        </div>
      ) : (
        <>
          {/* Multiverse view: Prime & Branch side by side, Difference below spanning full width */}
          {compareMode === 'side-by-side' && (
            <div className="flex flex-col gap-7">
              <div className="grid gap-6 md:grid-cols-2">
                <GridMap
                  title="Timeline Prime"
                  badgeVariant="prime"
                  subtitle="perturbation: null"
                  grid={primeGrid}
                  physicalState={primeState}
                  compareGrids={compareGrids}
                  channel={channel}
                  palette={basePalette}
                  scale={series.sharedScale}
                  unit={chMeta.displayUnit}
                  decimals={chMeta.decimals}
                  smoothInterpolation={smoothInterpolation}
                  showWindArrows={windArrowsActive}
                  windArrows={primeWindArrows}
                  showTpThresholdOutline={tpOutlineActive}
                  tpSegments={primeTpSegments}
                  shockConfig={shockConfig}
                  showShockOverlay={showShockOverlay}
                  allowShockPlacement
                  onShockClick={onShockClick}
                  onShockDrag={onShockDrag}
                  hoveredCell={hoveredCell}
                  onHoverCell={setHoveredCell}
                  emptyMessage="Click Run to simulate Timeline Prime"
                />

                <GridMap
                  title="Branch"
                  badgeVariant="branch"
                  subtitle={branchLabel}
                  grid={branchGrid}
                  physicalState={branchState}
                  compareGrids={compareGrids}
                  channel={channel}
                  palette={basePalette}
                  scale={series.sharedScale}
                  unit={chMeta.displayUnit}
                  decimals={chMeta.decimals}
                  smoothInterpolation={smoothInterpolation}
                  showWindArrows={windArrowsActive}
                  windArrows={branchWindArrows}
                  showTpThresholdOutline={tpOutlineActive}
                  tpSegments={branchTpSegments}
                  shockConfig={shockConfig}
                  showShockOverlay={showShockOverlay}
                  allowShockPlacement
                  onShockClick={onShockClick}
                  onShockDrag={onShockDrag}
                  hoveredCell={hoveredCell}
                  onHoverCell={setHoveredCell}
                  emptyMessage="Configure shock and click Run or Zero shock"
                />
              </div>

              {/* Full-width Difference map beneath Prime & Branch */}
              <GridMap
                title="Difference (Branch − Prime)"
                badgeVariant="diff"
                subtitle={diffSummary}
                grid={diffGrid}
                channel={channel}
                palette="diverging"
                scale={series.diffScale}
                unit={chMeta.displayUnit}
                decimals={
                  series.rolloutMaxAbsDiff > 0 && series.rolloutMaxAbsDiff < 0.1
                    ? 3
                    : chMeta.decimals
                }
                smoothInterpolation={smoothInterpolation}
                isDifference
                shockConfig={shockConfig}
                showShockOverlay={showShockOverlay}
                allowShockPlacement
                onShockClick={onShockClick}
                onShockDrag={onShockDrag}
                hoveredCell={hoveredCell}
                onHoverCell={setHoveredCell}
                emptyMessage="Run Branch or Zero shock to inspect bone-centered Difference"
              />
            </div>
          )}

          {compareMode === 'split' && (
            <SplitCompareMap
              primeGrid={primeGrid}
              branchGrid={branchGrid}
              diffGrid={diffGrid}
              primeState={primeState}
              branchState={branchState}
              channel={channel}
              channelLabel={chMeta.label}
              branchLabel={branchLabel}
              palette={basePalette}
              scale={series.sharedScale}
              unit={chMeta.displayUnit}
              decimals={chMeta.decimals}
              smoothInterpolation={smoothInterpolation}
              splitRatio={splitRatio}
              onSplitRatioChange={setSplitRatio}
              showWindArrows={windArrowsActive}
              primeWindArrows={primeWindArrows}
              branchWindArrows={branchWindArrows}
              showTpThresholdOutline={tpOutlineActive}
              primeTpSegments={primeTpSegments}
              branchTpSegments={branchTpSegments}
              shockConfig={shockConfig}
              showShockOverlay={showShockOverlay}
              onShockClick={onShockClick}
              onShockDrag={onShockDrag}
              hoveredCell={hoveredCell}
              onHoverCell={setHoveredCell}
            />
          )}

          {compareMode === 'difference' && (
            <GridMap
              title="Difference (Branch − Prime)"
              badgeVariant="diff"
              subtitle={diffSummary}
              grid={diffGrid}
              channel={channel}
              palette="diverging"
              scale={series.diffScale}
              unit={chMeta.displayUnit}
              decimals={
                series.rolloutMaxAbsDiff > 0 && series.rolloutMaxAbsDiff < 0.1 ? 3 : chMeta.decimals
              }
              smoothInterpolation={smoothInterpolation}
              isDifference
              shockConfig={shockConfig}
              showShockOverlay={showShockOverlay}
              allowShockPlacement
              onShockClick={onShockClick}
              onShockDrag={onShockDrag}
              hoveredCell={hoveredCell}
              onHoverCell={setHoveredCell}
              emptyMessage="Run Branch or Zero shock to inspect bone-centered Difference"
            />
          )}
        </>
      )}
    </div>
  )
}
