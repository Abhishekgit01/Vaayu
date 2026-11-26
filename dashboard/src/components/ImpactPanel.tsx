import { useMemo, useState } from 'react'
import type { StateSequence } from '../api/types'
import { computeScale } from '../lib/colormaps'
import {
  compareHeatwaveImpact,
  computeRowCellAreaKm2,
  DEFAULT_IMPACT_SETTINGS,
  TOTAL_GRID_AREA_KM2,
  TOTAL_GRID_CELLS,
  type ImpactSettings,
} from '../lib/impact'
import { GridMap, type HoveredCell } from './GridMap'

interface ImpactPanelProps {
  primeRollout: StateSequence | null
  branchRollout: StateSequence | null
  settings: ImpactSettings
  onSettingsChange: (next: ImpactSettings) => void
  smoothInterpolation?: boolean
}

function formatAreaHeadline(km2: number): string {
  if (km2 === 0) return '0 km²'
  if (Math.abs(km2) >= 1_000_000) {
    return `${(km2 / 1_000_000).toFixed(2)}M km²`
  }
  return `${Math.round(km2).toLocaleString()} km²`
}

function formatSignedArea(deltaKm2: number): string {
  const sign = deltaKm2 > 0 ? '+' : ''
  if (deltaKm2 === 0) return '0 km²'
  if (Math.abs(deltaKm2) >= 1_000_000) {
    return `${sign}${(deltaKm2 / 1_000_000).toFixed(2)}M km²`
  }
  return `${sign}${Math.round(deltaKm2).toLocaleString()} km²`
}

export function ImpactPanel({
  primeRollout,
  branchRollout,
  settings,
  onSettingsChange,
  smoothInterpolation = false,
}: ImpactPanelProps) {
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null)
  const [showDiffMap, setShowDiffMap] = useState<boolean>(false)

  const impact = useMemo(() => {
    if (!primeRollout) return null
    return compareHeatwaveImpact(primeRollout, branchRollout, settings)
  }, [primeRollout, branchRollout, settings])

  const totalSteps = primeRollout?.length ?? 0

  const riskScale = useMemo(
    () => ({ min: 0, max: Math.max(1, totalSteps), diverging: false }),
    [totalSteps],
  )

  const diffScale = useMemo(() => {
    if (!impact?.diffGrid) return { min: -1, max: 1, diverging: true }
    return computeScale([impact.diffGrid], true)
  }, [impact])

  const compareRiskGrids = useMemo(
    () => ({
      prime: impact?.prime.daysGrid ?? null,
      branch: impact?.branch?.daysGrid ?? null,
      diff: impact?.diffGrid ?? null,
    }),
    [impact],
  )

  const handleThresholdChange = (raw: string) => {
    const val = Number.parseFloat(raw)
    if (Number.isFinite(val)) {
      onSettingsChange({
        ...settings,
        heatwaveThresholdC: Math.max(-40, Math.min(60, val)),
      })
    }
  }

  const hoveredCellAreaKm2 =
    hoveredCell !== null ? Math.round(computeRowCellAreaKm2(hoveredCell.row)) : null

  const rolloutRangeNote = useMemo(() => {
    if (!impact) return null
    const minC =
      impact.branch && Number.isFinite(impact.branch.rolloutMinTempC)
        ? Math.min(impact.prime.rolloutMinTempC, impact.branch.rolloutMinTempC)
        : impact.prime.rolloutMinTempC
    const maxC =
      impact.branch && Number.isFinite(impact.branch.rolloutMaxTempC)
        ? Math.max(impact.prime.rolloutMaxTempC, impact.branch.rolloutMaxTempC)
        : impact.prime.rolloutMaxTempC
    if (!Number.isFinite(minC) || !Number.isFinite(maxC)) return null
    return `rollout t2m ${minC.toFixed(1)}°–${maxC.toFixed(1)}°C`
  }, [impact])

  const activeSummary = impact?.branch ?? impact?.prime ?? null
  const activeTimelineName = impact?.branch ? 'Branch' : 'Timeline Prime'

  const deltaCells = impact?.deltaCellsAtRisk ?? null
  const deltaAreaKm2 = impact?.deltaAreaAtRiskKm2 ?? null
  const deltaCellDays = impact?.deltaCellDays ?? null

  return (
    <section className="border-t border-[#26312B] pt-8 pb-12">
      {/* Header & visible settings object */}
      <div className="flex flex-wrap items-baseline justify-between gap-6">
        <div>
          <h2 className="font-serif-display text-3xl leading-none text-[#ECE7DA]">
            Heatwave risk
          </h2>
          <p className="mt-1.5 text-xs text-[#8C958D]">
            Cells and spherical surface area exceeding the threshold on at least one rollout day
            {rolloutRangeNote ? ` (${rolloutRangeNote})` : ''}.
            {hoveredCellAreaKm2 !== null && hoveredCell && (
              <span className="ml-2 font-mono text-[#ECE7DA]">
                cell [{hoveredCell.row},{hoveredCell.col}] = {hoveredCellAreaKm2.toLocaleString()} km²
              </span>
            )}
          </p>
        </div>

        {/* Visible settings object */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-[#8C958D]">
          <code className="rounded-[2px] border border-[#26312B] bg-[#141B18] px-2.5 py-1 font-mono text-xs text-[#ECE7DA] tabular-nums">
            {`settings = { heatwaveThresholdC: ${settings.heatwaveThresholdC.toFixed(1)} }`}
          </code>

          <label className="inline-flex items-center gap-1.5 font-mono">
            <input
              id="heatwave-threshold-input"
              type="number"
              step={0.5}
              min={-40}
              max={60}
              value={settings.heatwaveThresholdC}
              onChange={(e) => handleThresholdChange(e.target.value)}
              aria-label="Heatwave threshold in degrees Celsius"
              className="tabular-nums w-16 rounded-[2px] border border-[#26312B] bg-[#141B18] px-2 py-1 text-right font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
            />
            <span>°C</span>
          </label>

          <input
            type="range"
            min={15}
            max={48}
            step={0.5}
            value={Math.max(15, Math.min(48, settings.heatwaveThresholdC))}
            onChange={(e) => handleThresholdChange(e.target.value)}
            aria-label="Heatwave threshold slider"
            className="instrument-range w-28"
          />

          {settings.heatwaveThresholdC !== DEFAULT_IMPACT_SETTINGS.heatwaveThresholdC && (
            <button
              type="button"
              onClick={() => onSettingsChange({ ...DEFAULT_IMPACT_SETTINGS })}
              className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
            >
              Reset 35°C
            </button>
          )}

          <button
            type="button"
            onClick={() => setShowDiffMap((v) => !v)}
            aria-pressed={showDiffMap}
            className="ui-transition border-l border-[#26312B] pl-3 text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
          >
            {showDiffMap ? 'Hide Δ risk map' : 'Show Δ risk map'}
          </button>
        </div>
      </div>

      {/* Editorial Headline Row: Giant 96-140px Instrument Serif number + Prime vs Branch delta beside it */}
      <div className="mt-8 grid items-end gap-8 border-b border-[#26312B] pb-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <div>
          <p className="text-xs text-[#8C958D]">
            {activeTimelineName} cells at risk (&gt; {settings.heatwaveThresholdC.toFixed(1)}°C)
          </p>
          <p
            className="font-serif-display tabular-nums mt-2 leading-[0.86] tracking-tight text-[#ECE7DA]"
            style={{ fontSize: 'clamp(6rem, 9.5vw, 8.5rem)' }}
          >
            {activeSummary ? activeSummary.cellsAtRisk.toLocaleString() : '—'}
          </p>
          <p className="mt-3 font-mono text-sm text-[#ECE7DA] tabular-nums">
            {activeSummary ? (
              <>
                <span>{Math.round(activeSummary.areaAtRiskKm2).toLocaleString()} km² at risk</span>
                <span className="ml-2 text-xs text-[#8C958D]">
                  ({activeSummary.areaAtRiskPct.toFixed(1)}% of{' '}
                  {(TOTAL_GRID_AREA_KM2 / 1_000_000).toFixed(2)}M km² ·{' '}
                  {activeSummary.cellsAtRiskPct.toFixed(1)}% of {TOTAL_GRID_CELLS.toLocaleString()}{' '}
                  cells)
                </span>
              </>
            ) : (
              <span className="text-xs text-[#5A645D]">Awaiting rollout data</span>
            )}
          </p>
        </div>

        {/* Prime vs Branch comparison & delta beside the giant figure */}
        <div className="grid gap-6 border-t border-[#26312B] pt-6 sm:grid-cols-3 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
          {/* Timeline Prime */}
          <div className="border-t border-[#7FB5B0] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#7FB5B0]">
              Timeline Prime
            </p>
            <p className="mt-2 font-mono text-2xl font-medium text-[#ECE7DA] tabular-nums">
              {impact ? `${impact.prime.cellsAtRisk.toLocaleString()} cells` : '—'}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {impact ? formatAreaHeadline(impact.prime.areaAtRiskKm2) : '—'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {impact
                ? `${impact.prime.totalCellDays.toLocaleString()} cell-days · peak ${impact.prime.maxDaysInCell}d`
                : 'perturbation: null'}
            </p>
          </div>

          {/* Branch */}
          <div className="border-t border-[#F0B84B] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#F0B84B]">
              Branch
            </p>
            <p className="mt-2 font-mono text-2xl font-medium text-[#ECE7DA] tabular-nums">
              {impact?.branch ? `${impact.branch.cellsAtRisk.toLocaleString()} cells` : '—'}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {impact?.branch ? formatAreaHeadline(impact.branch.areaAtRiskKm2) : '—'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {impact?.branch
                ? `${impact.branch.totalCellDays.toLocaleString()} cell-days · peak ${impact.branch.maxDaysInCell}d`
                : 'run Branch to compare'}
            </p>
          </div>

          {/* Branch - Prime Delta */}
          <div className="border-t border-[#26312B] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#ECE7DA]">
              Branch − Prime
            </p>
            <p
              className={`mt-2 font-mono text-2xl font-medium tabular-nums ${
                deltaCells === null
                  ? 'text-[#5A645D]'
                  : deltaCells > 0
                    ? 'text-[#D9593D]'
                    : deltaCells < 0
                      ? 'text-[#7FB5B0]'
                      : 'text-[#ECE7DA]'
              }`}
            >
              {deltaCells === null
                ? '—'
                : `${deltaCells > 0 ? '+' : ''}${deltaCells.toLocaleString()} cells`}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {deltaAreaKm2 !== null ? formatSignedArea(deltaAreaKm2) : '—'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {deltaCellDays !== null
                ? `${deltaCellDays > 0 ? '+' : ''}${deltaCellDays.toLocaleString()} cell-days`
                : `${totalSteps}-day horizon`}
            </p>
          </div>
        </div>
      </div>

      {/* Side-by-Side Heatwave Risk Maps beneath the headline */}
      <div className="mt-7 grid gap-6 md:grid-cols-2">
        <GridMap
          title="Timeline Prime"
          badgeVariant="prime"
          subtitle={`days > ${settings.heatwaveThresholdC.toFixed(1)}°C per cell`}
          grid={impact?.prime.daysGrid ?? null}
          compareGrids={compareRiskGrids}
          palette="heatwave"
          scale={riskScale}
          unit="days"
          decimals={0}
          smoothInterpolation={smoothInterpolation}
          hoveredCell={hoveredCell}
          onHoverCell={setHoveredCell}
          emptyMessage="Run simulation to map Prime heatwave risk"
          compact
        />

        <GridMap
          title="Branch"
          badgeVariant="branch"
          subtitle={`days > ${settings.heatwaveThresholdC.toFixed(1)}°C per cell`}
          grid={impact?.branch?.daysGrid ?? null}
          compareGrids={compareRiskGrids}
          palette="heatwave"
          scale={riskScale}
          unit="days"
          decimals={0}
          smoothInterpolation={smoothInterpolation}
          hoveredCell={hoveredCell}
          onHoverCell={setHoveredCell}
          emptyMessage="Run Branch or Zero shock to map Branch heatwave risk"
          compact
        />
      </div>

      {showDiffMap && (
        <div className="mt-7">
          <GridMap
            title="Risk Difference"
            badgeVariant="diff"
            subtitle={`Branch − Prime heatwave days (> ${settings.heatwaveThresholdC.toFixed(1)}°C)`}
            grid={impact?.diffGrid ?? null}
            palette="diverging"
            scale={diffScale}
            unit="days"
            decimals={0}
            smoothInterpolation={smoothInterpolation}
            isDifference
            hoveredCell={hoveredCell}
            onHoverCell={setHoveredCell}
            emptyMessage="Run both Prime and Branch to inspect heatwave day differences"
            compact
          />
        </div>
      )}
    </section>
  )
}
