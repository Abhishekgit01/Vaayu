import { useMemo } from 'react'
import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type StateSequence,
} from '../api/types'
import { computeDiffGrid, getDisplayChannelMeta } from '../lib/constants'
import {
  computeSpreadAnalysis,
  DAY1_LOCALITY_CELLS,
  FRONT_THRESHOLD_FRACTION,
  REFERENCE_FRONT_CELLS_PER_DAY,
} from '../lib/spread'

const LOCALITY_MARKER_PCT = 60

interface SpreadChartProps {
  primeRollout: StateSequence | null
  branchRollout: StateSequence | null
  shockMask: boolean[][]
  channel: DisplayChannelId
  stepIndex: number
}

interface LocalityEnergyStats {
  hasEnergy: boolean
  sharePct: number
  localEnergy: number
  totalEnergy: number
}

function computeDay1LocalityGauge(
  primeRollout: StateSequence | null,
  branchRollout: StateSequence | null,
  shockMask: boolean[][],
  channel: DisplayChannelId,
): LocalityEnergyStats | null {
  if (!primeRollout || !branchRollout || primeRollout.length === 0 || branchRollout.length === 0) {
    return null
  }

  const shockCells: Array<[number, number]> = []
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (shockMask[r]?.[c]) {
        shockCells.push([r, c])
      }
    }
  }
  if (shockCells.length === 0) return null

  const diffDay1 = computeDiffGrid(primeRollout[0], branchRollout[0], channel)
  const radiusSq = DAY1_LOCALITY_CELLS * DAY1_LOCALITY_CELLS

  let localEnergy = 0
  let totalEnergy = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const d = diffDay1[r][c]
      const e = d * d
      totalEnergy += e
      if (e === 0) continue

      if (shockMask[r][c]) {
        localEnergy += e
      } else {
        let minSq = Infinity
        for (let i = 0; i < shockCells.length; i++) {
          const dr = r - shockCells[i][0]
          const dc = c - shockCells[i][1]
          const sq = dr * dr + dc * dc
          if (sq < minSq) minSq = sq
        }
        if (minSq <= radiusSq) {
          localEnergy += e
        }
      }
    }
  }

  if (totalEnergy <= 0) {
    return {
      hasEnergy: false,
      sharePct: 0,
      localEnergy: 0,
      totalEnergy: 0,
    }
  }

  return {
    hasEnergy: true,
    sharePct: (localEnergy / totalEnergy) * 100,
    localEnergy,
    totalEnergy,
  }
}

/** Pick a clean integer Y-axis ceiling that tightly auto-fits the observed front distance */
function autoFitFrontCeiling(maxObserved: number): number {
  const target = Math.max(DAY1_LOCALITY_CELLS + 1.5, maxObserved * 1.2 + 1)
  if (target <= 6) return 6
  if (target <= 8) return 8
  if (target <= 10) return 10
  if (target <= 12) return 12
  if (target <= 16) return 16
  if (target <= 20) return 20
  return Math.ceil(target / 5) * 5
}

export function SpreadChart({
  primeRollout,
  branchRollout,
  shockMask,
  channel,
  stepIndex,
}: SpreadChartProps) {
  const chMeta = getDisplayChannelMeta(channel)

  const analysis = useMemo(() => {
    if (!primeRollout || !branchRollout) return null
    return computeSpreadAnalysis(primeRollout, branchRollout, shockMask, channel)
  }, [primeRollout, branchRollout, shockMask, channel])

  const locality = useMemo(
    () => computeDay1LocalityGauge(primeRollout, branchRollout, shockMask, channel),
    [primeRollout, branchRollout, shockMask, channel],
  )

  const activeProfile = analysis?.profiles[stepIndex] ?? null

  const W = 740
  const H = 220
  const PAD = { top: 18, right: 112, bottom: 32, left: 38 }
  const PLOT_W = W - PAD.left - PAD.right
  const PLOT_H = H - PAD.top - PAD.bottom

  const numDays = analysis?.profiles.length ?? 1
  const maxObservedFront = useMemo(() => {
    if (!analysis) return 0
    let m = 0
    for (const p of analysis.profiles) {
      if (p.frontDistanceCells > m) m = p.frontDistanceCells
    }
    return m
  }, [analysis])

  const frontYMax = useMemo(() => autoFitFrontCeiling(maxObservedFront), [maxObservedFront])

  const xDay = (day: number) => PAD.left + (day / Math.max(1, numDays)) * PLOT_W
  const yFront = (cells: number) =>
    PAD.top + PLOT_H - (Math.min(cells, frontYMax) / frontYMax) * PLOT_H

  // 8 cells/day dashed reference line from (t=0, 0c) clipped cleanly to frontYMax
  const { refLinePoints, refLabelPos } = useMemo(() => {
    const maxT = Math.min(numDays, frontYMax / REFERENCE_FRONT_CELLS_PER_DAY)
    const endX = xDay(maxT)
    const endY = yFront(maxT * REFERENCE_FRONT_CELLS_PER_DAY)
    const lastFront = analysis?.profiles[analysis.profiles.length - 1]?.frontDistanceCells ?? 0
    const frontEndY = yFront(lastFront) + 3.5
    let labelY = Math.max(PAD.top + 9, endY + 3)
    if (Math.abs(endX - (W - PAD.right)) < 48 && Math.abs(labelY - frontEndY) < 13) {
      labelY = Math.max(PAD.top + 6, frontEndY - 13)
    }
    return {
      refLinePoints: `${xDay(0).toFixed(1)},${yFront(0).toFixed(1)} ${endX.toFixed(1)},${endY.toFixed(1)}`,
      refLabelPos: { x: endX + 6, y: labelY },
    }
  }, [analysis, frontYMax, numDays])

  const frontPolyline = useMemo(() => {
    if (!analysis) return ''
    const origin = `${xDay(0).toFixed(1)},${yFront(0).toFixed(1)}`
    const rest = analysis.profiles
      .map((p) => `${xDay(p.day).toFixed(1)},${yFront(p.frontDistanceCells).toFixed(1)}`)
      .join(' ')
    return `${origin} ${rest}`
  }, [analysis, frontYMax, numDays])

  const formatMag = (v: number) => {
    if (v === 0) return '0'
    if (v < 0.01) return v.toExponential(1)
    if (v < 1) return v.toFixed(2)
    return v.toFixed(1)
  }

  const hasShockRun = Boolean(analysis && analysis.hasShockRegion)
  const isLocalized = locality?.hasEnergy && locality.sharePct >= LOCALITY_MARKER_PCT
  const lastProfile = analysis?.profiles[analysis.profiles.length - 1] ?? null
  const yTicks = [0, Math.round(frontYMax / 2), frontYMax]

  // Avoid vertical collision between right-margin 'Front (Xc)' and '3c locality' labels
  const localityY = yFront(DAY1_LOCALITY_CELLS)
  const lastFrontY = lastProfile ? yFront(lastProfile.frontDistanceCells) : -999
  const localityLabelY =
    Math.abs(localityY - lastFrontY) < 12 ? localityY + 13 : localityY + 3.5

  return (
    <section className={`ui-transition pt-8 ${hasShockRun ? 'opacity-100' : 'opacity-45'}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div className="flex flex-wrap items-baseline gap-4">
          <h2 className="font-serif-display text-3xl leading-none text-[#ECE7DA]">
            Propagation &amp; locality
          </h2>
          {!hasShockRun ? (
            <p className="text-xs text-[#8C958D]">
              Run a shock or preset to measure Day-1 locality and front distance across days.
            </p>
          ) : (
            <p className="text-xs text-[#8C958D]">
              {chMeta.shortLabel} ({Math.round(FRONT_THRESHOLD_FRACTION * 100)}% peak contour on
              map)
            </p>
          )}
        </div>

        {hasShockRun && activeProfile && (
          <p className="font-mono text-xs text-[#8C958D] tabular-nums">
            Day {activeProfile.day} front{' '}
            <span className="text-[#F0B84B]">{activeProfile.frontDistanceCells}c</span> · peak |Δ|{' '}
            <span className="text-[#ECE7DA]">
              {formatMag(activeProfile.peakMagnitude)} {chMeta.displayUnit}
            </span>
          </p>
        )}
      </div>

      {hasShockRun && analysis && (
        <div className="panel-enter mt-6 grid gap-8 lg:grid-cols-[280px_1fr]">
          {/* Locality Gauge */}
          <div className="flex flex-col justify-between border-t border-[#26312B] pt-4 lg:border-t-0 lg:border-r lg:pt-0 lg:pr-8">
            <div>
              <p className="text-xs text-[#8C958D]">
                Day-1 energy within {DAY1_LOCALITY_CELLS} cells
              </p>
              <p className="font-serif-display tabular-nums mt-2 text-6xl leading-none text-[#ECE7DA]">
                {locality?.hasEnergy ? `${locality.sharePct.toFixed(1)}%` : '—'}
              </p>
              <p className="mt-2 text-xs text-[#8C958D]">
                {locality?.hasEnergy
                  ? isLocalized
                    ? 'Localized within 3 cells on Day 1.'
                    : 'Dispersed beyond 3 cells on Day 1.'
                  : 'Identical Prime and Branch (zero energy).'}
              </p>
            </div>

            <div className="mt-6">
              <div className="relative mb-1.5 h-4 font-mono text-[10px] text-[#8C958D] tabular-nums">
                <span className="absolute left-0 text-[#5A645D]">0%</span>
                <span
                  style={{ left: `${LOCALITY_MARKER_PCT}%` }}
                  className="absolute -translate-x-1/2 text-[#ECE7DA]"
                >
                  {LOCALITY_MARKER_PCT}%
                </span>
                <span className="absolute right-0 text-[#5A645D]">100%</span>
              </div>

              <div className="relative h-1.5 w-full bg-[#141B18] outline-1 outline-[#26312B]">
                {locality?.hasEnergy && (
                  <div
                    style={{
                      width: `${Math.max(0, Math.min(100, locality.sharePct))}%`,
                      backgroundColor: isLocalized ? '#7FB5B0' : '#F0B84B',
                    }}
                    className="h-full transition-all duration-150 ease-out"
                  />
                )}
                <div
                  style={{ left: `${LOCALITY_MARKER_PCT}%` }}
                  className="pointer-events-none absolute -top-1.5 -bottom-1.5 w-[1px] bg-[#ECE7DA]"
                />
              </div>
            </div>
          </div>

          {/* Auto-fitting Front Distance vs Day SVG chart */}
          <div>
            <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full overflow-visible">
              {yTicks.map((cells) => {
                const y = yFront(cells)
                return (
                  <g key={cells}>
                    <line
                      x1={PAD.left}
                      y1={y}
                      x2={W - PAD.right}
                      y2={y}
                      stroke="#26312B"
                      strokeWidth={1}
                    />
                    <text
                      x={PAD.left - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      fill="#5A645D"
                      fontSize={10}
                      fontFamily="var(--font-mono)"
                    >
                      {cells}c
                    </text>
                  </g>
                )
              })}

              {/* Thin dashed 3-cell locality reference line */}
              <line
                x1={PAD.left}
                y1={localityY}
                x2={W - PAD.right}
                y2={localityY}
                stroke="#7FB5B0"
                strokeOpacity={0.55}
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text
                x={W - PAD.right + 6}
                y={localityLabelY}
                fill="#7FB5B0"
                fillOpacity={0.8}
                fontSize={9.5}
                fontFamily="var(--font-mono)"
              >
                3c locality
              </text>

              {/* Thin dashed 8 cells/day reference line labelled in mono */}
              <polyline
                points={refLinePoints}
                fill="none"
                stroke="#8C958D"
                strokeWidth={1.25}
                strokeDasharray="5 4"
              />
              <text
                x={refLabelPos.x}
                y={refLabelPos.y}
                fill="#8C958D"
                fontSize={10}
                fontFamily="var(--font-mono)"
              >
                8 cells/day
              </text>

              {/* Scrubber-synced vertical cursor */}
              {activeProfile && (
                <line
                  x1={xDay(activeProfile.day)}
                  y1={PAD.top}
                  x2={xDay(activeProfile.day)}
                  y2={H - PAD.bottom}
                  stroke="#8C958D"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              )}

              {/* X-axis day ticks */}
              <text
                x={xDay(0)}
                y={H - PAD.bottom + 16}
                textAnchor="middle"
                fill="#5A645D"
                fontSize={10}
                fontFamily="var(--font-mono)"
              >
                0
              </text>
              {analysis.profiles.map((p, idx) => {
                const showTick =
                  numDays <= 12 ||
                  p.day === 1 ||
                  p.day === numDays ||
                  idx === stepIndex ||
                  p.day % Math.ceil(numDays / 6) === 0
                if (!showTick) return null
                const isCur = idx === stepIndex
                return (
                  <text
                    key={p.day}
                    x={xDay(p.day)}
                    y={H - PAD.bottom + 16}
                    textAnchor="middle"
                    fill={isCur ? '#ECE7DA' : '#5A645D'}
                    fontSize={10}
                    fontFamily="var(--font-mono)"
                  >
                    D{p.day}
                  </text>
                )
              })}

              {/* Observed front line (1.5px, no fill) */}
              <polyline
                points={frontPolyline}
                fill="none"
                stroke="#F0B84B"
                strokeWidth={1.5}
              />

              {/* Direct inline label at end of observed front line */}
              {lastProfile && (
                <text
                  x={W - PAD.right + 6}
                  y={yFront(lastProfile.frontDistanceCells) + 3.5}
                  fill="#F0B84B"
                  fontSize={10.5}
                  fontFamily="var(--font-mono)"
                >
                  Front ({lastProfile.frontDistanceCells}c)
                </text>
              )}

              {/* Day nodes */}
              {analysis.profiles.map((p, idx) => {
                const isCur = idx === stepIndex
                return (
                  <circle
                    key={p.day}
                    cx={xDay(p.day)}
                    cy={yFront(p.frontDistanceCells)}
                    r={isCur ? 4 : 2}
                    fill={isCur ? '#F0B84B' : '#0D1210'}
                    stroke="#F0B84B"
                    strokeWidth={1.5}
                  />
                )
              })}
            </svg>
          </div>
        </div>
      )}
    </section>
  )
}
