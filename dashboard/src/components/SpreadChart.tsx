import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type StateSequence,
} from '../api/types'
import { computeDiffGrid, getDisplayChannelMeta } from '../lib/constants'
import { extractIsoContourSegments } from '../lib/shock'
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

function buildNorthUpDistanceVertexField(
  shockMask: boolean[][],
  nx: number,
  ny: number,
): { field: number[][]; hasShock: boolean } {
  const shockCells: Array<[number, number]> = []
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (shockMask[r]?.[c]) {
        shockCells.push([r, c])
      }
    }
  }
  if (shockCells.length === 0) {
    return { field: [], hasShock: false }
  }

  const field: number[][] = new Array(ny + 1)
  for (let iy = 0; iy <= ny; iy++) {
    const gy = (iy / ny) * GRID_ROWS
    const rowCoord = GRID_ROWS - 1 - (gy - 0.5)
    const rowArr = new Array<number>(nx + 1)
    for (let ix = 0; ix <= nx; ix++) {
      const gx = (ix / nx) * GRID_COLS
      const colCoord = gx - 0.5
      let minSq = Infinity
      for (let i = 0; i < shockCells.length; i++) {
        const dr = rowCoord - shockCells[i][0]
        const dc = colCoord - shockCells[i][1]
        const sq = dr * dr + dc * dc
        if (sq < minSq) minSq = sq
      }
      rowArr[ix] = Math.sqrt(minSq)
    }
    field[iy] = rowArr
  }

  return { field, hasShock: true }
}

function DifferenceMapFrontRipple({
  host,
  distanceVertexField,
  nx,
  ny,
  frontDistanceCells,
  day,
  hasSignal,
}: {
  host: HTMLElement
  distanceVertexField: number[][]
  nx: number
  ny: number
  frontDistanceCells: number
  day: number
  hasSignal: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const radius = frontDistanceCells > 0 ? frontDistanceCells : 0.55

  const rings = useMemo(() => {
    if (!hasSignal || distanceVertexField.length === 0) {
      return { primary: [], echo: [] }
    }
    const primary = extractIsoContourSegments(distanceVertexField, nx, ny, radius)
    const echo =
      radius >= 1.8
        ? extractIsoContourSegments(distanceVertexField, nx, ny, Math.max(0.55, radius - 1.35))
        : []
    return { primary, echo }
  }, [distanceVertexField, nx, ny, radius, hasSignal])

  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = host.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = Math.max(1, Math.round(rect.width * dpr))
    const h = Math.max(1, Math.round(rect.height * dpr))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, w, h)

    if (!hasSignal || rings.primary.length === 0) return

    const toX = (gx: number) => (gx / GRID_COLS) * w
    const toY = (gy: number) => (gy / GRID_ROWS) * h

    // Inner ripple echo (contrasts cleanly on bone #ECE7DA zero background)
    if (rings.echo.length > 0) {
      ctx.save()
      ctx.setLineDash([4 * dpr, 4 * dpr])
      ctx.strokeStyle = 'rgba(20, 27, 24, 0.45)'
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      for (const [x1, y1, x2, y2] of rings.echo) {
        ctx.moveTo(toX(x1), toY(y1))
        ctx.lineTo(toX(x2), toY(y2))
      }
      ctx.stroke()
      ctx.restore()
    }

    // Primary front contour: subtle bone halo + crisp dark slate 1.5px ring
    ctx.save()
    ctx.lineCap = 'round'
    ctx.beginPath()
    let topSeg: [number, number] | null = null
    for (const [x1, y1, x2, y2] of rings.primary) {
      const px1 = toX(x1)
      const py1 = toY(y1)
      const px2 = toX(x2)
      const py2 = toY(y2)
      ctx.moveTo(px1, py1)
      ctx.lineTo(px2, py2)
      if (!topSeg || py1 < topSeg[1]) {
        topSeg = [(px1 + px2) / 2, (py1 + py2) / 2]
      }
    }

    ctx.strokeStyle = 'rgba(236, 231, 218, 0.72)'
    ctx.lineWidth = 3.2 * dpr
    ctx.stroke()

    ctx.strokeStyle = '#0D1210'
    ctx.lineWidth = 1.5 * dpr
    ctx.stroke()
    ctx.restore()

    if (topSeg) {
      ctx.save()
      const label = `D${day} front · ${frontDistanceCells}c`
      ctx.font = `400 ${Math.round(10 * dpr)}px 'Geist Mono', ui-monospace, monospace`
      const textW = ctx.measureText(label).width
      const padX = 5 * dpr
      const boxW = textW + padX * 2
      const boxH = 16 * dpr
      const bx = Math.max(4 * dpr, Math.min(w - boxW - 4 * dpr, topSeg[0] - boxW / 2))
      const by = Math.max(4 * dpr, Math.min(h - boxH - 4 * dpr, topSeg[1] - boxH - 4 * dpr))

      ctx.fillStyle = '#141B18'
      ctx.strokeStyle = '#26312B'
      ctx.lineWidth = 1 * dpr
      ctx.fillRect(bx, by, boxW, boxH)
      ctx.strokeRect(bx, by, boxW, boxH)

      ctx.fillStyle = '#ECE7DA'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, bx + boxW / 2, by + boxH / 2 + 0.5 * dpr)
      ctx.restore()
    }
  }, [host, hasSignal, rings, day, frontDistanceCells])

  useEffect(() => {
    paint()
  }, [paint])

  useEffect(() => {
    const obs = new ResizeObserver(() => paint())
    obs.observe(host)
    return () => obs.disconnect()
  }, [host, paint])

  return createPortal(
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-[6] block h-full w-full"
    />,
    host,
  )
}

export function SpreadChart({
  primeRollout,
  branchRollout,
  shockMask,
  channel,
  stepIndex,
}: SpreadChartProps) {
  const chMeta = getDisplayChannelMeta(channel)
  const [diffMapHost, setDiffMapHost] = useState<HTMLElement | null>(null)

  const analysis = useMemo(() => {
    if (!primeRollout || !branchRollout) return null
    return computeSpreadAnalysis(primeRollout, branchRollout, shockMask, channel)
  }, [primeRollout, branchRollout, shockMask, channel])

  const locality = useMemo(
    () => computeDay1LocalityGauge(primeRollout, branchRollout, shockMask, channel),
    [primeRollout, branchRollout, shockMask, channel],
  )

  const NX = GRID_COLS * 2
  const NY = GRID_ROWS * 2
  const distanceVertices = useMemo(
    () => buildNorthUpDistanceVertexField(shockMask, NX, NY),
    [shockMask, NX, NY],
  )

  useEffect(() => {
    const syncDiffHost = () => {
      let found: HTMLElement | null = null
      const headings = document.querySelectorAll('h3')
      for (let i = 0; i < headings.length; i++) {
        const h3 = headings[i]
        const txt = h3.textContent?.trim() ?? ''
        if (txt === 'Difference (Branch − Prime)' || txt === 'Difference') {
          const card = h3.closest('.panel-glass')
          const plot = card?.querySelector<HTMLElement>('div.aspect-\\[36\\/32\\]') ?? null
          if (plot && plot.querySelector('canvas')) {
            found = plot
            break
          }
        }
      }
      setDiffMapHost((prev) => (prev === found ? prev : found))
    }

    syncDiffHost()
    const obs = new MutationObserver(syncDiffHost)
    obs.observe(document.body, { childList: true, subtree: true })
    return () => obs.disconnect()
  }, [primeRollout, branchRollout])

  const activeProfile = analysis?.profiles[stepIndex] ?? null

  // Front Distance vs Day SVG geometry (room on right margin for direct labels)
  const W = 740
  const H = 232
  const PAD = { top: 20, right: 118, bottom: 36, left: 42 }
  const PLOT_W = W - PAD.left - PAD.right
  const PLOT_H = H - PAD.top - PAD.bottom

  const numDays = analysis?.profiles.length ?? 1
  const maxObservedFront = useMemo(() => {
    if (!analysis) return 10
    let m = 0
    for (const p of analysis.profiles) {
      if (p.frontDistanceCells > m) m = p.frontDistanceCells
    }
    return m
  }, [analysis])

  const frontYMax = Math.max(
    12,
    maxObservedFront + 4,
    Math.min(Math.max(analysis?.maxDistance ?? 24, 24), numDays * REFERENCE_FRONT_CELLS_PER_DAY),
  )

  const xDay = (day: number) =>
    PAD.left + (numDays <= 1 ? PLOT_W / 2 : ((day - 1) / (numDays - 1)) * PLOT_W)
  const yFront = (cells: number) =>
    PAD.top + PLOT_H - (Math.min(cells, frontYMax) / frontYMax) * PLOT_H

  const { refLinePoints, refEndPoint } = useMemo(() => {
    if (!analysis || analysis.profiles.length === 0) {
      return { refLinePoints: '', refEndPoint: { x: PAD.left, y: PAD.top } }
    }
    const pts: Array<[number, number]> = []
    const maxDay = analysis.profiles.length
    for (let i = 0; i <= 48; i++) {
      const d = 1 + (i / 48) * Math.max(0, maxDay - 1)
      const dist = d * REFERENCE_FRONT_CELLS_PER_DAY
      if (dist <= frontYMax) {
        pts.push([xDay(d), yFront(dist)])
      } else {
        const clipDay = frontYMax / REFERENCE_FRONT_CELLS_PER_DAY
        if (clipDay >= 1 && clipDay <= maxDay) {
          pts.push([xDay(clipDay), yFront(frontYMax)])
        }
        break
      }
    }
    const last = pts[pts.length - 1] ?? [PAD.left, PAD.top]
    return {
      refLinePoints: pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '),
      refEndPoint: { x: last[0], y: last[1] },
    }
  }, [analysis, frontYMax, numDays])

  const frontPolyline = useMemo(() => {
    if (!analysis) return ''
    return analysis.profiles
      .map((p) => `${xDay(p.day).toFixed(1)},${yFront(p.frontDistanceCells).toFixed(1)}`)
      .join(' ')
  }, [analysis, frontYMax, numDays])

  const formatMag = (v: number) => {
    if (v === 0) return '0'
    if (v < 0.01) return v.toExponential(1)
    if (v < 1) return v.toFixed(2)
    return v.toFixed(1)
  }

  const isLocalized = locality?.hasEnergy && locality.sharePct >= LOCALITY_MARKER_PCT
  const lastProfile = analysis?.profiles[analysis.profiles.length - 1] ?? null

  return (
    <section className="border-t border-[#26312B] pt-8">
      {diffMapHost &&
        analysis &&
        analysis.hasShockRegion &&
        activeProfile &&
        distanceVertices.hasShock && (
          <DifferenceMapFrontRipple
            host={diffMapHost}
            distanceVertexField={distanceVertices.field}
            nx={NX}
            ny={NY}
            frontDistanceCells={activeProfile.frontDistanceCells}
            day={activeProfile.day}
            hasSignal={analysis.globalPeak > 0}
          />
        )}

      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <h2 className="font-serif-display text-3xl leading-none text-[#ECE7DA]">
            Propagation &amp; locality
          </h2>
          <p className="mt-1.5 text-xs text-[#8C958D]">
            Front distance in cells from shock boundary ({chMeta.shortLabel},{' '}
            {Math.round(FRONT_THRESHOLD_FRACTION * 100)}% of rollout peak) and Day-1 energy
            concentration.
          </p>
        </div>

        {analysis && activeProfile && (
          <p className="font-mono text-xs text-[#8C958D] tabular-nums">
            Day {activeProfile.day} front{' '}
            <span className="text-[#F0B84B]">{activeProfile.frontDistanceCells} cells</span> · ref{' '}
            <span className="text-[#ECE7DA]">
              {activeProfile.day * REFERENCE_FRONT_CELLS_PER_DAY} cells
            </span>{' '}
            · peak |Δ|{' '}
            <span className="text-[#ECE7DA]">
              {formatMag(activeProfile.peakMagnitude)} {chMeta.displayUnit}
            </span>
          </p>
        )}
      </div>

      {!analysis ? (
        <p className="mt-6 py-8 text-xs text-[#5A645D]">
          Run Branch or Zero shock to measure Day-1 locality and front propagation against Timeline
          Prime.
        </p>
      ) : !analysis.hasShockRegion ? (
        <p className="mt-6 py-8 text-xs text-[#5A645D]">
          Define a shock footprint on the map to measure spatial propagation.
        </p>
      ) : (
        <div className="mt-6 grid gap-8 lg:grid-cols-[300px_1fr]">
          {/* Locality Gauge */}
          <div className="flex flex-col justify-between border-t border-[#26312B] pt-4 lg:border-t-0 lg:border-r lg:pt-0 lg:pr-8">
            <div>
              <p className="text-xs text-[#8C958D]">
                Locality gauge · Day-1 energy within {DAY1_LOCALITY_CELLS} cells
              </p>
              <p className="font-serif-display tabular-nums mt-2 text-6xl leading-none text-[#ECE7DA]">
                {locality?.hasEnergy ? `${locality.sharePct.toFixed(1)}%` : '—'}
              </p>
              <p className="mt-2 text-xs text-[#8C958D]">
                {locality?.hasEnergy
                  ? isLocalized
                    ? 'Concentrated inside the 3-cell Day-1 locality envelope.'
                    : 'Below the 60% Day-1 locality threshold.'
                  : 'Zero response energy (identical Prime and Branch).'}
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
                    style={{ width: `${Math.max(0, Math.min(100, locality.sharePct))}%` }}
                    className="h-full transition-all duration-150 ease-out"
                    data-localized={isLocalized ? 'true' : 'false'}
                  >
                    <div
                      className="h-full w-full"
                      style={{
                        backgroundColor: isLocalized ? '#7FB5B0' : '#F0B84B',
                      }}
                    />
                  </div>
                )}

                {/* 1px marker at 60% */}
                <div
                  style={{ left: `${LOCALITY_MARKER_PCT}%` }}
                  className="pointer-events-none absolute -top-1.5 -bottom-1.5 w-[1px] bg-[#ECE7DA]"
                />
              </div>

              <div className="mt-3 flex justify-between font-mono text-[11px] text-[#5A645D] tabular-nums">
                <span>≤3c ΣΔ² {locality?.hasEnergy ? formatMag(locality.localEnergy) : '0'}</span>
                <span>total {locality?.hasEnergy ? formatMag(locality.totalEnergy) : '0'}</span>
              </div>
            </div>
          </div>

          {/* Thin 1.5px Front Distance vs Day chart with direct labels (no legend box, no fills) */}
          <div>
            <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full overflow-visible">
              {/* Hairline Y gridlines */}
              {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                const y = PAD.top + PLOT_H * (1 - frac)
                const cells = Math.round(frontYMax * frac)
                return (
                  <g key={frac}>
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

              {/* 3-cell Day-1 locality line */}
              <line
                x1={PAD.left}
                y1={yFront(DAY1_LOCALITY_CELLS)}
                x2={W - PAD.right}
                y2={yFront(DAY1_LOCALITY_CELLS)}
                stroke="#26312B"
                strokeWidth={1}
                strokeDasharray="2 3"
              />
              <text
                x={W - PAD.right + 6}
                y={yFront(DAY1_LOCALITY_CELLS) + 3}
                fill="#5A645D"
                fontSize={10}
                fontFamily="var(--font-mono)"
              >
                3c locality
              </text>

              {/* Dashed reference line at 8 cells/day with direct mono label */}
              {refLinePoints && (
                <g>
                  <polyline
                    points={refLinePoints}
                    fill="none"
                    stroke="#7FB5B0"
                    strokeWidth={1.5}
                    strokeDasharray="5 4"
                  />
                  <text
                    x={Math.min(W - PAD.right + 6, refEndPoint.x + 6)}
                    y={Math.max(PAD.top + 10, refEndPoint.y + 3)}
                    fill="#7FB5B0"
                    fontSize={10.5}
                    fontFamily="var(--font-mono)"
                  >
                    8 cells/day
                  </text>
                </g>
              )}

              {/* Scrubber-synced vertical hairline */}
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
              {analysis.profiles.map((p, idx) => {
                const showTick =
                  numDays <= 14 ||
                  p.day === 1 ||
                  p.day === numDays ||
                  idx === stepIndex ||
                  p.day % Math.ceil(numDays / 7) === 0
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

              {/* Thin 1.5px observed front line (no fill) */}
              <polyline
                points={frontPolyline}
                fill="none"
                stroke="#F0B84B"
                strokeWidth={1.5}
              />

              {/* Direct inline label on observed front line */}
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

              {/* Minimal day nodes */}
              {analysis.profiles.map((p, idx) => {
                const isCur = idx === stepIndex
                const cx = xDay(p.day)
                const cy = yFront(p.frontDistanceCells)
                return (
                  <circle
                    key={p.day}
                    cx={cx}
                    cy={cy}
                    r={isCur ? 4 : 2}
                    fill={isCur ? '#F0B84B' : '#0D1210'}
                    stroke="#F0B84B"
                    strokeWidth={1.5}
                  />
                )
              })}

              <text
                x={PAD.left}
                y={H - 4}
                textAnchor="start"
                fill="#8C958D"
                fontSize={10.5}
                fontFamily="var(--font-sans)"
              >
                Forecast day
              </text>
            </svg>
          </div>
        </div>
      )}
    </section>
  )
}
