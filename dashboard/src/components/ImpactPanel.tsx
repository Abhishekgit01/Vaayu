import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { GRID_COLS, GRID_ROWS, type Grid2D, type State3D, type StateSequence } from '../api/types'
import { colToLon, formatCoordLat, formatCoordLon, rowToLat } from '../lib/constants'
import {
  CELL_AREA_SQ_KM,
  compareKharifDrySpellImpact,
  compareWetBulbStepImpact,
  DEFAULT_IMPACT_SETTINGS,
  T2M_CHANNEL_INDEX,
  TCWV_CHANNEL_INDEX,
  TOTAL_DOMAIN_SQ_KM,
  TOTAL_GRID_CELLS,
  WET_BULB_DANGER_THRESHOLD_K,
  type DrySpellImpactSummary,
  type ImpactSettings,
  type WetBulbStepSummary,
} from '../lib/impact'

interface ImpactPanelProps {
  primeRollout: StateSequence | null
  branchRollout: StateSequence | null
  stepIndex?: number
  settings: ImpactSettings
  onSettingsChange: (next: ImpactSettings) => void
  smoothInterpolation?: boolean
}

interface HoveredStressCell {
  row: number
  col: number
}

const MINI_LATS = [35, 25, 15] as const
const MINI_LONS = [70, 80, 90, 100] as const

function formatSqKm(sqKm: number): string {
  return `${Math.round(sqKm).toLocaleString()} sq km`
}

function formatSignedSqKm(deltaSqKm: number): string {
  if (deltaSqKm === 0) return '0 sq km'
  const sign = deltaSqKm > 0 ? '+' : '−'
  return `${sign}${Math.abs(Math.round(deltaSqKm)).toLocaleString()} sq km`
}

interface MiniStressHeatmapProps {
  title: string
  accentColor: string
  subtitle: string
  summary: DrySpellImpactSummary | null
  otherSummary?: DrySpellImpactSummary | null
  stressThresholdDays: number
  dryThresholdMm: number
  maxHorizonSteps: number
  hoveredCell: HoveredStressCell | null
  onHoverCell: (cell: HoveredStressCell | null) => void
  emptyMessage: string
}

/**
 * Compact north-up 32x36 agronomic risk heatmap highlighting Severe Crop Stress cells
 * (>= severeStressConsecutiveDays with tp < drySpellThresholdMmDay).
 */
function MiniStressHeatmap({
  title,
  accentColor,
  subtitle,
  summary,
  otherSummary,
  stressThresholdDays,
  dryThresholdMm,
  maxHorizonSteps,
  hoveredCell,
  onHoverCell,
  emptyMessage,
}: MiniStressHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [pointerInside, setPointerInside] = useState(false)

  const drawMiniMap = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !summary) return

    const rect = container.getBoundingClientRect()
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

    const cellW = w / GRID_COLS
    const cellH = h / GRID_ROWS
    const maxSpan = Math.max(stressThresholdDays + 1, maxHorizonSteps)

    // Base surface
    ctx.fillStyle = '#141B18'
    ctx.fillRect(0, 0, w, h)

    // Cell coloring (North up: row 31 at top y=0, row 0 at bottom y=31)
    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      for (let c = 0; c < GRID_COLS; c++) {
        const streak = summary.maxDrySpellGrid[r][c]
        const isSevere = summary.severeStressMask[r][c] === 1

        if (isSevere) {
          // Severe Crop Stress ramp: lightning #F0B84B (at threshold) -> ember #D9593D -> deep rust #9E3321
          const severityT = Math.max(
            0,
            Math.min(1, (streak - stressThresholdDays) / Math.max(1, maxSpan - stressThresholdDays)),
          )
          const red = Math.round(240 - 23 * severityT)
          const green = Math.round(184 - 95 * severityT)
          const blue = Math.round(75 - 14 * severityT)
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        } else if (streak > 0) {
          // Sub-threshold dry run (1..stressThresholdDays-1): subtle muted earth tone
          const subT = streak / Math.max(1, stressThresholdDays)
          const red = Math.round(20 + 28 * subT)
          const green = Math.round(27 + 26 * subT)
          const blue = Math.round(24 + 14 * subT)
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        } else {
          continue
        }

        ctx.fillRect(
          Math.floor(c * cellW),
          Math.floor(vy * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        )
      }
    }

    // Crisp 1px perimeter outline around contiguous Severe Crop Stress zones
    const isStressedAt = (r: number, c: number) =>
      r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS && summary.severeStressMask[r][c] === 1

    ctx.save()
    ctx.strokeStyle = '#ECE7DA'
    ctx.lineWidth = Math.max(1, 1.1 * dpr)
    ctx.beginPath()
    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      const yTop = vy * cellH
      const yBot = (vy + 1) * cellH
      for (let c = 0; c < GRID_COLS; c++) {
        if (!isStressedAt(r, c)) continue
        const xLeft = c * cellW
        const xRight = (c + 1) * cellW
        if (!isStressedAt(r + 1, c)) {
          ctx.moveTo(xLeft, yTop)
          ctx.lineTo(xRight, yTop)
        }
        if (!isStressedAt(r - 1, c)) {
          ctx.moveTo(xLeft, yBot)
          ctx.lineTo(xRight, yBot)
        }
        if (!isStressedAt(r, c - 1)) {
          ctx.moveTo(xLeft, yTop)
          ctx.lineTo(xLeft, yBot)
        }
        if (!isStressedAt(r, c + 1)) {
          ctx.moveTo(xRight, yTop)
          ctx.lineTo(xRight, yBot)
        }
      }
    }
    ctx.stroke()
    ctx.restore()

    // Subtle 1px lat/lon graticule lines
    ctx.save()
    ctx.strokeStyle = 'rgba(90, 100, 93, 0.34)'
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    for (const lat of MINI_LATS) {
      const y = Math.round(((38 - lat) / GRID_ROWS) * h) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    for (const lon of MINI_LONS) {
      const x = Math.round(((lon - 65) / GRID_COLS) * w) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    ctx.stroke()
    ctx.restore()

    // Hovered cell reticle
    if (hoveredCell) {
      const hx = hoveredCell.col * cellW
      const hy = (GRID_ROWS - 1 - hoveredCell.row) * cellH
      ctx.save()
      ctx.strokeStyle = '#0D1210'
      ctx.lineWidth = 2.4 * dpr
      ctx.strokeRect(hx, hy, cellW, cellH)
      ctx.strokeStyle = '#ECE7DA'
      ctx.lineWidth = 1.2 * dpr
      ctx.strokeRect(hx, hy, cellW, cellH)
      ctx.restore()
    }
  }, [summary, stressThresholdDays, maxHorizonSteps, hoveredCell])

  useEffect(() => {
    drawMiniMap()
  }, [drawMiniMap])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => drawMiniMap())
    obs.observe(el)
    return () => obs.disconnect()
  }, [drawMiniMap])

  const pointerToCell = (e: ReactPointerEvent<HTMLDivElement>): HoveredStressCell | null => {
    const el = containerRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = (e.clientX - rect.left) / rect.width
    const ny = (e.clientY - rect.top) / rect.height
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null
    const col = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(nx * GRID_COLS)))
    const vy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(ny * GRID_ROWS)))
    return { row: GRID_ROWS - 1 - vy, col }
  }

  const hoverDetails = useMemo(() => {
    if (!hoveredCell || !summary) return null
    const { row, col } = hoveredCell
    const streak = summary.maxDrySpellGrid[row]?.[col]
    if (streak === undefined) return null
    const isSevere = summary.severeStressMask[row][col] === 1
    const rainMm = summary.totalRainfallGrid[row][col]
    const otherStreak = otherSummary?.maxDrySpellGrid[row]?.[col]
    return {
      row,
      col,
      lat: rowToLat(row),
      lon: colToLon(col),
      streak,
      isSevere,
      rainMm,
      deltaDays: otherStreak !== undefined ? streak - otherStreak : null,
    }
  }, [hoveredCell, summary, otherSummary])

  return (
    <div className="flex flex-col pt-2.5" style={{ borderTop: `1px solid ${accentColor}` }}>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-2">
          <h3 className="font-serif-display shrink-0 text-lg italic leading-none text-[#ECE7DA]">
            {title}
          </h3>
          <span className="truncate font-mono text-[11px] text-[#8C958D]">{subtitle}</span>
        </div>
        {summary && (
          <span className="shrink-0 font-mono text-[11px] text-[#ECE7DA] tabular-nums">
            {summary.severeStressCells} stressed ({formatSqKm(summary.severeStressAreaSqKm)})
          </span>
        )}
      </div>

      <div className="grid grid-cols-[32px_1fr] gap-x-1.5">
        <div className="relative select-none">
          {MINI_LATS.map((lat) => (
            <span
              key={lat}
              style={{ top: `${((38 - lat) / GRID_ROWS) * 100}%` }}
              className="tabular-nums absolute right-0 -translate-y-1/2 font-mono text-[9.5px] text-[#5A645D]"
            >
              {lat}°N
            </span>
          ))}
        </div>

        <div
          ref={containerRef}
          onPointerMove={(e) => {
            setPointerInside(true)
            if (containerRef.current) {
              const rect = containerRef.current.getBoundingClientRect()
              setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
            }
            onHoverCell(pointerToCell(e))
          }}
          onPointerLeave={() => {
            setPointerInside(false)
            setTooltipPos(null)
            onHoverCell(null)
          }}
          className="relative aspect-[36/32] w-full overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18]"
        >
          {summary ? (
            <>
              <canvas ref={canvasRef} className="block h-full w-full" />
              {pointerInside && tooltipPos && hoverDetails && (
                <div
                  style={{
                    left: Math.min(
                      Math.max(6, tooltipPos.x + 12),
                      (containerRef.current?.clientWidth ?? 240) - 196,
                    ),
                    top: tooltipPos.y > 88 ? tooltipPos.y - 82 : tooltipPos.y + 12,
                  }}
                  className="tooltip-float pointer-events-none absolute z-20 w-[188px] rounded-[3px] px-2.5 py-1.5 font-mono text-[10.5px] tabular-nums"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span
                      className={
                        hoverDetails.isSevere
                          ? 'font-medium text-[#F0B84B]'
                          : 'font-medium text-[#ECE7DA]'
                      }
                    >
                      {hoverDetails.isSevere ? 'Severe Crop Stress' : 'Below stress threshold'}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[#ECE7DA]">
                    Max dry spell: {hoverDetails.streak}d (&lt;{dryThresholdMm} mm)
                  </div>
                  <div className="text-[#8C958D]">
                    Total tp: {hoverDetails.rainMm.toFixed(1)} mm · [{hoverDetails.row},
                    {hoverDetails.col}]
                  </div>
                  <div className="text-[#5A645D]">
                    {formatCoordLat(hoverDetails.lat)}, {formatCoordLon(hoverDetails.lon)}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-[#5A645D]">
              {emptyMessage}
            </div>
          )}
        </div>

        <div />

        <div className="relative mt-1 h-3.5 select-none">
          {MINI_LONS.map((lon) => (
            <span
              key={lon}
              style={{ left: `${((lon - 65) / GRID_COLS) * 100}%` }}
              className={`tabular-nums absolute top-0 font-mono text-[9.5px] text-[#5A645D] ${
                lon === 100 ? '-translate-x-[82%]' : '-translate-x-1/2'
              }`}
            >
              {lon}°E
            </span>
          ))}
        </div>
      </div>

      {/* Legend bar */}
      <div className="mt-1.5 flex items-center justify-between pl-[38px] font-mono text-[10px] text-[#8C958D]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 border border-[#26312B] bg-[#141B18]" />
          0d dry
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 bg-[#303526]" />
          1–{Math.max(1, stressThresholdDays - 1)}d watch
        </span>
        <span className="inline-flex items-center gap-1.5 text-[#ECE7DA]">
          <span className="inline-block h-2 w-2 border border-[#ECE7DA] bg-[#F0B84B]" />
          ≥{stressThresholdDays}d Severe Crop Stress
        </span>
      </div>
    </div>
  )
}

function MiniStressShiftMap({
  shiftGrid,
  newlyStressed,
  relieved,
  title = 'Stress Shift (Branch − Prime)',
  emptyMessage = 'Run Branch to map newly stressed vs relieved cells',
  newlyLabel = 'Newly stressed',
  hoveredCell,
  onHoverCell,
}: {
  shiftGrid: Grid2D | null
  newlyStressed: number
  relieved: number
  title?: string
  emptyMessage?: string
  newlyLabel?: string
  hoveredCell: HoveredStressCell | null
  onHoverCell: (cell: HoveredStressCell | null) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  const drawShift = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !shiftGrid) return

    const rect = container.getBoundingClientRect()
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

    const cellW = w / GRID_COLS
    const cellH = h / GRID_ROWS

    // Clean bone zero background (#ECE7DA) matching the diverging Difference standard
    ctx.fillStyle = '#ECE7DA'
    ctx.fillRect(0, 0, w, h)

    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      for (let c = 0; c < GRID_COLS; c++) {
        const s = shiftGrid[r][c]
        if (s === 0) continue
        // +1 = newly stressed in Branch (#D9593D ember), -1 = stress relieved in Branch (#3F7C85 rain-teal)
        ctx.fillStyle = s > 0 ? '#D9593D' : '#3F7C85'
        ctx.fillRect(
          Math.floor(c * cellW),
          Math.floor(vy * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        )
      }
    }

    ctx.strokeStyle = 'rgba(90, 100, 93, 0.34)'
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    for (const lat of MINI_LATS) {
      const y = Math.round(((38 - lat) / GRID_ROWS) * h) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    for (const lon of MINI_LONS) {
      const x = Math.round(((lon - 65) / GRID_COLS) * w) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    ctx.stroke()

    if (hoveredCell) {
      ctx.strokeStyle = '#0D1210'
      ctx.lineWidth = 1.6 * dpr
      ctx.strokeRect(
        hoveredCell.col * cellW,
        (GRID_ROWS - 1 - hoveredCell.row) * cellH,
        cellW,
        cellH,
      )
    }
  }, [shiftGrid, hoveredCell])

  useEffect(() => {
    drawShift()
  }, [drawShift])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => drawShift())
    obs.observe(el)
    return () => obs.disconnect()
  }, [drawShift])

  const pointerToCell = (e: ReactPointerEvent<HTMLDivElement>): HoveredStressCell | null => {
    const el = containerRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = (e.clientX - rect.left) / rect.width
    const ny = (e.clientY - rect.top) / rect.height
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null
    return {
      row: GRID_ROWS - 1 - Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(ny * GRID_ROWS))),
      col: Math.max(0, Math.min(GRID_COLS - 1, Math.floor(nx * GRID_COLS))),
    }
  }

  return (
    <div className="flex flex-col border-t border-[#26312B] pt-2.5">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <h3 className="font-serif-display text-lg italic leading-none text-[#ECE7DA]">
            {title}
          </h3>
        </div>
        <span className="font-mono text-[11px] text-[#8C958D] tabular-nums">
          +{newlyStressed} new · −{relieved} relieved
        </span>
      </div>

      <div className="grid grid-cols-[32px_1fr] gap-x-1.5">
        <div className="relative select-none">
          {MINI_LATS.map((lat) => (
            <span
              key={lat}
              style={{ top: `${((38 - lat) / GRID_ROWS) * 100}%` }}
              className="tabular-nums absolute right-0 -translate-y-1/2 font-mono text-[9.5px] text-[#5A645D]"
            >
              {lat}°N
            </span>
          ))}
        </div>

        <div
          ref={containerRef}
          onPointerMove={(e) => onHoverCell(pointerToCell(e))}
          onPointerLeave={() => onHoverCell(null)}
          className="relative aspect-[36/32] w-full overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18]"
        >
          {shiftGrid ? (
            <canvas ref={canvasRef} className="block h-full w-full" />
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-[#5A645D]">
              {emptyMessage}
            </div>
          )}
        </div>

        <div />

        <div className="relative mt-1 h-3.5 select-none">
          {MINI_LONS.map((lon) => (
            <span
              key={lon}
              style={{ left: `${((lon - 65) / GRID_COLS) * 100}%` }}
              className={`tabular-nums absolute top-0 font-mono text-[9.5px] text-[#5A645D] ${
                lon === 100 ? '-translate-x-[82%]' : '-translate-x-1/2'
              }`}
            >
              {lon}°E
            </span>
          ))}
        </div>
      </div>

      <div className="mt-1.5 flex items-center justify-between pl-[38px] font-mono text-[10px] text-[#8C958D]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 bg-[#3F7C85]" />
          Relieved
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 border border-[#26312B] bg-[#ECE7DA]" />
          Unchanged (0)
        </span>
        <span className="inline-flex items-center gap-1.5 text-[#ECE7DA]">
          <span className="inline-block h-2 w-2 bg-[#D9593D]" />
          {newlyLabel}
        </span>
      </div>
    </div>
  )
}

interface MiniWetBulbHeatmapProps {
  title: string
  accentColor: string
  subtitle: string
  summary: WetBulbStepSummary | null
  physicalState: State3D | null
  otherSummary?: WetBulbStepSummary | null
  thresholdK: number
  sharedMinK: number
  sharedMaxK: number
  hoveredCell: HoveredStressCell | null
  onHoverCell: (cell: HoveredStressCell | null) => void
  emptyMessage: string
}

/**
 * Static north-up 32x36 heatmap for the current timeline step's Heat-Humidity Stress Index (screening proxy):
 *   HHSI = t2m + (tcwv * 0.15)
 * Highlights cells exceeding thresholdK (default 310.0 K, approx 37°C) as Severe Human Danger.
 */
function MiniWetBulbHeatmap({
  title,
  accentColor,
  subtitle,
  summary,
  physicalState,
  otherSummary,
  thresholdK,
  sharedMinK,
  sharedMaxK,
  hoveredCell,
  onHoverCell,
  emptyMessage,
}: MiniWetBulbHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [pointerInside, setPointerInside] = useState(false)

  const drawWetBulbMap = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !summary) return

    const rect = container.getBoundingClientRect()
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

    const cellW = w / GRID_COLS
    const cellH = h / GRID_ROWS
    const subFloor = Math.min(sharedMinK, thresholdK - 22)
    const subSpan = Math.max(1e-3, thresholdK - subFloor)
    const dangerSpan = Math.max(4, sharedMaxK - thresholdK)

    ctx.fillStyle = '#141B18'
    ctx.fillRect(0, 0, w, h)

    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      for (let c = 0; c < GRID_COLS; c++) {
        const wb = summary.wetBulbGrid[r][c]
        const isDanger = summary.severeDangerMask[r][c] === 1

        if (isDanger) {
          // Severe Human Danger (> thresholdK): Lightning Gold (#F0B84B) -> Ember (#D9593D) -> Deep Crimson (#8C2318)
          const sev = Math.max(0, Math.min(1, (wb - thresholdK) / dangerSpan))
          const red = Math.round(240 - 100 * sev * 0.55)
          const green = Math.round(184 - 149 * sev)
          const blue = Math.round(75 - 51 * sev)
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        } else {
          // Sub-threshold static Wet_Bulb_Eq field: deep Petrichor (#141B18) -> warm slate-ochre (#5B4D36)
          const subT = Math.max(0, Math.min(1, (wb - subFloor) / subSpan))
          const red = Math.round(20 + 71 * subT)
          const green = Math.round(27 + 50 * subT)
          const blue = Math.round(24 + 30 * subT)
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        }

        ctx.fillRect(
          Math.floor(c * cellW),
          Math.floor(vy * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        )
      }
    }

    // Crisp 1px bone perimeter outline around contiguous Severe Human Danger cells
    const isDangerAt = (r: number, c: number) =>
      r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS && summary.severeDangerMask[r][c] === 1

    ctx.save()
    ctx.strokeStyle = '#ECE7DA'
    ctx.lineWidth = Math.max(1, 1.1 * dpr)
    ctx.beginPath()
    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      const yTop = vy * cellH
      const yBot = (vy + 1) * cellH
      for (let c = 0; c < GRID_COLS; c++) {
        if (!isDangerAt(r, c)) continue
        const xLeft = c * cellW
        const xRight = (c + 1) * cellW
        if (!isDangerAt(r + 1, c)) {
          ctx.moveTo(xLeft, yTop)
          ctx.lineTo(xRight, yTop)
        }
        if (!isDangerAt(r - 1, c)) {
          ctx.moveTo(xLeft, yBot)
          ctx.lineTo(xRight, yBot)
        }
        if (!isDangerAt(r, c - 1)) {
          ctx.moveTo(xLeft, yTop)
          ctx.lineTo(xLeft, yBot)
        }
        if (!isDangerAt(r, c + 1)) {
          ctx.moveTo(xRight, yTop)
          ctx.lineTo(xRight, yBot)
        }
      }
    }
    ctx.stroke()
    ctx.restore()

    // Subtle 1px lat/lon graticule lines
    ctx.save()
    ctx.strokeStyle = 'rgba(90, 100, 93, 0.34)'
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    for (const lat of MINI_LATS) {
      const y = Math.round(((38 - lat) / GRID_ROWS) * h) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    for (const lon of MINI_LONS) {
      const x = Math.round(((lon - 65) / GRID_COLS) * w) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    ctx.stroke()
    ctx.restore()

    if (hoveredCell) {
      const hx = hoveredCell.col * cellW
      const hy = (GRID_ROWS - 1 - hoveredCell.row) * cellH
      ctx.save()
      ctx.strokeStyle = '#0D1210'
      ctx.lineWidth = 2.4 * dpr
      ctx.strokeRect(hx, hy, cellW, cellH)
      ctx.strokeStyle = '#ECE7DA'
      ctx.lineWidth = 1.2 * dpr
      ctx.strokeRect(hx, hy, cellW, cellH)
      ctx.restore()
    }
  }, [summary, thresholdK, sharedMinK, sharedMaxK, hoveredCell])

  useEffect(() => {
    drawWetBulbMap()
  }, [drawWetBulbMap])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => drawWetBulbMap())
    obs.observe(el)
    return () => obs.disconnect()
  }, [drawWetBulbMap])

  const pointerToCell = (e: ReactPointerEvent<HTMLDivElement>): HoveredStressCell | null => {
    const el = containerRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = (e.clientX - rect.left) / rect.width
    const ny = (e.clientY - rect.top) / rect.height
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null
    const col = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(nx * GRID_COLS)))
    const vy = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(ny * GRID_ROWS)))
    return { row: GRID_ROWS - 1 - vy, col }
  }

  const hoverDetails = useMemo(() => {
    if (!hoveredCell || !summary) return null
    const { row, col } = hoveredCell
    const wb = summary.wetBulbGrid[row]?.[col]
    if (wb === undefined) return null
    const isDanger = summary.severeDangerMask[row][col] === 1
    const t2m = physicalState?.[T2M_CHANNEL_INDEX]?.[row]?.[col]
    const tcwv = physicalState?.[TCWV_CHANNEL_INDEX]?.[row]?.[col]
    const otherWb = otherSummary?.wetBulbGrid[row]?.[col]
    return {
      row,
      col,
      lat: rowToLat(row),
      lon: colToLon(col),
      wb,
      wbCelsius: wb - 273.15,
      isDanger,
      t2m,
      tcwv,
      deltaWb: otherWb !== undefined ? wb - otherWb : null,
    }
  }, [hoveredCell, summary, physicalState, otherSummary])

  return (
    <div className="flex flex-col pt-2.5" style={{ borderTop: `1px solid ${accentColor}` }}>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-2">
          <h3 className="font-serif-display shrink-0 text-lg italic leading-none text-[#ECE7DA]">
            {title}
          </h3>
          <span className="truncate font-mono text-[11px] text-[#8C958D]">{subtitle}</span>
        </div>
        {summary && (
          <span className="shrink-0 font-mono text-[11px] text-[#ECE7DA] tabular-nums">
            {summary.severeDangerCells} danger ({formatSqKm(summary.severeDangerAreaSqKm)})
          </span>
        )}
      </div>

      <div className="grid grid-cols-[32px_1fr] gap-x-1.5">
        <div className="relative select-none">
          {MINI_LATS.map((lat) => (
            <span
              key={lat}
              style={{ top: `${((38 - lat) / GRID_ROWS) * 100}%` }}
              className="tabular-nums absolute right-0 -translate-y-1/2 font-mono text-[9.5px] text-[#5A645D]"
            >
              {lat}°N
            </span>
          ))}
        </div>

        <div
          ref={containerRef}
          onPointerMove={(e) => {
            setPointerInside(true)
            if (containerRef.current) {
              const rect = containerRef.current.getBoundingClientRect()
              setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
            }
            onHoverCell(pointerToCell(e))
          }}
          onPointerLeave={() => {
            setPointerInside(false)
            setTooltipPos(null)
            onHoverCell(null)
          }}
          className="relative aspect-[36/32] w-full overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18]"
        >
          {summary ? (
            <>
              <canvas ref={canvasRef} className="block h-full w-full" />
              {pointerInside && tooltipPos && hoverDetails && (
                <div
                  style={{
                    left: Math.min(
                      Math.max(6, tooltipPos.x + 12),
                      (containerRef.current?.clientWidth ?? 240) - 208,
                    ),
                    top: tooltipPos.y > 96 ? tooltipPos.y - 90 : tooltipPos.y + 12,
                  }}
                  className="tooltip-float pointer-events-none absolute z-20 w-[200px] rounded-[3px] px-2.5 py-1.5 font-mono text-[10.5px] tabular-nums"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span
                      className={
                        hoverDetails.isDanger
                          ? 'font-medium text-[#D9593D]'
                          : 'font-medium text-[#ECE7DA]'
                      }
                    >
                      {hoverDetails.isDanger ? 'Severe Human Danger' : 'Below danger threshold'}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[#ECE7DA]">
                    HHSI: {hoverDetails.wb.toFixed(1)} K ({hoverDetails.wbCelsius.toFixed(1)}°C)
                  </div>
                  {hoverDetails.t2m !== undefined && hoverDetails.tcwv !== undefined && (
                    <div className="text-[#8C958D]">
                      t2m {hoverDetails.t2m.toFixed(1)} K · tcwv {hoverDetails.tcwv.toFixed(1)} kg/m²
                    </div>
                  )}
                  <div className="text-[#5A645D]">
                    [{hoverDetails.row},{hoverDetails.col}] · {formatCoordLat(hoverDetails.lat)},{' '}
                    {formatCoordLon(hoverDetails.lon)}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-[#5A645D]">
              {emptyMessage}
            </div>
          )}
        </div>

        <div />

        <div className="relative mt-1 h-3.5 select-none">
          {MINI_LONS.map((lon) => (
            <span
              key={lon}
              style={{ left: `${((lon - 65) / GRID_COLS) * 100}%` }}
              className={`tabular-nums absolute top-0 font-mono text-[9.5px] text-[#5A645D] ${
                lon === 100 ? '-translate-x-[82%]' : '-translate-x-1/2'
              }`}
            >
              {lon}°E
            </span>
          ))}
        </div>
      </div>

      <div className="mt-1.5 flex items-center justify-between pl-[38px] font-mono text-[10px] text-[#8C958D]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 border border-[#26312B] bg-[#141B18]" />
          Cool / moderate
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 bg-[#5B4D36]" />
          ≤{thresholdK.toFixed(1)} K
        </span>
        <span className="inline-flex items-center gap-1.5 text-[#ECE7DA]">
          <span className="inline-block h-2 w-2 border border-[#ECE7DA] bg-[#D9593D]" />
          &gt;{thresholdK.toFixed(1)} K Severe Human Danger
        </span>
      </div>
    </div>
  )
}

export function ImpactPanel({
  primeRollout,
  branchRollout,
  stepIndex = 0,
  settings,
  onSettingsChange,
}: ImpactPanelProps) {
  const [hoveredCell, setHoveredCell] = useState<HoveredStressCell | null>(null)
  const [hoveredWetBulbCell, setHoveredWetBulbCell] = useState<HoveredStressCell | null>(null)

  // Ensure clean defaults even if an older settings shape is passed in
  const activeSettings: ImpactSettings = useMemo(
    () => ({
      drySpellThresholdMmDay:
        typeof settings?.drySpellThresholdMmDay === 'number' &&
        Number.isFinite(settings.drySpellThresholdMmDay)
          ? settings.drySpellThresholdMmDay
          : DEFAULT_IMPACT_SETTINGS.drySpellThresholdMmDay,
      severeStressConsecutiveDays:
        typeof settings?.severeStressConsecutiveDays === 'number' &&
        Number.isFinite(settings.severeStressConsecutiveDays)
          ? Math.max(1, Math.round(settings.severeStressConsecutiveDays))
          : DEFAULT_IMPACT_SETTINGS.severeStressConsecutiveDays,
      wetBulbDangerThresholdK:
        typeof settings?.wetBulbDangerThresholdK === 'number' &&
        Number.isFinite(settings.wetBulbDangerThresholdK)
          ? settings.wetBulbDangerThresholdK
          : WET_BULB_DANGER_THRESHOLD_K,
    }),
    [settings],
  )

  const comparison = useMemo(
    () => compareKharifDrySpellImpact(primeRollout, branchRollout, activeSettings),
    [primeRollout, branchRollout, activeSettings],
  )

  const wetBulbThresholdK = activeSettings.wetBulbDangerThresholdK ?? WET_BULB_DANGER_THRESHOLD_K

  // Calculate Lethal Wet-Bulb Index statically on-the-fly whenever timeline scrubber (stepIndex) moves
  const wetBulbComparison = useMemo(
    () => compareWetBulbStepImpact(primeRollout, branchRollout, stepIndex, wetBulbThresholdK),
    [primeRollout, branchRollout, stepIndex, wetBulbThresholdK],
  )

  const wetBulbSharedRange = useMemo(() => {
    if (!wetBulbComparison) {
      return { min: 285, max: Math.max(314, wetBulbThresholdK + 4) }
    }
    const min = wetBulbComparison.branch
      ? Math.min(wetBulbComparison.prime.minWetBulbK, wetBulbComparison.branch.minWetBulbK)
      : wetBulbComparison.prime.minWetBulbK
    const max = wetBulbComparison.branch
      ? Math.max(wetBulbComparison.prime.maxWetBulbK, wetBulbComparison.branch.maxWetBulbK)
      : wetBulbComparison.prime.maxWetBulbK
    return {
      min,
      max: Math.max(max, wetBulbThresholdK + 2),
    }
  }, [wetBulbComparison, wetBulbThresholdK])

  const evaluatedSteps = comparison?.prime.evaluatedSteps ?? 0
  const hasBranch = Boolean(comparison?.branch)
  const isShortHorizon =
    evaluatedSteps > 0 && evaluatedSteps < activeSettings.severeStressConsecutiveDays

  const headlineSummary = comparison?.branch ?? comparison?.prime ?? null
  const headlineSourceLabel = comparison?.branch ? 'Timeline Branch' : 'Timeline Prime'
  const deltaAreaSqKm = comparison?.deltaSevereAreaSqKm ?? null
  const deltaCells = comparison?.deltaSevereCells ?? null

  const wetBulbHeadline = wetBulbComparison?.branch ?? wetBulbComparison?.prime ?? null
  const wetBulbSourceLabel = wetBulbComparison?.branch ? 'Timeline Branch' : 'Timeline Prime'
  const wetBulbDeltaAreaSqKm = wetBulbComparison?.deltaDangerAreaSqKm ?? null
  const wetBulbDeltaCells = wetBulbComparison?.deltaDangerCells ?? null
  const activeStepNumber = (wetBulbComparison?.stepIndex ?? stepIndex) + 1
  const primeStepState =
    primeRollout && primeRollout.length > 0
      ? primeRollout[Math.max(0, Math.min(primeRollout.length - 1, stepIndex))]
      : null
  const branchStepState =
    branchRollout && branchRollout.length > 0
      ? branchRollout[Math.max(0, Math.min(branchRollout.length - 1, stepIndex))]
      : null

  const isCustomCriteria =
    activeSettings.drySpellThresholdMmDay !== DEFAULT_IMPACT_SETTINGS.drySpellThresholdMmDay ||
    activeSettings.severeStressConsecutiveDays !==
      DEFAULT_IMPACT_SETTINGS.severeStressConsecutiveDays
  const isCustomWetBulb = Math.abs(wetBulbThresholdK - WET_BULB_DANGER_THRESHOLD_K) > 1e-6

  return (
    <section
      aria-label="Agronomic & Human Risk Decision Support System"
      className={`ui-transition border-t border-[#26312B] pt-8 pb-14 ${
        primeRollout ? 'opacity-100' : 'opacity-50'
      }`}
    >
      {/* Header & Official Agronomic Index Criteria */}
      <div className="flex flex-wrap items-baseline justify-between gap-6">
        <div>
          <p className="font-mono text-[11px] text-[#7FB5B0]">
            Agronomic Decision Support System
          </p>
          <h2 className="font-serif-display mt-1 text-3xl leading-none text-[#ECE7DA]">
            Dry-Spell Onset Flag
          </h2>
          <p className="mt-1.5 text-xs text-[#8C958D]">
            Consecutive rollout days with rainfall <code className="font-mono text-[#ECE7DA]">tp &lt; {activeSettings.drySpellThresholdMmDay.toFixed(1)} mm/day</code> (channel 1);{' '}
            <strong className="font-medium text-[#ECE7DA]">Severe Crop Stress</strong> flagged at{' '}
            <code className="font-mono text-[#ECE7DA]">≥ {activeSettings.severeStressConsecutiveDays}</code> consecutive dry days (1 cell ≈ {CELL_AREA_SQ_KM.toLocaleString()} sq km).
          </p>
        </div>

        {/* Visible Agronomic Criteria Controls */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-[#8C958D]">
          <label className="inline-flex items-center gap-1.5">
            <span>Dry day &lt;</span>
            <input
              type="number"
              step={0.5}
              min={0.1}
              max={25}
              value={activeSettings.drySpellThresholdMmDay}
              onChange={(e) => {
                const v = Number.parseFloat(e.target.value)
                if (Number.isFinite(v)) {
                  onSettingsChange({
                    ...activeSettings,
                    drySpellThresholdMmDay: Math.max(0, Math.min(50, v)),
                  })
                }
              }}
              aria-label="Dry-day rainfall threshold in mm/day"
              className="tabular-nums w-14 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1.5 py-0.5 text-right font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
            />
            <span className="font-mono">mm/d</span>
          </label>

          <label className="inline-flex items-center gap-1.5 border-l border-[#26312B] pl-3">
            <span>Severe ≥</span>
            <input
              type="number"
              step={1}
              min={1}
              max={30}
              value={activeSettings.severeStressConsecutiveDays}
              onChange={(e) => {
                const v = Number.parseInt(e.target.value, 10)
                if (!Number.isNaN(v)) {
                  onSettingsChange({
                    ...activeSettings,
                    severeStressConsecutiveDays: Math.max(1, Math.min(60, v)),
                  })
                }
              }}
              aria-label="Consecutive dry days for Severe Crop Stress"
              className="tabular-nums w-12 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1.5 py-0.5 text-right font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
            />
            <span className="font-mono">days</span>
          </label>

          {isCustomCriteria && (
            <button
              type="button"
              onClick={() => onSettingsChange({ ...DEFAULT_IMPACT_SETTINGS })}
              className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
            >
              Official standard (2.5 mm, 5d)
            </button>
          )}
        </div>
      </div>

      {isShortHorizon && (
        <p className="mt-4 border-l-2 border-[#F0B84B] bg-[#141B18] px-3.5 py-2 font-mono text-xs text-[#ECE7DA]">
          Rollout horizon is {evaluatedSteps} {evaluatedSteps === 1 ? 'day' : 'days'} (&lt;{' '}
          {activeSettings.severeStressConsecutiveDays} consecutive days required for Severe Crop
          Stress) — stressed area reads 0 sq km.
        </p>
      )}

      {/* Headline Comparison: Severe Crop Stress Area (sq km) in Prime vs Branch */}
      <div className="mt-8 grid items-end gap-8 border-b border-[#26312B] pb-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1.2fr)]">
        <div>
          <p className="text-xs text-[#8C958D]">
            Severe Crop Stress area · {headlineSourceLabel}
          </p>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-3">
            <span
              className="font-serif-display tabular-nums leading-[0.86] tracking-tight text-[#ECE7DA]"
              style={{ fontSize: 'clamp(4.75rem, 7.5vw, 7.25rem)' }}
            >
              {headlineSummary
                ? Math.round(headlineSummary.severeStressAreaSqKm).toLocaleString()
                : '—'}
            </span>
            <span className="font-mono text-base text-[#8C958D]">sq km</span>
          </div>
          <p className="mt-3 font-mono text-xs text-[#8C958D] tabular-nums">
            {headlineSummary ? (
              <>
                <span className="text-[#ECE7DA]">
                  {headlineSummary.severeStressCells.toLocaleString()} stressed cells
                </span>{' '}
                of {TOTAL_GRID_CELLS.toLocaleString()} ({headlineSummary.severeStressPct.toFixed(1)}%
                of {(TOTAL_DOMAIN_SQ_KM / 1_000_000).toFixed(2)}M sq km · longest dry spell{' '}
                {headlineSummary.maxConsecutiveDryDays}d)
              </>
            ) : (
              <span>Awaiting rollout data</span>
            )}
          </p>
        </div>

        {/* Prime vs Branch sq km Comparison Table */}
        <div className="grid gap-6 border-t border-[#26312B] pt-6 sm:grid-cols-3 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
          {/* Timeline Prime */}
          <div className="border-t border-[#7FB5B0] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#7FB5B0]">
              Timeline Prime
            </p>
            <p className="mt-2 font-mono text-xl font-medium text-[#ECE7DA] tabular-nums">
              {comparison ? formatSqKm(comparison.prime.severeStressAreaSqKm) : '—'}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {comparison
                ? `${comparison.prime.severeStressCells.toLocaleString()} cells stressed`
                : '—'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {comparison
                ? `max streak ${comparison.prime.maxConsecutiveDryDays}d / ${evaluatedSteps}d`
                : 'perturbation: null'}
            </p>
          </div>

          {/* Timeline Branch */}
          <div className="border-t border-[#F0B84B] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#F0B84B]">
              Timeline Branch
            </p>
            <p className="mt-2 font-mono text-xl font-medium text-[#ECE7DA] tabular-nums">
              {comparison?.branch ? formatSqKm(comparison.branch.severeStressAreaSqKm) : '—'}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {comparison?.branch
                ? `${comparison.branch.severeStressCells.toLocaleString()} cells stressed`
                : 'run shock to compare'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {comparison?.branch
                ? `max streak ${comparison.branch.maxConsecutiveDryDays}d / ${comparison.branch.evaluatedSteps}d`
                : 'awaiting Branch rollout'}
            </p>
          </div>

          {/* Net Crop Stress Delta (Branch - Prime) */}
          <div className="border-t border-[#26312B] pt-2.5">
            <p className="font-serif-display text-lg italic leading-none text-[#ECE7DA]">
              Branch − Prime
            </p>
            <p
              className={`mt-2 font-mono text-xl font-medium tabular-nums ${
                deltaAreaSqKm === null
                  ? 'text-[#5A645D]'
                  : deltaAreaSqKm > 0
                    ? 'text-[#D9593D]'
                    : deltaAreaSqKm < 0
                      ? 'text-[#7FB5B0]'
                      : 'text-[#ECE7DA]'
              }`}
            >
              {deltaAreaSqKm !== null ? formatSignedSqKm(deltaAreaSqKm) : '—'}
            </p>
            <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
              {deltaCells !== null
                ? `${deltaCells > 0 ? '+' : deltaCells < 0 ? '−' : ''}${Math.abs(deltaCells).toLocaleString()} stressed cells`
                : '—'}
            </p>
            <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
              {hasBranch && comparison
                ? `+${comparison.newlyStressedCells} new · −${comparison.relievedCells} relieved`
                : '1 cell ≈ 10,000 sq km'}
            </p>
          </div>
        </div>
      </div>

      {/* Mini Risk-Heatmaps of Stressed Cells */}
      <div className="mt-7 grid gap-6 lg:grid-cols-3">
        <MiniStressHeatmap
          title="Timeline Prime"
          accentColor="#7FB5B0"
          subtitle={`tp < ${activeSettings.drySpellThresholdMmDay.toFixed(1)} mm/d`}
          summary={comparison?.prime ?? null}
          otherSummary={comparison?.branch ?? null}
          stressThresholdDays={activeSettings.severeStressConsecutiveDays}
          dryThresholdMm={activeSettings.drySpellThresholdMmDay}
          maxHorizonSteps={evaluatedSteps}
          hoveredCell={hoveredCell}
          onHoverCell={setHoveredCell}
          emptyMessage="Awaiting Prime rollout"
        />

        <MiniStressHeatmap
          title="Timeline Branch"
          accentColor="#F0B84B"
          subtitle={`tp < ${activeSettings.drySpellThresholdMmDay.toFixed(1)} mm/d`}
          summary={comparison?.branch ?? null}
          otherSummary={comparison?.prime ?? null}
          stressThresholdDays={activeSettings.severeStressConsecutiveDays}
          dryThresholdMm={activeSettings.drySpellThresholdMmDay}
          maxHorizonSteps={comparison?.branch?.evaluatedSteps ?? evaluatedSteps}
          hoveredCell={hoveredCell}
          onHoverCell={setHoveredCell}
          emptyMessage="Run a shock or preset to map Branch crop stress"
        />

        <MiniStressShiftMap
          shiftGrid={comparison?.stressShiftGrid ?? null}
          newlyStressed={comparison?.newlyStressedCells ?? 0}
          relieved={comparison?.relievedCells ?? 0}
          hoveredCell={hoveredCell}
          onHoverCell={setHoveredCell}
        />
      </div>

      {/* Secondary Metric: Heat-Humidity Stress Index (screening proxy, Current Scrubber Step) */}
      <div className="mt-12 border-t border-[#26312B] pt-8">
        <div className="flex flex-wrap items-baseline justify-between gap-6">
          <div>
            <p className="font-mono text-[11px] text-[#F0B84B]">
              Human Heat-Humidity Risk · Day {evaluatedSteps > 0 ? activeStepNumber : '—'}
              {evaluatedSteps > 0 ? ` / ${evaluatedSteps}` : ''} (scrubber-synced)
            </p>
            <h2 className="font-serif-display mt-1 text-3xl leading-none text-[#ECE7DA]">
              Heat-Humidity Stress Index (screening proxy)
            </h2>
            <p className="mt-1.5 text-xs text-[#8C958D]">
              Current-step screening proxy{' '}
              <code className="font-mono text-[#ECE7DA]">HHSI = t2m + (tcwv × 0.15)</code>{' '}
              combining 2m Temperature (<code className="font-mono text-[#ECE7DA]">t2m</code>, index
              0) and Total Column Water Vapor (<code className="font-mono text-[#ECE7DA]">tcwv</code>
              , index 5);{' '}
              <strong className="font-medium text-[#D9593D]">Severe Human Danger</strong> flagged
              where <code className="font-mono text-[#ECE7DA]">HHSI &gt; {wetBulbThresholdK.toFixed(1)} K</code>{' '}
              (≈ {(wetBulbThresholdK - 273.15).toFixed(0)}°C threshold; 1 cell ≈{' '}
              {CELL_AREA_SQ_KM.toLocaleString()} sq km).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs text-[#8C958D]">
            <label className="inline-flex items-center gap-1.5">
              <span>Severe Human Danger &gt;</span>
              <input
                type="number"
                step={0.5}
                min={290}
                max={335}
                value={wetBulbThresholdK}
                onChange={(e) => {
                  const v = Number.parseFloat(e.target.value)
                  if (Number.isFinite(v)) {
                    onSettingsChange({
                      ...activeSettings,
                      wetBulbDangerThresholdK: Math.max(280, Math.min(350, v)),
                    })
                  }
                }}
                aria-label="Severe Human Danger Heat-Humidity Stress Index threshold in Kelvin"
                className="tabular-nums w-16 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1.5 py-0.5 text-right font-mono text-xs text-[#ECE7DA] focus:border-[#F0B84B] focus:outline-none"
              />
              <span className="font-mono">K</span>
            </label>

            {isCustomWetBulb && (
              <button
                type="button"
                onClick={() =>
                  onSettingsChange({
                    ...activeSettings,
                    wetBulbDangerThresholdK: WET_BULB_DANGER_THRESHOLD_K,
                  })
                }
                className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
              >
                Standard threshold (310.0 K)
              </button>
            )}
          </div>
        </div>

        {/* Headline: Severe Human Danger Total Area & Cell Count at Current Step */}
        <div className="mt-8 grid items-end gap-8 border-b border-[#26312B] pb-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1.2fr)]">
          <div>
            <p className="text-xs text-[#8C958D]">
              Severe Human Danger area · {wetBulbSourceLabel} · Day{' '}
              {evaluatedSteps > 0 ? activeStepNumber : '—'}
            </p>
            <div className="mt-2 flex flex-wrap items-baseline gap-x-3">
              <span
                className="font-serif-display tabular-nums leading-[0.86] tracking-tight text-[#ECE7DA]"
                style={{ fontSize: 'clamp(4.25rem, 6.75vw, 6.5rem)' }}
              >
                {wetBulbHeadline
                  ? Math.round(wetBulbHeadline.severeDangerAreaSqKm).toLocaleString()
                  : '—'}
              </span>
              <span className="font-mono text-base text-[#8C958D]">sq km</span>
            </div>
            <p className="mt-3 font-mono text-xs text-[#8C958D] tabular-nums">
              {wetBulbHeadline ? (
                <>
                  <span
                    className={
                      wetBulbHeadline.severeDangerCells > 0 ? 'text-[#D9593D]' : 'text-[#ECE7DA]'
                    }
                  >
                    {wetBulbHeadline.severeDangerCells.toLocaleString()} Severe Human Danger{' '}
                    {wetBulbHeadline.severeDangerCells === 1 ? 'cell' : 'cells'}
                  </span>{' '}
                  of {TOTAL_GRID_CELLS.toLocaleString()} (
                  {wetBulbHeadline.severeDangerPct.toFixed(1)}% of{' '}
                  {(TOTAL_DOMAIN_SQ_KM / 1_000_000).toFixed(2)}M sq km · peak HHSI{' '}
                  {wetBulbHeadline.maxWetBulbK.toFixed(1)} K /{' '}
                  {(wetBulbHeadline.maxWetBulbK - 273.15).toFixed(1)}°C)
                </>
              ) : (
                <span>Awaiting rollout data</span>
              )}
            </p>
          </div>

          {/* Prime vs Branch Severe Human Danger Comparison Table */}
          <div className="grid gap-6 border-t border-[#26312B] pt-6 sm:grid-cols-3 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
            <div className="border-t border-[#7FB5B0] pt-2.5">
              <p className="font-serif-display text-lg italic leading-none text-[#7FB5B0]">
                Timeline Prime
              </p>
              <p className="mt-2 font-mono text-xl font-medium text-[#ECE7DA] tabular-nums">
                {wetBulbComparison
                  ? formatSqKm(wetBulbComparison.prime.severeDangerAreaSqKm)
                  : '—'}
              </p>
              <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
                {wetBulbComparison
                  ? `${wetBulbComparison.prime.severeDangerCells.toLocaleString()} danger cells`
                  : '—'}
              </p>
              <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
                {wetBulbComparison
                  ? `peak ${wetBulbComparison.prime.maxWetBulbK.toFixed(1)} K (Day ${activeStepNumber})`
                  : 'perturbation: null'}
              </p>
            </div>

            <div className="border-t border-[#F0B84B] pt-2.5">
              <p className="font-serif-display text-lg italic leading-none text-[#F0B84B]">
                Timeline Branch
              </p>
              <p className="mt-2 font-mono text-xl font-medium text-[#ECE7DA] tabular-nums">
                {wetBulbComparison?.branch
                  ? formatSqKm(wetBulbComparison.branch.severeDangerAreaSqKm)
                  : '—'}
              </p>
              <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
                {wetBulbComparison?.branch
                  ? `${wetBulbComparison.branch.severeDangerCells.toLocaleString()} danger cells`
                  : 'run shock to compare'}
              </p>
              <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
                {wetBulbComparison?.branch
                  ? `peak ${wetBulbComparison.branch.maxWetBulbK.toFixed(1)} K (Day ${activeStepNumber})`
                  : 'awaiting Branch rollout'}
              </p>
            </div>

            <div className="border-t border-[#26312B] pt-2.5">
              <p className="font-serif-display text-lg italic leading-none text-[#ECE7DA]">
                Branch − Prime
              </p>
              <p
                className={`mt-2 font-mono text-xl font-medium tabular-nums ${
                  wetBulbDeltaAreaSqKm === null
                    ? 'text-[#5A645D]'
                    : wetBulbDeltaAreaSqKm > 0
                      ? 'text-[#D9593D]'
                      : wetBulbDeltaAreaSqKm < 0
                        ? 'text-[#7FB5B0]'
                        : 'text-[#ECE7DA]'
                }`}
              >
                {wetBulbDeltaAreaSqKm !== null ? formatSignedSqKm(wetBulbDeltaAreaSqKm) : '—'}
              </p>
              <p className="mt-1 font-mono text-xs text-[#8C958D] tabular-nums">
                {wetBulbDeltaCells !== null
                  ? `${wetBulbDeltaCells > 0 ? '+' : wetBulbDeltaCells < 0 ? '−' : ''}${Math.abs(wetBulbDeltaCells).toLocaleString()} danger cells`
                  : '—'}
              </p>
              <p className="mt-2 font-mono text-[11px] text-[#5A645D] tabular-nums">
                {wetBulbComparison?.branch
                  ? `+${wetBulbComparison.newlyEndangeredCells} new · −${wetBulbComparison.relievedCells} relieved`
                  : '1 cell ≈ 10,000 sq km'}
              </p>
            </div>
          </div>
        </div>

        {/* Static Current-Step Heat-Humidity Stress Index Heatmaps */}
        <div className="mt-7 grid gap-6 lg:grid-cols-3">
          <MiniWetBulbHeatmap
            title="Timeline Prime"
            accentColor="#7FB5B0"
            subtitle={`Day ${evaluatedSteps > 0 ? activeStepNumber : '—'} · HHSI`}
            summary={wetBulbComparison?.prime ?? null}
            physicalState={primeStepState}
            otherSummary={wetBulbComparison?.branch ?? null}
            thresholdK={wetBulbThresholdK}
            sharedMinK={wetBulbSharedRange.min}
            sharedMaxK={wetBulbSharedRange.max}
            hoveredCell={hoveredWetBulbCell}
            onHoverCell={setHoveredWetBulbCell}
            emptyMessage="Awaiting Prime rollout"
          />

          <MiniWetBulbHeatmap
            title="Timeline Branch"
            accentColor="#F0B84B"
            subtitle={`Day ${evaluatedSteps > 0 ? activeStepNumber : '—'} · HHSI`}
            summary={wetBulbComparison?.branch ?? null}
            physicalState={branchStepState}
            otherSummary={wetBulbComparison?.prime ?? null}
            thresholdK={wetBulbThresholdK}
            sharedMinK={wetBulbSharedRange.min}
            sharedMaxK={wetBulbSharedRange.max}
            hoveredCell={hoveredWetBulbCell}
            onHoverCell={setHoveredWetBulbCell}
            emptyMessage="Run a shock or preset to map Branch stress index"
          />

          <MiniStressShiftMap
            title="Danger Shift (Branch − Prime)"
            emptyMessage="Run Branch to map newly endangered vs relieved cells"
            newlyLabel="Newly in danger"
            shiftGrid={wetBulbComparison?.dangerShiftGrid ?? null}
            newlyStressed={wetBulbComparison?.newlyEndangeredCells ?? 0}
            relieved={wetBulbComparison?.relievedCells ?? 0}
            hoveredCell={hoveredWetBulbCell}
            onHoverCell={setHoveredWetBulbCell}
          />
        </div>
      </div>
    </section>
  )
}
