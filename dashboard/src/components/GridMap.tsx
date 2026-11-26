import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import {
  GRID_COLS,
  GRID_ROWS,
  type Grid2D,
  type State3D,
} from '../api/types'
import {
  paletteToCssGradient,
  writePaletteColor,
  type ColorScale,
  type PaletteKind,
} from '../lib/colormaps'
import {
  colToLon,
  formatCoordLat,
  formatCoordLon,
  rowToLat,
} from '../lib/constants'
import {
  drawCanvasMapOverlays,
  GRATICULE_LATS,
  GRATICULE_LONS,
  latToNormY,
  lonToNormX,
  type ContourSegment,
  type WindArrow,
} from '../lib/heatmap'
import type { ShockConfig } from '../lib/shock'
import type { ViewFieldId } from '../lib/vorticity'
import { WindFlowCanvas } from './WindFlowCanvas'

export interface HoveredCell {
  row: number
  col: number
}

// 10x spatial downscaling: 32x36 -> 320x360 (rows x cols) pre-indexed for 60fps+ bilinear interpolation
const DOWNSCALE_FACTOR = 10
const UPSCALED_COLS = GRID_COLS * DOWNSCALE_FACTOR // 360
const UPSCALED_ROWS = GRID_ROWS * DOWNSCALE_FACTOR // 320
const SEA_TINT_STRENGTH = 0.24

const BILINEAR_C0 = new Int32Array(UPSCALED_COLS)
const BILINEAR_C1 = new Int32Array(UPSCALED_COLS)
const BILINEAR_FC = new Float64Array(UPSCALED_COLS)
for (let px = 0; px < UPSCALED_COLS; px++) {
  const gx = Math.max(0, Math.min(GRID_COLS - 1, (px + 0.5) / DOWNSCALE_FACTOR - 0.5))
  const c0 = Math.floor(gx)
  BILINEAR_C0[px] = c0
  BILINEAR_C1[px] = Math.min(GRID_COLS - 1, c0 + 1)
  BILINEAR_FC[px] = gx - c0
}

const BILINEAR_R0 = new Int32Array(UPSCALED_ROWS)
const BILINEAR_R1 = new Int32Array(UPSCALED_ROWS)
const BILINEAR_FR = new Float64Array(UPSCALED_ROWS)
for (let py = 0; py < UPSCALED_ROWS; py++) {
  // North up: py = 0 corresponds to northernmost row 31, py = 319 to southernmost row 0
  const gy = Math.max(0, Math.min(GRID_ROWS - 1, (py + 0.5) / DOWNSCALE_FACTOR - 0.5))
  const gr = GRID_ROWS - 1 - gy
  const r0 = Math.floor(gr)
  BILINEAR_R0[py] = r0
  BILINEAR_R1[py] = Math.min(GRID_ROWS - 1, r0 + 1)
  BILINEAR_FR[py] = gr - r0
}

/**
 * Upscale a 32x36 physical scalar grid to 320x360 via Bilinear Interpolation BEFORE
 * applying the sequential or zero-centered diverging color scale, reusing a single ImageData
 * buffer for zero-allocation 60fps scrubbing.
 */
function rasterizeBilinear320x360(
  bufferCanvas: HTMLCanvasElement,
  reusableImageData: ImageData,
  grid: Grid2D,
  scale: ColorScale,
  palette: PaletteKind,
  landSeaMask?: Grid2D | null,
): void {
  if (bufferCanvas.width !== UPSCALED_COLS || bufferCanvas.height !== UPSCALED_ROWS) {
    bufferCanvas.width = UPSCALED_COLS
    bufferCanvas.height = UPSCALED_ROWS
  }

  const ctx = bufferCanvas.getContext('2d')
  if (!ctx) return

  const data = reusableImageData.data
  const scaleMin = scale.min
  const scaleSpan = scale.max - scaleMin
  const isDiverging = scale.diverging
  const invSpan = scaleSpan > 0 ? 1 / scaleSpan : 0

  for (let py = 0; py < UPSCALED_ROWS; py++) {
    const r0 = BILINEAR_R0[py]
    const r1 = BILINEAR_R1[py]
    const fr = BILINEAR_FR[py]
    const row0 = grid[r0]
    const row1 = grid[r1]
    const mRow0 = landSeaMask ? landSeaMask[r0] : null
    const mRow1 = landSeaMask ? landSeaMask[r1] : null
    const rowByteOffset = py * UPSCALED_COLS * 4

    for (let px = 0; px < UPSCALED_COLS; px++) {
      const c0 = BILINEAR_C0[px]
      const c1 = BILINEAR_C1[px]
      const fc = BILINEAR_FC[px]

      const v0 = row0[c0] + (row0[c1] - row0[c0]) * fc
      const v1 = row1[c0] + (row1[c1] - row1[c0]) * fc
      const interpVal = v0 + (v1 - v0) * fr

      let norm: number
      if (isDiverging) {
        norm = interpVal === 0 || invSpan === 0 ? 0.5 : (interpVal - scaleMin) * invSpan
      } else {
        norm = invSpan === 0 ? 0 : (interpVal - scaleMin) * invSpan
      }
      if (norm < 0) norm = 0
      else if (norm > 1) norm = 1

      const off = rowByteOffset + px * 4
      writePaletteColor(palette, norm, data, off)

      if (mRow0 && mRow1) {
        const m0 = mRow0[c0] + (mRow0[c1] - mRow0[c0]) * fc
        const m1 = mRow1[c0] + (mRow1[c1] - mRow1[c0]) * fc
        const landFrac = m0 + (m1 - m0) * fr
        const sea = (1 - (landFrac < 0 ? 0 : landFrac > 1 ? 1 : landFrac)) * SEA_TINT_STRENGTH
        if (sea > 0) {
          const keep = 1 - sea
          data[off] = Math.round(data[off] * keep + 13 * sea)
          data[off + 1] = Math.round(data[off + 1] * keep + 18 * sea)
          data[off + 2] = Math.round(data[off + 2] * keep + 16 * sea)
        }
      }
    }
  }

  ctx.putImageData(reusableImageData, 0, 0)
}

interface GridMapProps {
  title: string
  badgeText?: string
  badgeVariant?: 'prime' | 'branch' | 'diff' | 'neutral'
  subtitle?: string
  /** 32x36 values already in display units (row 0 = southernmost lat 6.5) */
  grid: Grid2D | null
  /** Optional full physical state [6][32][36] for wire-unit tooltip notes */
  physicalState?: State3D | null
  /** Optional counterpart grids for cross-timeline hover readout */
  compareGrids?: {
    prime?: Grid2D | null
    branch?: Grid2D | null
    diff?: Grid2D | null
  }
  channel?: ViewFieldId
  palette: PaletteKind
  scale: ColorScale
  unit: string
  decimals?: number
  smoothInterpolation?: boolean
  landSeaMask?: Grid2D | null
  coastlineSegments?: ContourSegment[]
  isDifference?: boolean
  windFlowEnabled?: boolean
  diffWindArrows?: WindArrow[]
  showWindArrows?: boolean
  windArrows?: WindArrow[]
  showTpThresholdOutline?: boolean
  tpSegments?: ContourSegment[]
  shockConfig?: ShockConfig | null
  showShockOverlay?: boolean
  frontContourSegments?: ContourSegment[]
  localityRingSegments?: ContourSegment[]
  refRingSegments?: ContourSegment[]
  stepDayNumber?: number
  allowShockPlacement?: boolean
  onShockClick?: (row: number, col: number) => void
  onShockDrag?: (startRow: number, startCol: number, endRow: number, endCol: number) => void
  hoveredCell?: HoveredCell | null
  onHoverCell?: (cell: HoveredCell | null) => void
  emptyMessage?: string
  compact?: boolean
}

export function GridMap({
  title,
  badgeVariant = 'neutral',
  subtitle,
  grid,
  physicalState,
  compareGrids,
  channel,
  palette,
  scale,
  unit,
  decimals = 1,
  landSeaMask,
  coastlineSegments,
  isDifference = false,
  windFlowEnabled = false,
  diffWindArrows,
  showWindArrows = false,
  windArrows,
  showTpThresholdOutline = false,
  tpSegments,
  shockConfig,
  showShockOverlay = false,
  frontContourSegments,
  localityRingSegments,
  refRingSegments,
  stepDayNumber,
  allowShockPlacement = false,
  onShockClick,
  onShockDrag,
  hoveredCell,
  onHoverCell,
  emptyMessage = 'Run simulation to populate map',
}: GridMapProps) {
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const bufferCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageDataRef = useRef<ImageData | null>(null)
  const rasterCacheKeyRef = useRef<{
    grid: Grid2D | null
    min: number
    max: number
    diverging: boolean
    palette: PaletteKind
    landSeaMask: Grid2D | null | undefined
  } | null>(null)

  const containerRef = useRef<HTMLDivElement | null>(null)
  const dragStartRef = useRef<HoveredCell | null>(null)
  const didDragRef = useRef(false)
  const [localHover, setLocalHover] = useState<HoveredCell | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [isPointerInside, setIsPointerInside] = useState(false)

  const activeHover = hoveredCell !== undefined ? hoveredCell : localHover

  const paintCanvas = useCallback(() => {
    const canvas = displayCanvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const rect = container.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const targetW = Math.max(1, Math.round(rect.width * dpr))
    const targetH = Math.max(1, Math.round(rect.height * dpr))

    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW
      canvas.height = targetH
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, targetW, targetH)

    if (!grid) return

    if (!bufferCanvasRef.current) {
      bufferCanvasRef.current = document.createElement('canvas')
    }
    if (!imageDataRef.current) {
      imageDataRef.current = new ImageData(UPSCALED_COLS, UPSCALED_ROWS)
    }

    const prev = rasterCacheKeyRef.current
    const needsRaster =
      !prev ||
      prev.grid !== grid ||
      prev.min !== scale.min ||
      prev.max !== scale.max ||
      prev.diverging !== scale.diverging ||
      prev.palette !== palette ||
      prev.landSeaMask !== landSeaMask

    if (needsRaster) {
      rasterizeBilinear320x360(
        bufferCanvasRef.current,
        imageDataRef.current,
        grid,
        scale,
        palette,
        landSeaMask,
      )
      rasterCacheKeyRef.current = {
        grid,
        min: scale.min,
        max: scale.max,
        diverging: scale.diverging,
        palette,
        landSeaMask,
      }
    }

    ctx.save()
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bufferCanvasRef.current, 0, 0, targetW, targetH)
    ctx.restore()

    drawCanvasMapOverlays(ctx, {
      width: targetW,
      height: targetH,
      dpr,
      isLightBackground: scale.diverging,
      coastlineSegments,
      tpSegments: !isDifference && showTpThresholdOutline ? tpSegments : undefined,
      windArrows: !isDifference && showWindArrows ? windArrows : undefined,
      diffWindArrows: isDifference && windFlowEnabled ? diffWindArrows : undefined,
      shockConfig,
      showShockOverlay,
      frontContourSegments,
      localityRingSegments,
      refRingSegments,
      stepDayNumber,
      hoveredCell: activeHover,
      drawGraticule: true,
    })
  }, [
    grid,
    scale,
    palette,
    landSeaMask,
    coastlineSegments,
    isDifference,
    windFlowEnabled,
    diffWindArrows,
    showTpThresholdOutline,
    tpSegments,
    showWindArrows,
    windArrows,
    shockConfig,
    showShockOverlay,
    frontContourSegments,
    localityRingSegments,
    refRingSegments,
    stepDayNumber,
    activeHover,
  ])

  useEffect(() => {
    paintCanvas()
  }, [paintCanvas])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(() => {
      paintCanvas()
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [paintCanvas])

  // Snap pointer strictly to the nearest original 1-degree (32x36) grid cell
  const pointerToCell = (e: ReactPointerEvent<HTMLDivElement>): HoveredCell | null => {
    const el = containerRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = (e.clientX - rect.left) / rect.width
    const ny = (e.clientY - rect.top) / rect.height
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null
    const col = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(nx * GRID_COLS)))
    const visualY = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(ny * GRID_ROWS)))
    const row = GRID_ROWS - 1 - visualY
    return { row, col }
  }

  const updateHover = (cell: HoveredCell | null) => {
    if (onHoverCell) {
      onHoverCell(cell)
    } else {
      setLocalHover(cell)
    }
  }

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!allowShockPlacement) return
    const cell = pointerToCell(e)
    if (!cell) return
    e.currentTarget.setPointerCapture(e.pointerId)
    dragStartRef.current = cell
    didDragRef.current = false
  }

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const cell = pointerToCell(e)
    setIsPointerInside(true)
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
    updateHover(cell)

    if (allowShockPlacement && dragStartRef.current && cell) {
      const start = dragStartRef.current
      if (start.row !== cell.row || start.col !== cell.col) {
        didDragRef.current = true
        onShockDrag?.(start.row, start.col, cell.row, cell.col)
      }
    }
  }

  const handlePointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!allowShockPlacement || !dragStartRef.current) return
    const cell = pointerToCell(e) ?? dragStartRef.current
    if (didDragRef.current) {
      onShockDrag?.(dragStartRef.current.row, dragStartRef.current.col, cell.row, cell.col)
    } else {
      onShockClick?.(cell.row, cell.col)
    }
    dragStartRef.current = null
    didDragRef.current = false
  }

  const handlePointerLeave = () => {
    setIsPointerInside(false)
    setTooltipPos(null)
    if (!dragStartRef.current) {
      updateHover(null)
    }
  }

  const topRuleColor =
    badgeVariant === 'prime'
      ? '#7FB5B0'
      : badgeVariant === 'branch'
        ? '#F0B84B'
        : '#26312B'

  // Exact, un-interpolated physical value from the original 32x36 grid cell [row][col]
  const hoverReadout = useMemo(() => {
    if (!activeHover || !grid) return null
    const { row, col } = activeHover
    const val = grid[row]?.[col]
    if (val === undefined) return null
    const lat = rowToLat(row)
    const lon = colToLon(col)
    const sign = isDifference && val > 0 ? '+' : ''
    const formatted = `${sign}${val.toFixed(decimals)} ${unit}`

    let wireNote: string | null = null
    if (channel === 'vorticity') {
      wireNote = 'computed'
    } else if (!isDifference && physicalState && channel) {
      if (channel === 't2m') {
        wireNote = `${physicalState[0][row][col].toFixed(2)} K`
      } else if (channel === 'msl') {
        wireNote = `${Math.round(physicalState[4][row][col])} Pa`
      } else if (channel === 'wind') {
        const u = physicalState[2][row][col]
        const v = physicalState[3][row][col]
        wireNote = `u=${u.toFixed(1)}, v=${v.toFixed(1)}`
      }
    }

    const diffVal = compareGrids?.diff?.[row]?.[col]
    const diffNote =
      !isDifference && diffVal !== undefined
        ? `Δ ${diffVal > 0 ? '+' : ''}${diffVal.toFixed(
            Math.abs(diffVal) > 0 && Math.abs(diffVal) < 0.1 ? 3 : decimals,
          )} ${unit}`
        : null

    return {
      row,
      col,
      lat,
      lon,
      formatted,
      wireNote,
      diffNote,
    }
  }, [activeHover, grid, decimals, unit, isDifference, physicalState, compareGrids, channel])

  const formatTick = (v: number) => {
    if (Math.abs(v) < 1e-5) return '0'
    const sign = scale.diverging && v > 0 ? '+' : ''
    if (Math.abs(v) < 0.1) return `${sign}${v.toFixed(3)}`
    return `${sign}${v.toFixed(decimals)}`
  }

  return (
    <div
      className="panel-glass flex flex-col pt-2.5"
      style={{ borderTop: `1px solid ${topRuleColor}` }}
    >
      {/* Header: Serif italic label + quiet mono readout */}
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <h3 className="font-serif-display shrink-0 text-xl italic leading-none text-[#ECE7DA]">
            {title}
          </h3>
          {subtitle && (
            <p className="truncate font-mono text-[11px] text-[#8C958D]">{subtitle}</p>
          )}
        </div>

        {hoverReadout && (
          <p className="shrink-0 font-mono text-[11px] text-[#ECE7DA] tabular-nums">
            <span>{hoverReadout.formatted}</span>
            {hoverReadout.diffNote && (
              <span className="ml-1.5 text-[#8C958D]">({hoverReadout.diffNote})</span>
            )}
            <span className="ml-2 text-[#5A645D]">
              [{hoverReadout.row},{hoverReadout.col}]
            </span>
          </p>
        )}
      </div>

      {/* Map body with 5-degree Lat/Lon graticule labels in mono along edges (North Up: row 0 = south) */}
      <div className="grid grid-cols-[36px_1fr] gap-x-1.5">
        <div className="relative select-none">
          {GRATICULE_LATS.map((lat) => {
            const topPct = latToNormY(lat) * 100
            return (
              <span
                key={lat}
                style={{ top: `${topPct}%` }}
                className="tabular-nums absolute right-0 -translate-y-1/2 font-mono text-[10px] text-[#5A645D]"
              >
                {lat}°N
              </span>
            )
          })}
        </div>

        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerLeave}
          className={`relative aspect-[36/32] w-full overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18] ${
            allowShockPlacement ? 'cursor-crosshair select-none touch-none' : 'cursor-default'
          }`}
        >
          {grid ? (
            <>
              <canvas ref={displayCanvasRef} className="block h-full w-full" />
              <WindFlowCanvas
                active={windFlowEnabled && !isDifference}
                primaryState={physicalState}
              />

              {/* Small mono tooltip snapped to exact un-interpolated 1° grid cell */}
              {isPointerInside && tooltipPos && hoverReadout && (
                <div
                  style={{
                    left: Math.min(
                      Math.max(6, tooltipPos.x + 12),
                      (containerRef.current?.clientWidth ?? 240) - 178,
                    ),
                    top: tooltipPos.y > 76 ? tooltipPos.y - 68 : tooltipPos.y + 12,
                  }}
                  className="tooltip-float pointer-events-none absolute z-20 w-[170px] rounded-[3px] px-2.5 py-1.5 font-mono text-[11px] tabular-nums"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-medium text-[#ECE7DA]">{hoverReadout.formatted}</span>
                    {hoverReadout.wireNote && (
                      <span className="text-[10px] text-[#8C958D]">{hoverReadout.wireNote}</span>
                    )}
                  </div>
                  <div className="mt-0.5 text-[10px] text-[#8C958D]">
                    row {hoverReadout.row}, col {hoverReadout.col}
                  </div>
                  <div className="text-[10px] text-[#5A645D]">
                    {formatCoordLat(hoverReadout.lat)}, {formatCoordLon(hoverReadout.lon)}
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

        <div className="relative mt-1 h-4 select-none">
          {GRATICULE_LONS.map((lon) => {
            const leftPct = lonToNormX(lon) * 100
            const alignClass = lon === 100 ? '-translate-x-[82%]' : '-translate-x-1/2'
            return (
              <span
                key={lon}
                style={{ left: `${leftPct}%` }}
                className={`tabular-nums absolute top-0 font-mono text-[10px] text-[#5A645D] ${alignClass}`}
              >
                {lon}°E
              </span>
            )
          })}
        </div>
      </div>

      {/* Minimal scale bar */}
      <div className="mt-1.5 pl-[42px]">
        <div
          className="h-1.5 w-full rounded-[1px] border border-[#26312B]"
          style={{ background: paletteToCssGradient(palette) }}
        />
        <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-[#8C958D] tabular-nums">
          <span>{formatTick(scale.min)}</span>
          <span className="text-[#5A645D]">
            {scale.diverging ? `0 ${unit}` : `${formatTick((scale.min + scale.max) / 2)} ${unit}`}
          </span>
          <span>{formatTick(scale.max)}</span>
        </div>
      </div>
    </div>
  )
}
