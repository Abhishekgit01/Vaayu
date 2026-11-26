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
  type DisplayChannelId,
  type Grid2D,
  type State3D,
} from '../api/types'
import {
  paletteToCssGradient,
  type ColorScale,
  type PaletteKind,
} from '../lib/colormaps'
import {
  colToLon,
  formatCoordLat,
  formatCoordLon,
  getDisplayChannelMeta,
  rowToLat,
} from '../lib/constants'
import {
  drawCanvasMapOverlays,
  GRATICULE_LATS,
  GRATICULE_LONS,
  latToNormY,
  lonToNormX,
  renderHeatmapToBuffer,
  type ContourSegment,
  type WindArrow,
} from '../lib/heatmap'
import type { ShockConfig } from '../lib/shock'

export interface HoveredCell {
  row: number
  col: number
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
  channel?: DisplayChannelId
  palette: PaletteKind
  scale: ColorScale
  unit: string
  decimals?: number
  smoothInterpolation?: boolean
  isDifference?: boolean
  showWindArrows?: boolean
  windArrows?: WindArrow[]
  showTpThresholdOutline?: boolean
  tpSegments?: ContourSegment[]
  shockConfig?: ShockConfig | null
  showShockOverlay?: boolean
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
  smoothInterpolation = false,
  isDifference = false,
  showWindArrows = false,
  windArrows,
  showTpThresholdOutline = false,
  tpSegments,
  shockConfig,
  showShockOverlay = false,
  allowShockPlacement = false,
  onShockClick,
  onShockDrag,
  hoveredCell,
  onHoverCell,
  emptyMessage = 'Run simulation to populate map',
}: GridMapProps) {
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const bufferCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const dragStartRef = useRef<HoveredCell | null>(null)
  const didDragRef = useRef(false)
  const [localHover, setLocalHover] = useState<HoveredCell | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [isPointerInside, setIsPointerInside] = useState(false)
  const [bufferTick, setBufferTick] = useState(0)

  const activeHover = hoveredCell !== undefined ? hoveredCell : localHover

  // Re-rasterize offscreen heatmap buffer ONLY when grid, scale, palette, or smooth mode changes
  useEffect(() => {
    if (!grid) return
    if (!bufferCanvasRef.current) {
      bufferCanvasRef.current = document.createElement('canvas')
    }
    renderHeatmapToBuffer(bufferCanvasRef.current, grid, scale, palette, smoothInterpolation)
    setBufferTick((t) => t + 1)
  }, [grid, scale, palette, smoothInterpolation])

  const paintCanvas = useCallback(() => {
    const canvas = displayCanvasRef.current
    const container = containerRef.current
    const buffer = bufferCanvasRef.current
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

    if (!grid || !buffer) return

    ctx.save()
    ctx.imageSmoothingEnabled = smoothInterpolation
    if (smoothInterpolation) {
      ctx.imageSmoothingQuality = 'high'
    }
    ctx.drawImage(buffer, 0, 0, targetW, targetH)
    ctx.restore()

    drawCanvasMapOverlays(ctx, {
      width: targetW,
      height: targetH,
      dpr,
      tpSegments: !isDifference && showTpThresholdOutline ? tpSegments : undefined,
      windArrows: !isDifference && showWindArrows ? windArrows : undefined,
      shockConfig,
      showShockOverlay,
      hoveredCell: activeHover,
      drawGraticule: true,
    })
  }, [
    grid,
    smoothInterpolation,
    isDifference,
    showTpThresholdOutline,
    tpSegments,
    showWindArrows,
    windArrows,
    shockConfig,
    showShockOverlay,
    activeHover,
  ])

  useEffect(() => {
    paintCanvas()
  }, [paintCanvas, bufferTick])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(() => {
      paintCanvas()
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [paintCanvas])

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
    if (!isDifference && physicalState && channel) {
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
      channelLabel: channel ? getDisplayChannelMeta(channel).shortLabel : '',
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

              {/* Small mono tooltip with value, row, col */}
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
