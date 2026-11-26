import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { GripVertical } from 'lucide-react'
import {
  GRID_COLS,
  GRID_ROWS,
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
  rowToLat,
} from '../lib/constants'
import {
  drawCanvasMapOverlays,
  drawFieldVectorsAndContours,
  renderHeatmapToBuffer,
  type ContourSegment,
  type WindArrow,
} from '../lib/heatmap'
import type { ShockConfig } from '../lib/shock'
import {
  DEFAULT_MAP_VIEWPORT,
  getViewportBounds,
  getViewportGraticuleTicks,
  panViewportByNormDelta,
  viewportNormToCell,
  zoomViewportAtNorm,
  type MapViewport,
} from '../lib/viewport'
import type { ViewFieldId } from '../lib/vorticity'
import type { HoveredCell } from './GridMap'
import { WindFlowCanvas } from './WindFlowCanvas'

interface SplitCompareMapProps {
  primeGrid: Grid2D | null
  branchGrid: Grid2D | null
  diffGrid: Grid2D | null
  primeState?: State3D | null
  branchState?: State3D | null
  channel: ViewFieldId
  channelLabel: string
  branchLabel: string
  palette: PaletteKind
  scale: ColorScale
  unit: string
  decimals: number
  smoothInterpolation: boolean
  landSeaMask?: Grid2D | null
  coastlineSegments?: ContourSegment[]
  windFlowEnabled?: boolean
  splitRatio: number
  onSplitRatioChange: (ratio: number) => void
  showWindArrows: boolean
  primeWindArrows?: WindArrow[]
  branchWindArrows?: WindArrow[]
  showTpThresholdOutline: boolean
  primeTpSegments?: ContourSegment[]
  branchTpSegments?: ContourSegment[]
  shockConfig: ShockConfig
  showShockOverlay: boolean
  frontContourSegments?: ContourSegment[]
  localityRingSegments?: ContourSegment[]
  refRingSegments?: ContourSegment[]
  stepDayNumber?: number
  viewport?: MapViewport
  onViewportChange?: (next: MapViewport) => void
  onShockClick: (row: number, col: number) => void
  onShockDrag: (startRow: number, startCol: number, endRow: number, endCol: number) => void
  hoveredCell: HoveredCell | null
  onHoverCell: (cell: HoveredCell | null) => void
}

export function SplitCompareMap({
  primeGrid,
  branchGrid,
  diffGrid,
  primeState,
  branchState,
  channelLabel,
  branchLabel,
  palette,
  scale,
  unit,
  decimals,
  smoothInterpolation,
  landSeaMask,
  coastlineSegments,
  windFlowEnabled = false,
  splitRatio,
  onSplitRatioChange,
  showWindArrows,
  primeWindArrows,
  branchWindArrows,
  showTpThresholdOutline,
  primeTpSegments,
  branchTpSegments,
  shockConfig,
  showShockOverlay,
  frontContourSegments,
  localityRingSegments,
  refRingSegments,
  stepDayNumber,
  viewport,
  onViewportChange,
  onShockClick,
  onShockDrag,
  hoveredCell,
  onHoverCell,
}: SplitCompareMapProps) {
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const primeBufferRef = useRef<HTMLCanvasElement | null>(null)
  const branchBufferRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const shockDragStartRef = useRef<HoveredCell | null>(null)
  const didShockDragRef = useRef(false)
  const isDraggingSplitRef = useRef(false)
  const panStartRef = useRef<{ clientX: number; clientY: number; viewport: MapViewport } | null>(null)

  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [isPointerInside, setIsPointerInside] = useState(false)
  const [bufferTick, setBufferTick] = useState(0)
  const [isPanning, setIsPanning] = useState(false)

  const activeViewport = viewport ?? DEFAULT_MAP_VIEWPORT
  const viewportBounds = useMemo(() => getViewportBounds(activeViewport), [activeViewport])
  const graticuleTicks = useMemo(
    () => getViewportGraticuleTicks(viewportBounds),
    [viewportBounds],
  )
  const latestViewportRef = useRef(activeViewport)
  latestViewportRef.current = activeViewport
  const onViewportChangeRef = useRef(onViewportChange)
  onViewportChangeRef.current = onViewportChange

  useEffect(() => {
    if (!primeGrid) return
    if (!primeBufferRef.current) {
      primeBufferRef.current = document.createElement('canvas')
    }
    renderHeatmapToBuffer(
      primeBufferRef.current,
      primeGrid,
      scale,
      palette,
      smoothInterpolation,
      landSeaMask,
    )
    setBufferTick((t) => t + 1)
  }, [primeGrid, scale, palette, smoothInterpolation, landSeaMask])

  useEffect(() => {
    if (!branchGrid) return
    if (!branchBufferRef.current) {
      branchBufferRef.current = document.createElement('canvas')
    }
    renderHeatmapToBuffer(
      branchBufferRef.current,
      branchGrid,
      scale,
      palette,
      smoothInterpolation,
      landSeaMask,
    )
    setBufferTick((t) => t + 1)
  }, [branchGrid, scale, palette, smoothInterpolation, landSeaMask])

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
    const splitX = Math.round(splitRatio * targetW)

    // Left half: Timeline Prime
    ctx.save()
    ctx.beginPath()
    ctx.rect(0, 0, splitX, targetH)
    ctx.clip()
    if (primeGrid && primeBufferRef.current) {
      const pBuf = primeBufferRef.current
      const sx = (viewportBounds.gxMin / GRID_COLS) * pBuf.width
      const sy = (viewportBounds.gyMin / GRID_ROWS) * pBuf.height
      const sw = (viewportBounds.spanGx / GRID_COLS) * pBuf.width
      const sh = (viewportBounds.spanGy / GRID_ROWS) * pBuf.height
      ctx.imageSmoothingEnabled = smoothInterpolation
      if (smoothInterpolation) ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(pBuf, sx, sy, sw, sh, 0, 0, targetW, targetH)
      drawFieldVectorsAndContours(
        ctx,
        targetW,
        targetH,
        dpr,
        showTpThresholdOutline ? primeTpSegments : undefined,
        showWindArrows ? primeWindArrows : undefined,
        viewportBounds,
      )
    } else {
      ctx.fillStyle = '#141B18'
      ctx.fillRect(0, 0, splitX, targetH)
    }
    ctx.restore()

    // Right half: Branch
    ctx.save()
    ctx.beginPath()
    ctx.rect(splitX, 0, targetW - splitX, targetH)
    ctx.clip()
    if (branchGrid && branchBufferRef.current) {
      const bBuf = branchBufferRef.current
      const sx = (viewportBounds.gxMin / GRID_COLS) * bBuf.width
      const sy = (viewportBounds.gyMin / GRID_ROWS) * bBuf.height
      const sw = (viewportBounds.spanGx / GRID_COLS) * bBuf.width
      const sh = (viewportBounds.spanGy / GRID_ROWS) * bBuf.height
      ctx.imageSmoothingEnabled = smoothInterpolation
      if (smoothInterpolation) ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(bBuf, sx, sy, sw, sh, 0, 0, targetW, targetH)
      drawFieldVectorsAndContours(
        ctx,
        targetW,
        targetH,
        dpr,
        showTpThresholdOutline ? branchTpSegments : undefined,
        showWindArrows ? branchWindArrows : undefined,
        viewportBounds,
      )
    } else {
      ctx.fillStyle = '#141B18'
      ctx.fillRect(splitX, 0, targetW - splitX, targetH)
      if (targetW - splitX > 130 * dpr) {
        ctx.fillStyle = '#8C958D'
        ctx.font = `400 ${Math.round(11 * dpr)}px 'Geist Mono', ui-monospace, monospace`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(
          'Run Branch to populate right curtain',
          splitX + (targetW - splitX) / 2,
          targetH / 2,
        )
      }
    }
    ctx.restore()

    // Shared graticules, shock region, and hover reticle
    drawCanvasMapOverlays(ctx, {
      width: targetW,
      height: targetH,
      dpr,
      isLightBackground: scale.diverging,
      viewportBounds,
      landSeaMask,
      coastlineSegments,
      shockConfig,
      showShockOverlay,
      frontContourSegments,
      localityRingSegments,
      refRingSegments,
      stepDayNumber,
      hoveredCell,
      drawGraticule: true,
    })

    // Crisp 1px vertical divider line
    ctx.save()
    ctx.strokeStyle = '#ECE7DA'
    ctx.lineWidth = 1.25 * dpr
    ctx.beginPath()
    ctx.moveTo(splitX + 0.5, 0)
    ctx.lineTo(splitX + 0.5, targetH)
    ctx.stroke()
    ctx.restore()
  }, [
    primeGrid,
    branchGrid,
    scale.diverging,
    smoothInterpolation,
    coastlineSegments,
    splitRatio,
    showTpThresholdOutline,
    primeTpSegments,
    branchTpSegments,
    showWindArrows,
    primeWindArrows,
    branchWindArrows,
    shockConfig,
    showShockOverlay,
    frontContourSegments,
    localityRingSegments,
    refRingSegments,
    stepDayNumber,
    hoveredCell,
    viewportBounds,
    landSeaMask,
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

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      const changeCb = onViewportChangeRef.current
      if (!changeCb) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return
      const nx = (e.clientX - rect.left) / rect.width
      const ny = (e.clientY - rect.top) / rect.height
      const cur = latestViewportRef.current
      const factor = e.deltaY < 0 ? 1.25 : 0.8
      changeCb(zoomViewportAtNorm(cur, cur.zoom * factor, nx, ny))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const pointerToCell = (e: ReactPointerEvent<HTMLElement>): HoveredCell | null => {
    const el = containerRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = (e.clientX - rect.left) / rect.width
    const ny = (e.clientY - rect.top) / rect.height
    return viewportNormToCell(nx, ny, viewportBounds)
  }

  const handleSplitPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.stopPropagation()
    e.currentTarget.setPointerCapture(e.pointerId)
    isDraggingSplitRef.current = true
  }

  const handleSplitPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDraggingSplitRef.current || !containerRef.current) return
    e.stopPropagation()
    const rect = containerRef.current.getBoundingClientRect()
    if (rect.width <= 0) return
    const next = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / rect.width))
    onSplitRatioChange(next)
  }

  const handleSplitPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDraggingSplitRef.current) return
    e.stopPropagation()
    isDraggingSplitRef.current = false
  }

  const handleSplitKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      e.stopPropagation()
      onSplitRatioChange(Math.max(0.02, splitRatio - 0.04))
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      e.stopPropagation()
      onSplitRatioChange(Math.min(0.98, splitRatio + 0.04))
    }
  }

  const handleMapPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (Boolean(onViewportChange) && activeViewport.zoom > 1.001) {
      e.preventDefault()
      e.currentTarget.setPointerCapture(e.pointerId)
      panStartRef.current = {
        clientX: e.clientX,
        clientY: e.clientY,
        viewport: activeViewport,
      }
      setIsPanning(true)
      return
    }
    const cell = pointerToCell(e)
    if (!cell) return
    e.currentTarget.setPointerCapture(e.pointerId)
    shockDragStartRef.current = cell
    didShockDragRef.current = false
  }

  const handleMapPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (panStartRef.current && onViewportChange && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        const dnx = (e.clientX - panStartRef.current.clientX) / rect.width
        const dny = (e.clientY - panStartRef.current.clientY) / rect.height
        onViewportChange(panViewportByNormDelta(panStartRef.current.viewport, dnx, dny))
      }
      return
    }

    const cell = pointerToCell(e)
    setIsPointerInside(true)
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
    onHoverCell(cell)

    if (activeViewport.zoom <= 1.001 && shockDragStartRef.current && cell) {
      const start = shockDragStartRef.current
      if (start.row !== cell.row || start.col !== cell.col) {
        didShockDragRef.current = true
        onShockDrag(start.row, start.col, cell.row, cell.col)
      }
    }
  }

  const handleMapPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (panStartRef.current) {
      panStartRef.current = null
      setIsPanning(false)
      return
    }
    if (!shockDragStartRef.current) return
    const cell = pointerToCell(e) ?? shockDragStartRef.current
    if (didShockDragRef.current) {
      onShockDrag(shockDragStartRef.current.row, shockDragStartRef.current.col, cell.row, cell.col)
    } else {
      onShockClick(cell.row, cell.col)
    }
    shockDragStartRef.current = null
    didShockDragRef.current = false
  }

  const handleMapPointerLeave = () => {
    setIsPointerInside(false)
    setTooltipPos(null)
    if (!shockDragStartRef.current && !panStartRef.current) {
      onHoverCell(null)
    }
  }

  const hoverReadout = useMemo(() => {
    if (!hoveredCell || !primeGrid) return null
    const { row, col } = hoveredCell
    const pVal = primeGrid[row]?.[col]
    if (pVal === undefined) return null
    const bVal = branchGrid?.[row]?.[col]
    const dVal = diffGrid?.[row]?.[col]
    const lat = rowToLat(row)
    const lon = colToLon(col)

    return {
      row,
      col,
      lat,
      lon,
      primeText: `${pVal.toFixed(decimals)} ${unit}`,
      branchText: bVal !== undefined ? `${bVal.toFixed(decimals)} ${unit}` : '—',
      diffText:
        dVal !== undefined
          ? `${dVal > 0 ? '+' : ''}${dVal.toFixed(
              Math.abs(dVal) > 0 && Math.abs(dVal) < 0.1 ? 3 : decimals,
            )} ${unit}`
          : '—',
    }
  }, [hoveredCell, primeGrid, branchGrid, diffGrid, decimals, unit])

  const formatTick = (v: number) => {
    if (Math.abs(v) < 1e-5) return '0'
    const sign = scale.diverging && v > 0 ? '+' : ''
    if (Math.abs(v) < 0.1) return `${sign}${v.toFixed(3)}`
    return `${sign}${v.toFixed(decimals)}`
  }

  return (
    <div className="flex flex-col">
      {/* Dual rain/lightning top rule */}
      <div className="mb-2.5 flex h-[1px] w-full">
        <div style={{ width: `${splitRatio * 100}%` }} className="bg-[#7FB5B0]" />
        <div style={{ width: `${(1 - splitRatio) * 100}%` }} className="bg-[#F0B84B]" />
      </div>

      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-3">
          <h3 className="font-serif-display text-xl italic leading-none text-[#ECE7DA]">
            <span className="text-[#7FB5B0]">Timeline Prime</span>
            <span className="mx-2 font-sans text-xs not-italic text-[#5A645D]">/</span>
            <span className="text-[#F0B84B]">Branch</span>
          </h3>
          <span className="font-mono text-[11px] text-[#8C958D]">
            {channelLabel} · {branchLabel}
          </span>
          {Math.abs(splitRatio - 0.5) > 0.01 && (
            <button
              type="button"
              onClick={() => onSplitRatioChange(0.5)}
              className="ui-transition font-mono text-[11px] text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
            >
              Center 50%
            </button>
          )}
        </div>

        {hoverReadout && (
          <p className="font-mono text-[11px] text-[#ECE7DA] tabular-nums">
            <span className="text-[#7FB5B0]">P</span> {hoverReadout.primeText} ·{' '}
            <span className="text-[#F0B84B]">B</span> {hoverReadout.branchText} ·{' '}
            <span className="text-[#8C958D]">Δ {hoverReadout.diffText}</span>{' '}
            <span className="text-[#5A645D]">
              [{hoverReadout.row},{hoverReadout.col}]
            </span>
          </p>
        )}
      </div>

      <div className="grid grid-cols-[36px_1fr] gap-x-1.5">
        <div className="relative select-none">
          {graticuleTicks.lats.map((tick) => (
            <span
              key={tick.value}
              style={{ top: `${tick.normPos * 100}%` }}
              className="tabular-nums absolute right-0 -translate-y-1/2 font-mono text-[10px] text-[#5A645D]"
            >
              {tick.value}°N
            </span>
          ))}
        </div>

        <div
          ref={containerRef}
          onPointerDown={handleMapPointerDown}
          onPointerMove={handleMapPointerMove}
          onPointerUp={handleMapPointerUp}
          onPointerLeave={handleMapPointerLeave}
          className={`relative aspect-[36/32] w-full select-none overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18] touch-none ${
            isPanning
              ? 'cursor-grabbing'
              : activeViewport.zoom > 1.001
                ? 'cursor-grab'
                : 'cursor-crosshair'
          }`}
        >
          {primeGrid ? (
            <>
              <canvas ref={displayCanvasRef} className="block h-full w-full" />
              <WindFlowCanvas
                active={windFlowEnabled}
                primaryState={primeState}
                secondaryState={branchState}
                splitRatio={splitRatio}
                viewportBounds={viewportBounds}
              />

              {/* Draggable rectangular instrument split handle */}
              <div
                data-split-handle="true"
                role="slider"
                tabIndex={0}
                aria-label="Prime vs Branch split slider"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(splitRatio * 100)}
                style={{ left: `${splitRatio * 100}%` }}
                onPointerDown={handleSplitPointerDown}
                onPointerMove={handleSplitPointerMove}
                onPointerUp={handleSplitPointerUp}
                onKeyDown={handleSplitKeyDown}
                className="absolute top-0 bottom-0 z-10 flex w-6 -translate-x-1/2 cursor-ew-resize items-center justify-center"
              >
                <div className="flex h-8 w-4 items-center justify-center rounded-[2px] border border-[#ECE7DA] bg-[#0D1210]">
                  <GripVertical className="h-3.5 w-3.5 text-[#ECE7DA]" strokeWidth={1.5} />
                </div>
              </div>

              {isPointerInside && tooltipPos && hoverReadout && (
                <div
                  style={{
                    left: Math.min(
                      Math.max(6, tooltipPos.x + 12),
                      (containerRef.current?.clientWidth ?? 260) - 196,
                    ),
                    top: tooltipPos.y > 88 ? tooltipPos.y - 82 : tooltipPos.y + 12,
                  }}
                  className="tooltip-float pointer-events-none absolute z-20 w-[188px] rounded-[3px] px-2.5 py-1.5 font-mono text-[11px] tabular-nums"
                >
                  <div className="flex justify-between">
                    <span className="text-[#7FB5B0]">Prime</span>
                    <span className="text-[#ECE7DA]">{hoverReadout.primeText}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#F0B84B]">Branch</span>
                    <span className="text-[#ECE7DA]">{hoverReadout.branchText}</span>
                  </div>
                  <div className="flex justify-between border-b border-[#26312B] pb-1">
                    <span className="text-[#8C958D]">Diff</span>
                    <span className="text-[#ECE7DA]">{hoverReadout.diffText}</span>
                  </div>
                  <div className="mt-1 text-[10px] text-[#8C958D]">
                    row {hoverReadout.row}, col {hoverReadout.col} ·{' '}
                    {formatCoordLat(hoverReadout.lat)}, {formatCoordLon(hoverReadout.lon)}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-xs text-[#5A645D]">
              Simulating Timeline Prime…
            </div>
          )}
        </div>

        <div />

        <div className="relative mt-1 h-4 select-none">
          {graticuleTicks.lons.map((tick) => {
            const leftPct = tick.normPos * 100
            const alignClass = leftPct > 92 ? '-translate-x-[82%]' : '-translate-x-1/2'
            return (
              <span
                key={tick.value}
                style={{ left: `${leftPct}%` }}
                className={`tabular-nums absolute top-0 font-mono text-[10px] text-[#5A645D] ${alignClass}`}
              >
                {tick.value}°E
              </span>
            )
          })}
        </div>
      </div>

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
