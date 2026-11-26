import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { createPortal } from 'react-dom'
import { RotateCcw } from 'lucide-react'
import { GRID_COLS, GRID_ROWS, type ChannelKey, type State3D } from '../api/types'
import { CHANNELS, colToLon, rowToLat } from '../lib/constants'
import {
  applyDragToShock,
  computeShockPreview,
  createEmptyShockConfig,
  createWarmBayOfBengalPreset,
  createWeakerArabianSeaWindPreset,
  extractIsoContourSegments,
  getCachedLastInputDay,
  getGaussianStrokePoints,
  moveShockCenter,
  subscribeLastInputDay,
  type AmplitudeMode,
  type ShockConfig,
  type ShockShape,
} from '../lib/shock'

interface ShockPanelProps {
  config: ShockConfig
  onChange: (next: ShockConfig) => void
  onRun: () => void
  onRunZeroShock: () => void
  isSimulating: boolean
  hasPrime: boolean
}

function PaintedGaussianMapOverlay({
  host,
  weights,
}: {
  host: HTMLElement
  weights: number[][]
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const contours = useMemo(() => {
    const nx = GRID_COLS * 2
    const ny = GRID_ROWS * 2
    const field: number[][] = Array.from({ length: ny + 1 }, (_, iy) => {
      const gy = (iy / ny) * GRID_ROWS
      const gr = Math.max(0, Math.min(GRID_ROWS - 1, GRID_ROWS - 1 - (gy - 0.5)))
      const r0 = Math.floor(gr)
      const r1 = Math.min(GRID_ROWS - 1, r0 + 1)
      const fr = gr - r0
      return Array.from({ length: nx + 1 }, (__, ix) => {
        const gx = (ix / nx) * GRID_COLS
        const gc = Math.max(0, Math.min(GRID_COLS - 1, gx - 0.5))
        const c0 = Math.floor(gc)
        const c1 = Math.min(GRID_COLS - 1, c0 + 1)
        const fc = gc - c0
        const v0 = weights[r0][c0] + (weights[r0][c1] - weights[r0][c0]) * fc
        const v1 = weights[r1][c0] + (weights[r1][c1] - weights[r1][c0]) * fc
        return v0 + (v1 - v0) * fr
      })
    })
    return {
      core: extractIsoContourSegments(field, nx, ny, Math.exp(-0.5)),
      outer: extractIsoContourSegments(field, nx, ny, 0.08),
    }
  }, [weights])

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

    const toX = (gx: number) => (gx / GRID_COLS) * w
    const toY = (gy: number) => (gy / GRID_ROWS) * h

    if (contours.outer.length > 0) {
      ctx.save()
      ctx.setLineDash([4 * dpr, 3 * dpr])
      ctx.strokeStyle = 'rgba(240, 184, 75, 0.65)'
      ctx.lineWidth = 1.2 * dpr
      ctx.beginPath()
      for (const [x1, y1, x2, y2] of contours.outer) {
        ctx.moveTo(toX(x1), toY(y1))
        ctx.lineTo(toX(x2), toY(y2))
      }
      ctx.stroke()
      ctx.restore()
    }

    if (contours.core.length > 0) {
      ctx.save()
      ctx.strokeStyle = 'rgba(13, 18, 16, 0.72)'
      ctx.lineWidth = 2.6 * dpr
      ctx.beginPath()
      for (const [x1, y1, x2, y2] of contours.core) {
        ctx.moveTo(toX(x1), toY(y1))
        ctx.lineTo(toX(x2), toY(y2))
      }
      ctx.stroke()

      ctx.strokeStyle = '#F0B84B'
      ctx.lineWidth = 1.4 * dpr
      ctx.stroke()
      ctx.restore()
    }
  }, [host, contours])

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
      className="pointer-events-none absolute inset-0 z-[5] block h-full w-full"
    />,
    host,
  )
}

export function ShockPanel({
  config,
  onChange,
  onRun,
  onRunZeroShock,
  isSimulating,
  hasPrime,
}: ShockPanelProps) {
  const [lastInputDay, setLastInputDay] = useState<State3D | null>(() => getCachedLastInputDay())
  const [interactiveMapHosts, setInteractiveMapHosts] = useState<HTMLElement[]>([])

  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const previewBoxRef = useRef<HTMLDivElement | null>(null)
  const dragOriginRef = useRef<{ row: number; col: number } | null>(null)
  const didDragRef = useRef(false)
  const latestConfigRef = useRef(config)
  latestConfigRef.current = config

  useEffect(() => {
    return subscribeLastInputDay(() => {
      setLastInputDay(getCachedLastInputDay())
    })
  }, [])

  const strokePoints = getGaussianStrokePoints(config)
  const hasPaintedTrail = config.shape === 'gaussian' && strokePoints.length > 1

  useEffect(() => {
    if (!hasPaintedTrail) {
      setInteractiveMapHosts((prev) => (prev.length === 0 ? prev : []))
      return
    }

    const syncHosts = () => {
      const nodes = Array.from(
        document.querySelectorAll<HTMLElement>('div.cursor-crosshair.aspect-\\[36\\/32\\]'),
      ).filter((el) => el !== previewBoxRef.current && el.querySelector('canvas'))
      setInteractiveMapHosts((prev) => {
        if (prev.length === nodes.length && prev.every((el, i) => el === nodes[i])) {
          return prev
        }
        return nodes
      })
    }

    syncHosts()
    const obs = new MutationObserver(syncHosts)
    obs.observe(document.body, { childList: true, subtree: true })
    return () => obs.disconnect()
  }, [hasPaintedTrail])

  const preview = useMemo(
    () => computeShockPreview(config, lastInputDay),
    [config, lastInputDay],
  )

  // Live north-up 36x32 footprint preview canvas
  useEffect(() => {
    const canvas = previewCanvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = GRID_COLS * 6 * dpr
    const h = GRID_ROWS * 6 * dpr
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cellW = w / GRID_COLS
    const cellH = h / GRID_ROWS

    ctx.fillStyle = '#141B18'
    ctx.fillRect(0, 0, w, h)

    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      for (let c = 0; c < GRID_COLS; c++) {
        const weight = preview.weights[r][c]
        if (weight <= 0) continue
        // Interpolate surface #141B18 -> lightning #F0B84B
        const red = Math.round(20 + 220 * weight)
        const green = Math.round(27 + 157 * weight)
        const blue = Math.round(24 + 51 * weight)
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        ctx.fillRect(c * cellW, vy * cellH, Math.ceil(cellW), Math.ceil(cellH))
      }
    }

    // Crisp 1px graticule lines
    ctx.strokeStyle = 'rgba(90, 100, 93, 0.35)'
    ctx.lineWidth = 1 * dpr
    ctx.beginPath()
    for (const lat of [10, 20, 30]) {
      const y = Math.round(((38 - lat) / GRID_ROWS) * h) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    for (const lon of [70, 80, 90, 100]) {
      const x = Math.round(((lon - 65) / GRID_COLS) * w) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    ctx.stroke()

    if (config.shape === 'box') {
      const rMin = Math.min(config.boxRowMin, config.boxRowMax)
      const rMax = Math.max(config.boxRowMin, config.boxRowMax)
      const cMin = Math.min(config.boxColMin, config.boxColMax)
      const cMax = Math.max(config.boxColMin, config.boxColMax)
      ctx.strokeStyle = '#ECE7DA'
      ctx.lineWidth = 1.2 * dpr
      ctx.strokeRect(
        cMin * cellW,
        (GRID_ROWS - 1 - rMax) * cellH,
        (cMax - cMin + 1) * cellW,
        (rMax - rMin + 1) * cellH,
      )
    }
  }, [preview.weights, config.shape, config.boxRowMin, config.boxRowMax, config.boxColMin, config.boxColMax])

  const boxHeight = Math.abs(config.boxRowMax - config.boxRowMin) + 1
  const boxWidth = Math.abs(config.boxColMax - config.boxColMin) + 1
  const rMin = Math.min(config.boxRowMin, config.boxRowMax)
  const rMax = Math.max(config.boxRowMin, config.boxRowMax)
  const cMin = Math.min(config.boxColMin, config.boxColMax)
  const cMax = Math.max(config.boxColMin, config.boxColMax)

  const latEdgeSouth = rowToLat(rMin) - 0.5
  const latEdgeNorth = rowToLat(rMax) + 0.5
  const lonEdgeWest = colToLon(cMin) - 0.5
  const lonEdgeEast = colToLon(cMax) + 0.5

  const setShape = (shape: ShockShape) => {
    const midR = Math.round((rMin + rMax) / 2)
    const midC = Math.round((cMin + cMax) / 2)
    onChange({
      ...config,
      shape,
      centerRow: shape === 'gaussian' ? midR : (rMin + rMax) / 2,
      centerCol: shape === 'gaussian' ? midC : (cMin + cMax) / 2,
      gaussianStroke:
        shape === 'gaussian' && (!config.gaussianStroke || config.gaussianStroke.length === 0)
          ? [{ row: midR, col: midC }]
          : config.gaussianStroke,
      strokeOrigin: null,
      presetName: null,
    })
  }

  const resizeBox = (newHeight: number, newWidth: number) => {
    const h = Math.max(1, Math.min(GRID_ROWS, newHeight))
    const w = Math.max(1, Math.min(GRID_COLS, newWidth))
    const cy = (rMin + rMax) / 2
    const cx = (cMin + cMax) / 2
    let nrMin = Math.round(cy - (h - 1) / 2)
    let nrMax = nrMin + h - 1
    if (nrMin < 0) {
      nrMin = 0
      nrMax = h - 1
    }
    if (nrMax >= GRID_ROWS) {
      nrMax = GRID_ROWS - 1
      nrMin = nrMax - h + 1
    }
    let ncMin = Math.round(cx - (w - 1) / 2)
    let ncMax = ncMin + w - 1
    if (ncMin < 0) {
      ncMin = 0
      ncMax = w - 1
    }
    if (ncMax >= GRID_COLS) {
      ncMax = GRID_COLS - 1
      ncMin = ncMax - w + 1
    }
    onChange({
      ...config,
      boxRowMin: nrMin,
      boxRowMax: nrMax,
      boxColMin: ncMin,
      boxColMax: ncMax,
      centerRow: (nrMin + nrMax) / 2,
      centerCol: (ncMin + ncMax) / 2,
      strokeOrigin: null,
      presetName: null,
    })
  }

  const pointerToPreviewCell = (e: ReactPointerEvent<HTMLDivElement>) => {
    const el = previewBoxRef.current
    if (!el) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const nx = Math.max(0, Math.min(0.9999, (e.clientX - rect.left) / rect.width))
    const ny = Math.max(0, Math.min(0.9999, (e.clientY - rect.top) / rect.height))
    const col = Math.floor(nx * GRID_COLS)
    const row = GRID_ROWS - 1 - Math.floor(ny * GRID_ROWS)
    return { row, col }
  }

  const handlePreviewPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    const cell = pointerToPreviewCell(e)
    if (!cell) return
    e.currentTarget.setPointerCapture(e.pointerId)
    dragOriginRef.current = cell
    didDragRef.current = false
    if (latestConfigRef.current.shape === 'gaussian') {
      onChange(moveShockCenter(latestConfigRef.current, cell.row, cell.col))
    }
  }

  const handlePreviewPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragOriginRef.current) return
    const cell = pointerToPreviewCell(e)
    if (!cell) return
    const start = dragOriginRef.current
    if (start.row !== cell.row || start.col !== cell.col) {
      didDragRef.current = true
      onChange(applyDragToShock(latestConfigRef.current, start.row, start.col, cell.row, cell.col))
    }
  }

  const handlePreviewPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragOriginRef.current) return
    const cell = pointerToPreviewCell(e) ?? dragOriginRef.current
    if (didDragRef.current) {
      onChange(
        applyDragToShock(
          latestConfigRef.current,
          dragOriginRef.current.row,
          dragOriginRef.current.col,
          cell.row,
          cell.col,
        ),
      )
    } else {
      onChange(moveShockCenter(latestConfigRef.current, cell.row, cell.col))
    }
    dragOriginRef.current = null
    didDragRef.current = false
  }

  const toggleChannel = (key: ChannelKey) => {
    const cur = config.channels[key]
    onChange({
      ...config,
      channels: {
        ...config.channels,
        [key]: {
          ...cur,
          enabled: !cur.enabled,
        },
      },
      presetName: null,
    })
  }

  const updateChannelMode = (key: ChannelKey, mode: AmplitudeMode) => {
    onChange({
      ...config,
      channels: {
        ...config.channels,
        [key]: {
          ...config.channels[key],
          enabled: true,
          mode,
        },
      },
      presetName: null,
    })
  }

  const updateChannelValue = (key: ChannelKey, raw: string) => {
    const val = Number.parseFloat(raw)
    onChange({
      ...config,
      channels: {
        ...config.channels,
        [key]: {
          ...config.channels[key],
          enabled: true,
          value: Number.isFinite(val) ? val : 0,
        },
      },
      presetName: null,
    })
  }

  const formatDeltaVal = (val: number, wireUnit: string, isResolved: boolean) => {
    const sign = val > 0 ? '+' : ''
    if (!isResolved) {
      return `${sign}${val.toFixed(1)}%`
    }
    if (wireUnit === 'Pa') {
      return `${sign}${Math.round(val)} Pa`
    }
    return `${sign}${val.toFixed(2)} ${wireUnit}`
  }

  return (
    <aside className="flex flex-col gap-5">
      {hasPaintedTrail &&
        interactiveMapHosts.map((host, idx) => (
          <PaintedGaussianMapOverlay key={idx} host={host} weights={preview.weights} />
        ))}

      {/* Section header */}
      <div className="flex items-baseline justify-between">
        <div>
          <h2 className="font-serif-display text-2xl leading-none text-[#ECE7DA]">
            Atmospheric shock
          </h2>
          <p className="mt-1 text-xs text-[#8C958D]">
            {config.shape === 'box'
              ? 'Drag a box on map or footprint preview'
              : 'Paint a Gaussian stroke on map or preview'}
          </p>
        </div>
        <button
          type="button"
          onClick={() => onChange(createEmptyShockConfig())}
          title="Reset perturbation"
          className="ui-transition inline-flex items-center gap-1 text-xs text-[#8C958D] hover:text-[#ECE7DA] cursor-pointer"
        >
          <RotateCcw className="h-3 w-3" strokeWidth={1.5} />
          Reset
        </button>
      </div>

      {/* Run / Zero shock actions placed high in the rail for immediate reach */}
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          disabled={isSimulating}
          onClick={onRun}
          className="ui-transition rounded-[3px] border border-[#F0B84B] bg-[#F0B84B] px-3 py-2 text-xs font-medium text-[#0D1210] hover:bg-[#f4c668] disabled:opacity-40 cursor-pointer"
        >
          {isSimulating ? 'Simulating…' : hasPrime ? 'Run Branch' : 'Run simulation'}
        </button>

        <button
          type="button"
          disabled={isSimulating}
          onClick={onRunZeroShock}
          className="ui-transition rounded-[3px] border border-[#26312B] bg-[#141B18] px-3 py-2 text-xs text-[#ECE7DA] hover:border-[#8C958D] disabled:opacity-40 cursor-pointer"
        >
          Zero shock
        </button>
      </div>

      {/* Presets */}
      <div className="border-t border-[#26312B] pt-4">
        <p className="mb-2 text-xs text-[#8C958D]">Scenario presets</p>
        <div className="grid gap-1.5">
          <button
            type="button"
            onClick={() => onChange(createWarmBayOfBengalPreset())}
            className={`ui-transition rounded-[2px] border px-3 py-2 text-left cursor-pointer ${
              config.presetName === 'Warm Bay of Bengal'
                ? 'border-[#F0B84B] bg-[#1A231F] text-[#ECE7DA]'
                : 'border-[#26312B] bg-[#141B18] text-[#ECE7DA] hover:border-[#5A645D]'
            }`}
          >
            <p className="text-xs font-medium">Warm Bay of Bengal</p>
            <p className="mt-0.5 font-mono text-[11px] text-[#8C958D]">
              +3.0 K t2m, +15% tcwv · 8–20°N, 82–94°E
            </p>
          </button>

          <button
            type="button"
            onClick={() => onChange(createWeakerArabianSeaWindPreset())}
            className={`ui-transition rounded-[2px] border px-3 py-2 text-left cursor-pointer ${
              config.presetName === 'Weaker Arabian Sea wind'
                ? 'border-[#F0B84B] bg-[#1A231F] text-[#ECE7DA]'
                : 'border-[#26312B] bg-[#141B18] text-[#ECE7DA] hover:border-[#5A645D]'
            }`}
          >
            <p className="text-xs font-medium">Weaker Arabian Sea wind</p>
            <p className="mt-0.5 font-mono text-[11px] text-[#8C958D]">
              −40% u10, −40% v10 · 8–20°N, 65–75°E
            </p>
          </button>
        </div>
      </div>

      {/* Footprint tool & live 32x36 preview */}
      <div className="border-t border-[#26312B] pt-4">
        <div className="mb-2.5 flex items-center justify-between gap-2">
          <span className="text-xs text-[#8C958D]">Footprint geometry</span>
          <div className="inline-flex overflow-hidden rounded-[2px] border border-[#26312B] text-xs">
            <button
              type="button"
              onClick={() => setShape('box')}
              className={`ui-transition px-2.5 py-1 cursor-pointer ${
                config.shape === 'box'
                  ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                  : 'bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
              }`}
            >
              Box
            </button>
            <button
              type="button"
              onClick={() => setShape('gaussian')}
              className={`ui-transition border-l border-[#26312B] px-2.5 py-1 cursor-pointer ${
                config.shape === 'gaussian'
                  ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                  : 'bg-[#141B18] text-[#8C958D] hover:text-[#ECE7DA]'
              }`}
            >
              Gaussian
            </button>
          </div>
        </div>

        <div
          ref={previewBoxRef}
          onPointerDown={handlePreviewPointerDown}
          onPointerMove={handlePreviewPointerMove}
          onPointerUp={handlePreviewPointerUp}
          className="relative aspect-[36/32] w-full cursor-crosshair select-none overflow-hidden rounded-[2px] border border-[#26312B] bg-[#141B18] touch-none"
        >
          <canvas ref={previewCanvasRef} className="block h-full w-full" />
          <span className="pointer-events-none absolute bottom-1 left-1.5 font-mono text-[9px] text-[#5A645D]">
            6.5°N, 65.5°E
          </span>
          <span className="pointer-events-none absolute top-1 right-1.5 font-mono text-[9px] text-[#8C958D] tabular-nums">
            {preview.activeCells} / {GRID_ROWS * GRID_COLS} cells
          </span>
        </div>

        {config.shape === 'box' ? (
          <div className="mt-2.5 space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="text-xs text-[#8C958D]">Lat span (cells)</span>
                <input
                  type="number"
                  min={1}
                  max={GRID_ROWS}
                  value={boxHeight}
                  onChange={(e) => {
                    const val = Number.parseInt(e.target.value, 10)
                    if (!Number.isNaN(val)) resizeBox(val, boxWidth)
                  }}
                  className="tabular-nums mt-1 w-full rounded-[2px] border border-[#26312B] bg-[#141B18] px-2 py-1 font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
                />
              </label>
              <label className="block">
                <span className="text-xs text-[#8C958D]">Lon span (cells)</span>
                <input
                  type="number"
                  min={1}
                  max={GRID_COLS}
                  value={boxWidth}
                  onChange={(e) => {
                    const val = Number.parseInt(e.target.value, 10)
                    if (!Number.isNaN(val)) resizeBox(boxHeight, val)
                  }}
                  className="tabular-nums mt-1 w-full rounded-[2px] border border-[#26312B] bg-[#141B18] px-2 py-1 font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
                />
              </label>
            </div>
            <p className="font-mono text-[11px] text-[#ECE7DA] tabular-nums">
              {latEdgeSouth.toFixed(0)}°–{latEdgeNorth.toFixed(0)}°N, {lonEdgeWest.toFixed(0)}°–
              {lonEdgeEast.toFixed(0)}°E{' '}
              <span className="text-[#5A645D]">
                (r{rMin}..{rMax}, c{cMin}..{cMax})
              </span>
            </p>
          </div>
        ) : (
          <div className="mt-2.5 space-y-2">
            <label className="block">
              <div className="flex items-center justify-between text-xs text-[#8C958D]">
                <span>Brush sigma</span>
                <span className="font-mono text-[#ECE7DA] tabular-nums">
                  σ = {config.gaussianSigma.toFixed(1)}c
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                step={0.5}
                value={config.gaussianSigma}
                onChange={(e) =>
                  onChange({
                    ...config,
                    gaussianSigma: Number.parseFloat(e.target.value),
                    presetName: null,
                  })
                }
                className="instrument-range mt-2 w-full"
              />
            </label>
            <div className="flex items-center justify-between font-mono text-[11px] text-[#8C958D] tabular-nums">
              <span>
                {strokePoints.length > 1
                  ? `${strokePoints.length} painted nodes`
                  : `${rowToLat(config.centerRow).toFixed(1)}°N, ${colToLon(config.centerCol).toFixed(1)}°E`}
              </span>
              {strokePoints.length > 1 && (
                <button
                  type="button"
                  onClick={() =>
                    onChange(
                      moveShockCenter(
                        config,
                        Math.round(config.centerRow),
                        Math.round(config.centerCol),
                      ),
                    )
                  }
                  className="ui-transition text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
                >
                  Clear stroke
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Channel amplitudes */}
      <div className="border-t border-[#26312B] pt-4">
        <div className="mb-2 flex items-baseline justify-between">
          <span className="text-xs text-[#8C958D]">Channel amplitudes</span>
          <span className="font-mono text-[10px] text-[#5A645D]">% of init_states[2]</span>
        </div>

        <div className="border-t border-[#26312B]">
          {CHANNELS.map((ch) => {
            const spec = config.channels[ch.key]
            return (
              <div
                key={ch.key}
                className="flex items-center justify-between gap-2 border-b border-[#26312B] py-2"
              >
                <label className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={spec.enabled}
                    onChange={() => toggleChannel(ch.key)}
                    className="h-3.5 w-3.5 rounded-[2px] border-[#26312B] bg-[#141B18] accent-[#F0B84B]"
                  />
                  <span className="font-mono text-xs text-[#ECE7DA]">{ch.shortLabel}</span>
                  <span className="font-mono text-[10px] text-[#5A645D]">{ch.wireUnit}</span>
                </label>

                {spec.enabled && (
                  <div className="flex items-center gap-1.5">
                    <input
                      type="number"
                      step={spec.mode === 'percent' ? 5 : ch.key === 'msl' ? 100 : 0.5}
                      value={spec.value}
                      onChange={(e) => updateChannelValue(ch.key, e.target.value)}
                      aria-label={`${ch.shortLabel} shock amplitude`}
                      className="tabular-nums w-16 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1.5 py-0.5 text-right font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
                    />
                    <div className="inline-flex overflow-hidden rounded-[2px] border border-[#26312B] font-mono text-[10px]">
                      <button
                        type="button"
                        onClick={() => updateChannelMode(ch.key, 'absolute')}
                        className={`px-1.5 py-0.5 cursor-pointer ${
                          spec.mode === 'absolute'
                            ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                            : 'bg-[#141B18] text-[#8C958D]'
                        }`}
                      >
                        {ch.wireUnit}
                      </button>
                      <button
                        type="button"
                        onClick={() => updateChannelMode(ch.key, 'percent')}
                        className={`border-l border-[#26312B] px-1.5 py-0.5 cursor-pointer ${
                          spec.mode === 'percent'
                            ? 'bg-[#F0B84B] font-medium text-[#0D1210]'
                            : 'bg-[#141B18] text-[#8C958D]'
                        }`}
                      >
                        %
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Live numeric delta readout */}
      <div className="border-t border-[#26312B] pt-4">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-[#8C958D]">Physical delta ([6][32][36])</span>
          <span className="font-mono text-[10px] text-[#5A645D] tabular-nums">
            ∫w = {preview.effectiveCells.toFixed(1)}c
          </span>
        </div>

        {preview.channelReadouts.length === 0 ? (
          <p className="mt-1.5 font-mono text-xs text-[#5A645D]">All 6 channels = 0.00</p>
        ) : (
          <div className="mt-2 space-y-1 font-mono text-[11px] tabular-nums">
            {preview.channelReadouts.map((item) => (
              <div key={item.key} className="flex items-baseline justify-between gap-2">
                <span className="text-[#F0B84B]">{item.shortLabel}</span>
                <span className="text-[#ECE7DA]">
                  peak {formatDeltaVal(item.peakSignedDelta, item.wireUnit, item.hasPhysicalBase)}
                  <span className="ml-1.5 text-[#8C958D]">
                    · mean {formatDeltaVal(item.meanSignedDelta, item.wireUnit, item.hasPhysicalBase)}
                  </span>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
