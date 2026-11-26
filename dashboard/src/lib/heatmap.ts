import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type Grid2D,
  type State3D,
  type StateSequence,
} from '../api/types'
import {
  computeScale,
  normalizeToScale,
  writePaletteColor,
  type ColorScale,
  type PaletteKind,
} from './colormaps'
import { DISPLAY_CHANNELS, extractDisplayGrid } from './constants'
import type { ShockConfig } from './shock'

/** 5-degree latitude parallels inside [6°N, 38°N] */
export const GRATICULE_LATS = [35, 30, 25, 20, 15, 10] as const

/** 5-degree longitude meridians inside [65°E, 101°E] */
export const GRATICULE_LONS = [70, 75, 80, 85, 90, 95, 100] as const

/** North-up normalized vertical position [0..1] for a latitude in [6°N, 38°N] */
export function latToNormY(lat: number): number {
  return (38.0 - lat) / GRID_ROWS
}

/** Normalized horizontal position [0..1] for a longitude in [65°E, 101°E] */
export function lonToNormX(lon: number): number {
  return (lon - 65.0) / GRID_COLS
}

export type ContourSegment = [number, number, number, number]

export interface WindArrow {
  x1: number
  y1: number
  x2: number
  y2: number
  headA: [number, number]
  headB: [number, number]
}

export interface ChannelRolloutSeries {
  primeGrids: Grid2D[]
  branchGrids: Grid2D[]
  diffGrids: Grid2D[]
  sharedScale: ColorScale
  diffScale: ColorScale
  stepMaxAbsDiff: number[]
  rolloutMaxAbsDiff: number
}

export interface PrecomputedRolloutWorkspace {
  byChannel: Record<DisplayChannelId, ChannelRolloutSeries>
  primeWindArrows: WindArrow[][]
  branchWindArrows: WindArrow[][]
  primeTpSegments: ContourSegment[][]
  branchTpSegments: ContourSegment[][]
}

export function buildTpOutlineSegments(
  state: State3D,
  thresholdMmDay = 10,
): ContourSegment[] {
  const tpGrid = state[1]
  const segments: ContourSegment[] = []
  const above = (r: number, c: number) =>
    r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS && tpGrid[r][c] > thresholdMmDay

  for (let r = 0; r < GRID_ROWS; r++) {
    const y = GRID_ROWS - 1 - r
    for (let c = 0; c < GRID_COLS; c++) {
      if (!above(r, c)) continue
      if (!above(r + 1, c)) segments.push([c, y, c + 1, y])
      if (!above(r - 1, c)) segments.push([c, y + 1, c + 1, y + 1])
      if (!above(r, c - 1)) segments.push([c, y, c, y + 1])
      if (!above(r, c + 1)) segments.push([c + 1, y, c + 1, y + 1])
    }
  }
  return segments
}

export function buildWindArrows(state: State3D): WindArrow[] {
  const uGrid = state[2]
  const vGrid = state[3]
  let maxSpeed = 0
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const s = Math.hypot(uGrid[r][c], vGrid[r][c])
      if (s > maxSpeed) maxSpeed = s
    }
  }
  if (maxSpeed <= 1e-6) return []

  const arrows: WindArrow[] = []
  for (let r = 1; r < GRID_ROWS; r += 2) {
    const cy = GRID_ROWS - 1 - r + 0.5
    for (let c = 1; c < GRID_COLS; c += 2) {
      const u = uGrid[r][c]
      const v = vGrid[r][c]
      const speed = Math.hypot(u, v)
      if (speed < maxSpeed * 0.04) continue
      const len = 0.45 + 1.15 * Math.min(1, speed / maxSpeed)
      const ux = u / speed
      const uy = -v / speed // +v points north = negative visual Y
      const cx = c + 0.5
      const x1 = cx - ux * (len * 0.42)
      const y1 = cy - uy * (len * 0.42)
      const x2 = cx + ux * (len * 0.58)
      const y2 = cy + uy * (len * 0.58)

      const headLen = Math.min(0.45, len * 0.35)
      const angle = Math.atan2(uy, ux)
      const a1 = angle + Math.PI * 0.78
      const a2 = angle - Math.PI * 0.78
      arrows.push({
        x1,
        y1,
        x2,
        y2,
        headA: [x2 + Math.cos(a1) * headLen, y2 + Math.sin(a1) * headLen],
        headB: [x2 + Math.cos(a2) * headLen, y2 + Math.sin(a2) * headLen],
      })
    }
  }
  return arrows
}

/**
 * Precompute all channel display grids, difference grids, rollout-consistent color scales,
 * wind vectors, and tp > 10 mm/day contours ONCE when primeRollout or branchRollout changes.
 */
export function precomputeRolloutWorkspace(
  primeRollout: StateSequence | null,
  branchRollout: StateSequence | null,
): PrecomputedRolloutWorkspace {
  const primeLen = primeRollout?.length ?? 0
  const branchLen = branchRollout?.length ?? 0
  const diffLen = Math.min(primeLen, branchLen)

  const byChannel = {} as Record<DisplayChannelId, ChannelRolloutSeries>

  for (const ch of DISPLAY_CHANNELS) {
    const primeGrids: Grid2D[] = new Array(primeLen)
    for (let k = 0; k < primeLen; k++) {
      primeGrids[k] = extractDisplayGrid(primeRollout![k], ch.id)
    }

    const branchGrids: Grid2D[] = new Array(branchLen)
    for (let k = 0; k < branchLen; k++) {
      branchGrids[k] = extractDisplayGrid(branchRollout![k], ch.id)
    }

    const diffGrids: Grid2D[] = new Array(diffLen)
    const stepMaxAbsDiff: number[] = new Array(diffLen)
    let rolloutMaxAbsDiff = 0

    for (let k = 0; k < diffLen; k++) {
      const pg = primeGrids[k]
      const bg = branchGrids[k]
      const dg: Grid2D = new Array(GRID_ROWS)
      let stepMax = 0
      for (let r = 0; r < GRID_ROWS; r++) {
        const row = new Array<number>(GRID_COLS)
        for (let c = 0; c < GRID_COLS; c++) {
          const d = bg[r][c] - pg[r][c]
          row[c] = d
          const ad = Math.abs(d)
          if (ad > stepMax) stepMax = ad
        }
        dg[r] = row
      }
      diffGrids[k] = dg
      stepMaxAbsDiff[k] = stepMax
      if (stepMax > rolloutMaxAbsDiff) rolloutMaxAbsDiff = stepMax
    }

    const clampZero = ch.id === 'tp' || ch.id === 'wind' || ch.id === 'tcwv'
    const combinedGrids = [...primeGrids, ...branchGrids]
    const sharedScale = computeScale(combinedGrids, Boolean(ch.divergingBase), clampZero)
    const diffScale =
      diffGrids.length > 0
        ? computeScale(diffGrids, true)
        : { min: -1, max: 1, diverging: true }

    byChannel[ch.id] = {
      primeGrids,
      branchGrids,
      diffGrids,
      sharedScale,
      diffScale,
      stepMaxAbsDiff,
      rolloutMaxAbsDiff,
    }
  }

  const primeWindArrows = primeRollout ? primeRollout.map((s) => buildWindArrows(s)) : []
  const branchWindArrows = branchRollout ? branchRollout.map((s) => buildWindArrows(s)) : []
  const primeTpSegments = primeRollout ? primeRollout.map((s) => buildTpOutlineSegments(s, 10)) : []
  const branchTpSegments = branchRollout
    ? branchRollout.map((s) => buildTpOutlineSegments(s, 10))
    : []

  return {
    byChannel,
    primeWindArrows,
    branchWindArrows,
    primeTpSegments,
    branchTpSegments,
  }
}

const SMOOTH_UPSCALE = 8

/**
 * Render a 32x36 north-up heatmap into an offscreen buffer canvas.
 * - When smooth === false: crisp 36x32 cell grid (row 0 = south at bottom, row 31 = north at top).
 * - When smooth === true: scalar-space bilinear interpolation at 8x sub-cell resolution prior to
 *   palette lookup so gradients and zero-crossings remain physically faithful.
 */
export function renderHeatmapToBuffer(
  bufferCanvas: HTMLCanvasElement,
  grid: Grid2D,
  scale: ColorScale,
  palette: PaletteKind,
  smooth: boolean,
): void {
  const targetW = smooth ? GRID_COLS * SMOOTH_UPSCALE : GRID_COLS
  const targetH = smooth ? GRID_ROWS * SMOOTH_UPSCALE : GRID_ROWS

  if (bufferCanvas.width !== targetW || bufferCanvas.height !== targetH) {
    bufferCanvas.width = targetW
    bufferCanvas.height = targetH
  }

  const ctx = bufferCanvas.getContext('2d')
  if (!ctx) return

  const img = ctx.createImageData(targetW, targetH)
  const data = img.data

  if (!smooth) {
    for (let r = 0; r < GRID_ROWS; r++) {
      const vy = GRID_ROWS - 1 - r
      const rowData = grid[r]
      for (let c = 0; c < GRID_COLS; c++) {
        const t = normalizeToScale(rowData[c], scale)
        writePaletteColor(palette, t, data, (vy * GRID_COLS + c) * 4)
      }
    }
    ctx.putImageData(img, 0, 0)
    return
  }

  // Precompute column interpolation weights once
  const c0Arr = new Int32Array(targetW)
  const c1Arr = new Int32Array(targetW)
  const fcArr = new Float64Array(targetW)
  for (let px = 0; px < targetW; px++) {
    const gx = Math.max(0, Math.min(GRID_COLS - 1, (px + 0.5) / SMOOTH_UPSCALE - 0.5))
    const c0 = Math.floor(gx)
    c0Arr[px] = c0
    c1Arr[px] = Math.min(GRID_COLS - 1, c0 + 1)
    fcArr[px] = gx - c0
  }

  for (let py = 0; py < targetH; py++) {
    const gy = Math.max(0, Math.min(GRID_ROWS - 1, (py + 0.5) / SMOOTH_UPSCALE - 0.5))
    const gr = GRID_ROWS - 1 - gy
    const r0 = Math.floor(gr)
    const r1 = Math.min(GRID_ROWS - 1, r0 + 1)
    const fr = gr - r0
    const row0 = grid[r0]
    const row1 = grid[r1]
    const rowOffset = py * targetW * 4

    for (let px = 0; px < targetW; px++) {
      const c0 = c0Arr[px]
      const c1 = c1Arr[px]
      const fc = fcArr[px]
      const v0 = row0[c0] + (row0[c1] - row0[c0]) * fc
      const v1 = row1[c0] + (row1[c1] - row1[c0]) * fc
      const val = v0 + (v1 - v0) * fr
      const t = normalizeToScale(val, scale)
      writePaletteColor(palette, t, data, rowOffset + px * 4)
    }
  }

  ctx.putImageData(img, 0, 0)
}

export interface CanvasOverlayOptions {
  width: number
  height: number
  dpr: number
  tpSegments?: ContourSegment[]
  windArrows?: WindArrow[]
  shockConfig?: ShockConfig | null
  showShockOverlay?: boolean
  hoveredCell?: { row: number; col: number } | null
  drawGraticule?: boolean
  drawNorthBadge?: boolean
}

export function drawFieldVectorsAndContours(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  tpSegments?: ContourSegment[],
  windArrows?: WindArrow[],
): void {
  const toX = (gx: number) => (gx / GRID_COLS) * width
  const toY = (gy: number) => (gy / GRID_ROWS) * height

  if (tpSegments && tpSegments.length > 0) {
    ctx.save()
    ctx.lineCap = 'square'

    ctx.beginPath()
    for (let i = 0; i < tpSegments.length; i++) {
      const [x1, y1, x2, y2] = tpSegments[i]
      ctx.moveTo(toX(x1), toY(y1))
      ctx.lineTo(toX(x2), toY(y2))
    }
    ctx.strokeStyle = 'rgba(13, 18, 16, 0.75)'
    ctx.lineWidth = 2.5 * dpr
    ctx.stroke()

    ctx.strokeStyle = '#ece7da'
    ctx.lineWidth = 1.2 * dpr
    ctx.stroke()
    ctx.restore()
  }

  if (windArrows && windArrows.length > 0) {
    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    ctx.beginPath()
    for (let i = 0; i < windArrows.length; i++) {
      const a = windArrows[i]
      const x2 = toX(a.x2)
      const y2 = toY(a.y2)
      ctx.moveTo(toX(a.x1), toY(a.y1))
      ctx.lineTo(x2, y2)
      ctx.moveTo(toX(a.headA[0]), toY(a.headA[1]))
      ctx.lineTo(x2, y2)
      ctx.lineTo(toX(a.headB[0]), toY(a.headB[1]))
    }
    ctx.strokeStyle = 'rgba(13, 18, 16, 0.65)'
    ctx.lineWidth = 2.2 * dpr
    ctx.stroke()

    ctx.strokeStyle = 'rgba(236, 231, 218, 0.9)'
    ctx.lineWidth = 1.1 * dpr
    ctx.stroke()
    ctx.restore()
  }
}

/**
 * Draw lat/lon 5-degree graticule lines (no geographic outlines), shock region,
 * north-up indicator, and hover reticle onto a high-DPI canvas.
 */
export function drawCanvasMapOverlays(
  ctx: CanvasRenderingContext2D,
  opts: CanvasOverlayOptions,
): void {
  const {
    width,
    height,
    dpr,
    tpSegments,
    windArrows,
    shockConfig,
    showShockOverlay = false,
    hoveredCell,
    drawGraticule = true,
  } = opts

  const toX = (gx: number) => (gx / GRID_COLS) * width
  const toY = (gy: number) => (gy / GRID_ROWS) * height

  // 1. Crisp 1px Lat/Lon graticule lines (5° grid, zero geographic outlines)
  if (drawGraticule) {
    ctx.save()
    ctx.strokeStyle = 'rgba(90, 100, 93, 0.36)'
    ctx.lineWidth = 1 * dpr

    ctx.beginPath()
    for (const lat of GRATICULE_LATS) {
      const y = Math.round(latToNormY(lat) * height) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
    }
    for (const lon of GRATICULE_LONS) {
      const x = Math.round(lonToNormX(lon) * width) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
    }
    ctx.stroke()

    // Crisp edge tick marks
    ctx.strokeStyle = 'rgba(140, 149, 141, 0.65)'
    ctx.lineWidth = 1 * dpr
    const tickLen = 4 * dpr
    ctx.beginPath()
    for (const lat of GRATICULE_LATS) {
      const y = Math.round(latToNormY(lat) * height) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(tickLen, y)
      ctx.moveTo(width - tickLen, y)
      ctx.lineTo(width, y)
    }
    for (const lon of GRATICULE_LONS) {
      const x = Math.round(lonToNormX(lon) * width) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, tickLen)
      ctx.moveTo(x, height - tickLen)
      ctx.lineTo(x, height)
    }
    ctx.stroke()
    ctx.restore()
  }

  // 2. Field contours & wind vectors
  drawFieldVectorsAndContours(ctx, width, height, dpr, tpSegments, windArrows)

  // 3. Shock region footprint
  if (showShockOverlay && shockConfig) {
    ctx.save()
    if (shockConfig.shape === 'box') {
      const cMin = Math.min(shockConfig.boxColMin, shockConfig.boxColMax)
      const cMax = Math.max(shockConfig.boxColMin, shockConfig.boxColMax)
      const rMin = Math.min(shockConfig.boxRowMin, shockConfig.boxRowMax)
      const rMax = Math.max(shockConfig.boxRowMin, shockConfig.boxRowMax)

      const x = toX(cMin)
      const y = toY(GRID_ROWS - 1 - rMax)
      const w = toX(cMax - cMin + 1)
      const h = toY(rMax - rMin + 1)

      ctx.strokeStyle = 'rgba(13, 18, 16, 0.72)'
      ctx.lineWidth = 2.4 * dpr
      ctx.setLineDash([5 * dpr, 3 * dpr])
      ctx.strokeRect(x, y, w, h)

      ctx.strokeStyle = '#f0b84b'
      ctx.lineWidth = 1.3 * dpr
      ctx.strokeRect(x, y, w, h)
    } else {
      const cx = toX(shockConfig.centerCol + 0.5)
      const cy = toY(GRID_ROWS - 1 - shockConfig.centerRow + 0.5)
      const cellW = width / GRID_COLS
      const cellH = height / GRID_ROWS
      const sigma = shockConfig.gaussianSigma

      // Outer 2.5-sigma cutoff ellipse
      ctx.beginPath()
      ctx.ellipse(cx, cy, sigma * 2.5 * cellW, sigma * 2.5 * cellH, 0, 0, Math.PI * 2)
      ctx.setLineDash([4 * dpr, 3 * dpr])
      ctx.strokeStyle = 'rgba(240, 184, 75, 0.62)'
      ctx.lineWidth = 1.1 * dpr
      ctx.stroke()

      // Inner 1-sigma ellipse
      ctx.beginPath()
      ctx.ellipse(cx, cy, sigma * cellW, sigma * cellH, 0, 0, Math.PI * 2)
      ctx.setLineDash([])
      ctx.strokeStyle = '#f0b84b'
      ctx.lineWidth = 1.35 * dpr
      ctx.stroke()

      // Center crosshair
      const arm = 0.65 * Math.min(cellW, cellH)
      ctx.beginPath()
      ctx.moveTo(cx - arm, cy)
      ctx.lineTo(cx + arm, cy)
      ctx.moveTo(cx, cy - arm)
      ctx.lineTo(cx, cy + arm)
      ctx.strokeStyle = '#ece7da'
      ctx.lineWidth = 1.2 * dpr
      ctx.stroke()
    }
    ctx.restore()
  }

  // 4. Hovered cell reticle
  if (hoveredCell) {
    const cellW = width / GRID_COLS
    const cellH = height / GRID_ROWS
    const x = hoveredCell.col * cellW
    const y = (GRID_ROWS - 1 - hoveredCell.row) * cellH

    ctx.save()
    ctx.strokeStyle = 'rgba(13, 18, 16, 0.85)'
    ctx.lineWidth = 2.4 * dpr
    ctx.strokeRect(x, y, cellW, cellH)

    ctx.strokeStyle = '#f0b84b'
    ctx.lineWidth = 1.2 * dpr
    ctx.strokeRect(x, y, cellW, cellH)
    ctx.restore()
  }
}
