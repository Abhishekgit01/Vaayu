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
import {
  computeSpatialWeights,
  extractIsoContourSegments,
  type ShockConfig,
} from './shock'
import {
  DAY1_LOCALITY_CELLS,
  FRONT_THRESHOLD_FRACTION,
  REFERENCE_FRONT_CELLS_PER_DAY,
} from './spread'
import {
  computeRelativeVorticityGrid,
  VORTICITY_META,
  type ViewFieldId,
} from './vorticity'
import {
  DEFAULT_MAP_VIEWPORT,
  getViewportBounds,
  getViewportGraticuleTicks,
  type ViewportBounds,
} from './viewport'
import { buildDiffWindArrows } from './windFlow'

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
  /** Per-step isoline contour of |Branch - Prime| at 5% of rolloutMaxAbsDiff */
  frontContoursByStep: ContourSegment[][]
}

export interface PrecomputedRolloutWorkspace {
  byChannel: Record<ViewFieldId, ChannelRolloutSeries>
  primeWindArrows: WindArrow[][]
  branchWindArrows: WindArrow[][]
  diffWindArrows: WindArrow[][]
  primeTpSegments: ContourSegment[][]
  branchTpSegments: ContourSegment[][]
  /** Thin dashed 3-cell Day-1 locality ring around shockMask */
  localityRingSegments: ContourSegment[]
  /** Per-step thin dashed 8 cells/day reference rings (day 1..N) around shockMask */
  refRingsByStep: ContourSegment[][]
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

const CONTOUR_NX = GRID_COLS * 2
const CONTOUR_NY = GRID_ROWS * 2

/**
 * Convert a [32][36] cell-centered scalar grid (row 0 = south) into a bilinearly
 * sampled north-up vertex grid [ny + 1][nx + 1] over [0..36] x [0..32].
 */
export function buildNorthUpScalarVertexField(
  cellGrid: Grid2D,
  nx = CONTOUR_NX,
  ny = CONTOUR_NY,
  absValue = false,
): number[][] {
  const field: number[][] = new Array(ny + 1)
  for (let iy = 0; iy <= ny; iy++) {
    const gy = (iy / ny) * GRID_ROWS
    const gr = Math.max(0, Math.min(GRID_ROWS - 1, GRID_ROWS - 1 - (gy - 0.5)))
    const r0 = Math.floor(gr)
    const r1 = Math.min(GRID_ROWS - 1, r0 + 1)
    const fr = gr - r0
    const row0 = cellGrid[r0]
    const row1 = cellGrid[r1]
    const outRow = new Array<number>(nx + 1)

    for (let ix = 0; ix <= nx; ix++) {
      const gx = (ix / nx) * GRID_COLS
      const gc = Math.max(0, Math.min(GRID_COLS - 1, gx - 0.5))
      const c0 = Math.floor(gc)
      const c1 = Math.min(GRID_COLS - 1, c0 + 1)
      const fc = gc - c0

      const a00 = absValue ? Math.abs(row0[c0]) : row0[c0]
      const a01 = absValue ? Math.abs(row0[c1]) : row0[c1]
      const a10 = absValue ? Math.abs(row1[c0]) : row1[c0]
      const a11 = absValue ? Math.abs(row1[c1]) : row1[c1]

      const v0 = a00 + (a01 - a00) * fc
      const v1 = a10 + (a11 - a10) * fc
      outRow[ix] = v0 + (v1 - v0) * fr
    }
    field[iy] = outRow
  }
  return field
}

function buildDistanceVertexFieldFromMask(
  shockMask?: boolean[][],
  nx = CONTOUR_NX,
  ny = CONTOUR_NY,
): number[][] | null {
  if (!shockMask) return null
  const shockCells: Array<[number, number]> = []
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (shockMask[r]?.[c]) {
        shockCells.push([r, c])
      }
    }
  }
  if (shockCells.length === 0) return null

  const field: number[][] = new Array(ny + 1)
  for (let iy = 0; iy <= ny; iy++) {
    const gy = (iy / ny) * GRID_ROWS
    const rowCoord = GRID_ROWS - 1 - (gy - 0.5)
    const outRow = new Array<number>(nx + 1)
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
      outRow[ix] = Math.sqrt(minSq)
    }
    field[iy] = outRow
  }
  return field
}

/**
 * Precompute all channel display grids, difference grids, rollout-consistent color scales,
 * 5%-of-peak |Branch - Prime| response front contours, 3-cell locality ring, and 8 cells/day
 * reference rings ONCE when rollouts or applied shock mask change.
 */
export function precomputeRolloutWorkspace(
  primeRollout: StateSequence | null,
  branchRollout: StateSequence | null,
  shockMask?: boolean[][],
): PrecomputedRolloutWorkspace {
  const primeLen = primeRollout?.length ?? 0
  const branchLen = branchRollout?.length ?? 0
  const diffLen = Math.min(primeLen, branchLen)
  const maxSteps = Math.max(primeLen, branchLen)

  const byChannel = {} as Record<ViewFieldId, ChannelRolloutSeries>

  const allFields: Array<{
    id: ViewFieldId
    divergingBase: boolean
    clampZero: boolean
    extract: (state: State3D) => Grid2D
  }> = [
    ...DISPLAY_CHANNELS.map((ch) => ({
      id: ch.id as ViewFieldId,
      divergingBase: Boolean(ch.divergingBase),
      clampZero: ch.id === 'tp' || ch.id === 'wind' || ch.id === 'tcwv',
      extract: (state: State3D) => extractDisplayGrid(state, ch.id),
    })),
    {
      id: 'vorticity',
      divergingBase: VORTICITY_META.divergingBase,
      clampZero: false,
      extract: (state: State3D) => computeRelativeVorticityGrid(state),
    },
  ]

  for (const field of allFields) {
    const primeGrids: Grid2D[] = new Array(primeLen)
    for (let k = 0; k < primeLen; k++) {
      primeGrids[k] = field.extract(primeRollout![k])
    }

    const branchGrids: Grid2D[] = new Array(branchLen)
    for (let k = 0; k < branchLen; k++) {
      branchGrids[k] = field.extract(branchRollout![k])
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

    const combinedGrids = [...primeGrids, ...branchGrids]
    const sharedScale = computeScale(combinedGrids, field.divergingBase, field.clampZero)
    const diffScale =
      diffGrids.length > 0
        ? computeScale(diffGrids, true)
        : { min: -1, max: 1, diverging: true }

    const frontContoursByStep: ContourSegment[][] = new Array(diffLen)
    const frontThreshold = rolloutMaxAbsDiff * FRONT_THRESHOLD_FRACTION
    for (let k = 0; k < diffLen; k++) {
      if (frontThreshold <= 0 || stepMaxAbsDiff[k] < frontThreshold) {
        frontContoursByStep[k] = []
      } else {
        const magVertexField = buildNorthUpScalarVertexField(
          diffGrids[k],
          CONTOUR_NX,
          CONTOUR_NY,
          true,
        )
        frontContoursByStep[k] = extractIsoContourSegments(
          magVertexField,
          CONTOUR_NX,
          CONTOUR_NY,
          frontThreshold,
        )
      }
    }

    byChannel[field.id] = {
      primeGrids,
      branchGrids,
      diffGrids,
      sharedScale,
      diffScale,
      stepMaxAbsDiff,
      rolloutMaxAbsDiff,
      frontContoursByStep,
    }
  }

  const primeWindArrows = primeRollout ? primeRollout.map((s) => buildWindArrows(s)) : []
  const branchWindArrows = branchRollout ? branchRollout.map((s) => buildWindArrows(s)) : []
  const diffWindArrows: WindArrow[][] = new Array(diffLen)
  for (let k = 0; k < diffLen; k++) {
    diffWindArrows[k] = buildDiffWindArrows(primeRollout![k], branchRollout![k])
  }
  const primeTpSegments = primeRollout ? primeRollout.map((s) => buildTpOutlineSegments(s, 10)) : []
  const branchTpSegments = branchRollout
    ? branchRollout.map((s) => buildTpOutlineSegments(s, 10))
    : []

  const distField = buildDistanceVertexFieldFromMask(shockMask)
  const localityRingSegments = distField
    ? extractIsoContourSegments(distField, CONTOUR_NX, CONTOUR_NY, DAY1_LOCALITY_CELLS)
    : []
  const refRingsByStep: ContourSegment[][] = new Array(maxSteps)
  for (let k = 0; k < maxSteps; k++) {
    if (!distField) {
      refRingsByStep[k] = []
    } else {
      const refDist = (k + 1) * REFERENCE_FRONT_CELLS_PER_DAY
      refRingsByStep[k] = extractIsoContourSegments(distField, CONTOUR_NX, CONTOUR_NY, refDist)
    }
  }

  return {
    byChannel,
    primeWindArrows,
    branchWindArrows,
    diffWindArrows,
    primeTpSegments,
    branchTpSegments,
    localityRingSegments,
    refRingsByStep,
  }
}

const SMOOTH_UPSCALE = 8
const SEA_DARKEN_STRENGTH = 0.24

function applySeaTint(
  data: Uint8ClampedArray,
  rgbaOffset: number,
  landFraction: number,
): void {
  const sea = (1 - Math.max(0, Math.min(1, landFraction))) * SEA_DARKEN_STRENGTH
  if (sea <= 0) return
  const keep = 1 - sea
  // Blend sea pixels subtly toward #0D1210 (13, 18, 16)
  data[rgbaOffset] = Math.round(data[rgbaOffset] * keep + 13 * sea)
  data[rgbaOffset + 1] = Math.round(data[rgbaOffset + 1] * keep + 18 * sea)
  data[rgbaOffset + 2] = Math.round(data[rgbaOffset + 2] * keep + 16 * sea)
}

/**
 * Render a 32x36 north-up heatmap into an offscreen buffer canvas.
 * - When smooth === false: crisp 36x32 cell grid (row 0 = south at bottom, row 31 = north at top).
 * - When smooth === true: scalar-space bilinear interpolation at 8x sub-cell resolution prior to
 *   palette lookup so gradients and zero-crossings remain physically faithful.
 * - When landSeaMask is present: sea cells (mask < 1) are tinted subtly darker.
 */
export function renderHeatmapToBuffer(
  bufferCanvas: HTMLCanvasElement,
  grid: Grid2D,
  scale: ColorScale,
  palette: PaletteKind,
  smooth: boolean,
  landSeaMask?: Grid2D | null,
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
      const maskRow = landSeaMask ? landSeaMask[r] : null
      for (let c = 0; c < GRID_COLS; c++) {
        const t = normalizeToScale(rowData[c], scale)
        const off = (vy * GRID_COLS + c) * 4
        writePaletteColor(palette, t, data, off)
        if (maskRow) {
          applySeaTint(data, off, maskRow[c])
        }
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
    const mRow0 = landSeaMask ? landSeaMask[r0] : null
    const mRow1 = landSeaMask ? landSeaMask[r1] : null
    const rowOffset = py * targetW * 4

    for (let px = 0; px < targetW; px++) {
      const c0 = c0Arr[px]
      const c1 = c1Arr[px]
      const fc = fcArr[px]
      const v0 = row0[c0] + (row0[c1] - row0[c0]) * fc
      const v1 = row1[c0] + (row1[c1] - row1[c0]) * fc
      const val = v0 + (v1 - v0) * fr
      const t = normalizeToScale(val, scale)
      const off = rowOffset + px * 4
      writePaletteColor(palette, t, data, off)
      if (mRow0 && mRow1) {
        const m0 = mRow0[c0] + (mRow0[c1] - mRow0[c0]) * fc
        const m1 = mRow1[c0] + (mRow1[c1] - mRow1[c0]) * fc
        applySeaTint(data, off, m0 + (m1 - m0) * fr)
      }
    }
  }

  ctx.putImageData(img, 0, 0)
}

export interface CanvasOverlayOptions {
  width: number
  height: number
  dpr: number
  isLightBackground?: boolean
  viewportBounds?: ViewportBounds
  landSeaMask?: Grid2D | null
  coastlineSegments?: ContourSegment[]
  tpSegments?: ContourSegment[]
  windArrows?: WindArrow[]
  diffWindArrows?: WindArrow[]
  shockConfig?: ShockConfig | null
  showShockOverlay?: boolean
  frontContourSegments?: ContourSegment[]
  localityRingSegments?: ContourSegment[]
  refRingSegments?: ContourSegment[]
  stepDayNumber?: number
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
  viewportBounds?: ViewportBounds,
): void {
  const bounds = viewportBounds ?? getViewportBounds(DEFAULT_MAP_VIEWPORT)
  const toX = (gx: number) => ((gx - bounds.gxMin) / bounds.spanGx) * width
  const toY = (gy: number) => ((gy - bounds.gyMin) / bounds.spanGy) * height

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

function strokeSegments(
  ctx: CanvasRenderingContext2D,
  segments: ContourSegment[],
  toX: (gx: number) => number,
  toY: (gy: number) => number,
): [number, number] | null {
  let anchor: [number, number] | null = null
  ctx.beginPath()
  for (let i = 0; i < segments.length; i++) {
    const [x1, y1, x2, y2] = segments[i]
    const px1 = toX(x1)
    const py1 = toY(y1)
    const px2 = toX(x2)
    const py2 = toY(y2)
    ctx.moveTo(px1, py1)
    ctx.lineTo(px2, py2)
    if (!anchor || py1 < anchor[1]) {
      anchor = [(px1 + px2) / 2, (py1 + py2) / 2]
    }
  }
  ctx.stroke()
  return anchor
}

function drawInlineTag(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  width: number,
  height: number,
  dpr: number,
  textColor: string,
): void {
  ctx.save()
  ctx.font = `400 ${Math.round(9.5 * dpr)}px 'Geist Mono', ui-monospace, monospace`
  const tw = ctx.measureText(text).width
  const padX = 4 * dpr
  const bw = tw + padX * 2
  const bh = 14 * dpr
  const bx = Math.max(3 * dpr, Math.min(width - bw - 3 * dpr, x - bw / 2))
  const by = Math.max(3 * dpr, Math.min(height - bh - 3 * dpr, y - bh / 2))
  ctx.fillStyle = 'rgba(13, 18, 16, 0.86)'
  ctx.strokeStyle = '#26312B'
  ctx.lineWidth = 1 * dpr
  ctx.fillRect(bx, by, bw, bh)
  ctx.strokeRect(bx, by, bw, bh)
  ctx.fillStyle = textColor
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, bx + bw / 2, by + bh / 2 + 0.5 * dpr)
  ctx.restore()
}

/**
 * Draw lat/lon 5-degree graticule lines (no geographic outlines), shock region,
 * response front contour, 3-cell locality ring, 8 cells/day reference ring, and hover reticle.
 */
export function drawCanvasMapOverlays(
  ctx: CanvasRenderingContext2D,
  opts: CanvasOverlayOptions,
): void {
  const {
    width,
    height,
    dpr,
    isLightBackground = false,
    viewportBounds,
    landSeaMask,
    coastlineSegments,
    tpSegments,
    windArrows,
    diffWindArrows,
    shockConfig,
    showShockOverlay = false,
    frontContourSegments,
    localityRingSegments,
    refRingSegments,
    stepDayNumber,
    hoveredCell,
    drawGraticule = true,
  } = opts

  const bounds = viewportBounds ?? getViewportBounds(DEFAULT_MAP_VIEWPORT)
  const toX = (gx: number) => ((gx - bounds.gxMin) / bounds.spanGx) * width
  const toY = (gy: number) => ((gy - bounds.gyMin) / bounds.spanGy) * height

  // 1. Crisp 1px Lat/Lon graticule lines (dynamic step when zoomed, zero geographic outlines)
  if (drawGraticule) {
    const ticks = getViewportGraticuleTicks(bounds)
    ctx.save()
    ctx.strokeStyle = 'rgba(90, 100, 93, 0.36)'
    ctx.lineWidth = 1 * dpr

    ctx.beginPath()
    for (const latTick of ticks.lats) {
      const y = Math.round(latTick.normPos * height) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
    }
    for (const lonTick of ticks.lons) {
      const x = Math.round(lonTick.normPos * width) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
    }
    ctx.stroke()

    // Crisp edge tick marks
    ctx.strokeStyle = 'rgba(140, 149, 141, 0.65)'
    ctx.lineWidth = 1 * dpr
    const tickLen = 4 * dpr
    ctx.beginPath()
    for (const latTick of ticks.lats) {
      const y = Math.round(latTick.normPos * height) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(tickLen, y)
      ctx.moveTo(width - tickLen, y)
      ctx.lineTo(width, y)
    }
    for (const lonTick of ticks.lons) {
      const x = Math.round(lonTick.normPos * width) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, tickLen)
      ctx.moveTo(x, height - tickLen)
      ctx.lineTo(x, height)
    }
    ctx.stroke()
    ctx.restore()
  }

  // 1b. Imported land/sea coastline outline (1px bone #ECE7DA line via marching squares)
  if (coastlineSegments && coastlineSegments.length > 0) {
    ctx.save()
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'rgba(13, 18, 16, 0.65)'
    ctx.lineWidth = 2.1 * dpr
    strokeSegments(ctx, coastlineSegments, toX, toY)

    ctx.strokeStyle = '#ECE7DA'
    ctx.lineWidth = 1.0 * dpr
    strokeSegments(ctx, coastlineSegments, toX, toY)
    ctx.restore()
  }

  // 2. Field contours & wind vectors
  drawFieldVectorsAndContours(ctx, width, height, dpr, tpSegments, windArrows, bounds)

  // 2b. Difference-mode (Branch − Prime) wind arrows
  if (diffWindArrows && diffWindArrows.length > 0) {
    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    for (let i = 0; i < diffWindArrows.length; i++) {
      const a = diffWindArrows[i]
      const x2 = toX(a.x2)
      const y2 = toY(a.y2)
      ctx.moveTo(toX(a.x1), toY(a.y1))
      ctx.lineTo(x2, y2)
      ctx.moveTo(toX(a.headA[0]), toY(a.headA[1]))
      ctx.lineTo(x2, y2)
      ctx.lineTo(toX(a.headB[0]), toY(a.headB[1]))
    }
    ctx.strokeStyle = 'rgba(236, 231, 218, 0.75)'
    ctx.lineWidth = 2.4 * dpr
    ctx.stroke()

    ctx.strokeStyle = '#141B18'
    ctx.lineWidth = 1.2 * dpr
    ctx.stroke()
    ctx.restore()
  }

  // 3. Reference dashed rings: 3-cell locality ring & 8 cells/day reference ring
  if (localityRingSegments && localityRingSegments.length > 0) {
    ctx.save()
    ctx.setLineDash([3.5 * dpr, 3.5 * dpr])
    ctx.strokeStyle = isLightBackground ? 'rgba(13, 18, 16, 0.35)' : 'rgba(13, 18, 16, 0.55)'
    ctx.lineWidth = 2.2 * dpr
    strokeSegments(ctx, localityRingSegments, toX, toY)

    ctx.strokeStyle = isLightBackground ? '#2F6F73' : '#7FB5B0'
    ctx.lineWidth = 1.15 * dpr
    const locAnchor = strokeSegments(ctx, localityRingSegments, toX, toY)
    ctx.restore()

    if (locAnchor && width >= 260) {
      drawInlineTag(ctx, '3c locality', locAnchor[0], locAnchor[1], width, height, dpr, '#7FB5B0')
    }
  }

  if (refRingSegments && refRingSegments.length > 0) {
    ctx.save()
    ctx.setLineDash([6 * dpr, 4 * dpr])
    ctx.strokeStyle = isLightBackground ? 'rgba(90, 100, 93, 0.72)' : 'rgba(140, 149, 141, 0.65)'
    ctx.lineWidth = 1.15 * dpr
    const refAnchor = strokeSegments(ctx, refRingSegments, toX, toY)
    ctx.restore()

    if (refAnchor && width >= 260) {
      const tag = stepDayNumber ? `8c/d (${stepDayNumber * 8}c)` : '8c/d ref'
      drawInlineTag(ctx, tag, refAnchor[0], refAnchor[1], width, height, dpr, '#8C958D')
    }
  }

  // 4. Shock footprint outlined in solid lightning #F0B84B
  if (showShockOverlay && shockConfig) {
    ctx.save()
    const hasCustomClip =
      (shockConfig.edgeFeather ?? 0) > 0 ||
      ((shockConfig.domainMask ?? 'all') !== 'all' && Boolean(landSeaMask))

    if (shockConfig.shape === 'box' && !hasCustomClip) {
      const cMin = Math.min(shockConfig.boxColMin, shockConfig.boxColMax)
      const cMax = Math.max(shockConfig.boxColMin, shockConfig.boxColMax)
      const rMin = Math.min(shockConfig.boxRowMin, shockConfig.boxRowMax)
      const rMax = Math.max(shockConfig.boxRowMin, shockConfig.boxRowMax)

      const x = toX(cMin)
      const y = toY(GRID_ROWS - 1 - rMax)
      const w = toX(cMax + 1) - x
      const h = toY(GRID_ROWS - rMin) - y

      ctx.strokeStyle = 'rgba(13, 18, 16, 0.78)'
      ctx.lineWidth = 2.8 * dpr
      ctx.strokeRect(x, y, w, h)

      ctx.strokeStyle = '#F0B84B'
      ctx.lineWidth = 1.5 * dpr
      ctx.strokeRect(x, y, w, h)
    } else {
      const weights = computeSpatialWeights(shockConfig, landSeaMask)
      const weightField = buildNorthUpScalarVertexField(weights, CONTOUR_NX, CONTOUR_NY, false)
      const coreLevel = shockConfig.shape === 'gaussian' ? Math.exp(-0.5) : 0.5
      const coreSegs = extractIsoContourSegments(
        weightField,
        CONTOUR_NX,
        CONTOUR_NY,
        coreLevel,
      )
      const outerSegs = extractIsoContourSegments(weightField, CONTOUR_NX, CONTOUR_NY, 0.08)

      if (outerSegs.length > 0) {
        ctx.save()
        ctx.setLineDash([4 * dpr, 3 * dpr])
        ctx.strokeStyle = 'rgba(240, 184, 75, 0.68)'
        ctx.lineWidth = 1.1 * dpr
        strokeSegments(ctx, outerSegs, toX, toY)
        ctx.restore()
      }

      if (coreSegs.length > 0) {
        ctx.strokeStyle = 'rgba(13, 18, 16, 0.78)'
        ctx.lineWidth = 2.8 * dpr
        strokeSegments(ctx, coreSegs, toX, toY)

        ctx.strokeStyle = '#F0B84B'
        ctx.lineWidth = 1.5 * dpr
        strokeSegments(ctx, coreSegs, toX, toY)
      }
    }
    ctx.restore()
  }

  // 5. Response front contour: |Branch - Prime| at 5% of rollout peak
  if (frontContourSegments && frontContourSegments.length > 0) {
    ctx.save()
    ctx.lineCap = 'round'
    ctx.strokeStyle = isLightBackground ? 'rgba(236, 231, 218, 0.78)' : 'rgba(13, 18, 16, 0.8)'
    ctx.lineWidth = 3.0 * dpr
    strokeSegments(ctx, frontContourSegments, toX, toY)

    ctx.strokeStyle = isLightBackground ? '#0D1210' : '#ECE7DA'
    ctx.lineWidth = 1.5 * dpr
    const frontAnchor = strokeSegments(ctx, frontContourSegments, toX, toY)
    ctx.restore()

    if (frontAnchor && width >= 260) {
      const label = stepDayNumber ? `D${stepDayNumber} 5% front` : '5% front'
      drawInlineTag(
        ctx,
        label,
        frontAnchor[0],
        frontAnchor[1] - 8 * dpr,
        width,
        height,
        dpr,
        '#ECE7DA',
      )
    }
  }

  // 6. Hovered cell reticle
  if (hoveredCell) {
    const cellW = width / bounds.spanGx
    const cellH = height / bounds.spanGy
    const x = toX(hoveredCell.col)
    const y = toY(GRID_ROWS - 1 - hoveredCell.row)

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
