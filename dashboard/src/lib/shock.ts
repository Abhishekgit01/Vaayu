import {
  GRID_COLS,
  GRID_ROWS,
  NUM_CHANNELS,
  type ChannelKey,
  type State3D,
} from '../api/types'
import { CHANNEL_BY_KEY, CHANNELS } from './constants'

export type ShockShape = 'box' | 'gaussian'
export type AmplitudeMode = 'absolute' | 'percent'

export interface GridPoint {
  row: number
  col: number
}

export interface ChannelShockSpec {
  enabled: boolean
  mode: AmplitudeMode
  value: number
}

export interface ShockConfig {
  shape: ShockShape
  /** Inclusive grid row bounds [0..31] for box shape */
  boxRowMin: number
  boxRowMax: number
  /** Inclusive grid col bounds [0..35] for box shape */
  boxColMin: number
  boxColMax: number
  /** Center & spread (in cells) for gaussian shape */
  centerRow: number
  centerCol: number
  gaussianSigma: number
  /** Painted Gaussian brush stroke points in grid coordinates */
  gaussianStroke?: GridPoint[]
  /** Internal drag-origin tracker to append smoothly during a Gaussian paint drag */
  strokeOrigin?: GridPoint | null
  /** Per-channel delta specifications */
  channels: Record<ChannelKey, ChannelShockSpec>
  /** Active preset identifier, if any */
  presetName: string | null
}

// In-memory reference to the active scenario's init_states[2] once seen by buildPerturbation,
// cleared whenever getShockMask(null) runs on scenario import/reset. Never persisted.
let cachedLastInputDay: State3D | null = null
const inputDayListeners = new Set<() => void>()

function setCachedLastInputDay(next: State3D | null): void {
  if (cachedLastInputDay === next) return
  cachedLastInputDay = next
  for (const listener of inputDayListeners) {
    listener()
  }
}

export function getCachedLastInputDay(): State3D | null {
  return cachedLastInputDay
}

export function subscribeLastInputDay(listener: () => void): () => void {
  inputDayListeners.add(listener)
  return () => {
    inputDayListeners.delete(listener)
  }
}

export function createEmptyShockConfig(): ShockConfig {
  return {
    shape: 'box',
    boxRowMin: 8,
    boxRowMax: 19,
    boxColMin: 12,
    boxColMax: 23,
    centerRow: 14,
    centerCol: 18,
    gaussianSigma: 3.0,
    gaussianStroke: [{ row: 14, col: 18 }],
    strokeOrigin: null,
    channels: {
      t2m: { enabled: false, mode: 'absolute', value: 0 },
      tp: { enabled: false, mode: 'absolute', value: 0 },
      u10: { enabled: false, mode: 'percent', value: 0 },
      v10: { enabled: false, mode: 'percent', value: 0 },
      msl: { enabled: false, mode: 'absolute', value: 0 },
      tcwv: { enabled: false, mode: 'percent', value: 0 },
    },
    presetName: null,
  }
}

/**
 * Preset: "Warm Bay of Bengal"
 * Snaps box to lat 8–20°N (rows 2..13, cell edges 8.0°–20.0°N, centers 8.5°–19.5°N)
 * and lon 82–94°E (cols 17..28, cell edges 82.0°–94.0°E, centers 82.5°–93.5°E).
 * +3.0 K on t2m and +15% of last input day tcwv.
 */
export function createWarmBayOfBengalPreset(): ShockConfig {
  const base = createEmptyShockConfig()
  return {
    ...base,
    shape: 'box',
    boxRowMin: 2,
    boxRowMax: 13,
    boxColMin: 17,
    boxColMax: 28,
    centerRow: 7.5,
    centerCol: 22.5,
    gaussianSigma: 4.0,
    gaussianStroke: [{ row: 7.5, col: 22.5 }],
    strokeOrigin: null,
    channels: {
      ...base.channels,
      t2m: { enabled: true, mode: 'absolute', value: 3.0 },
      tcwv: { enabled: true, mode: 'percent', value: 15 },
    },
    presetName: 'Warm Bay of Bengal',
  }
}

/**
 * Preset: "Weaker Arabian Sea wind"
 * Snaps box to lat 8–20°N (rows 2..13, cell edges 8.0°–20.0°N, centers 8.5°–19.5°N)
 * and lon 65–75°E (cols 0..9, cell edges 65.0°–75.0°E, centers 65.5°–74.5°E).
 * -40% of last input day u10 and v10.
 */
export function createWeakerArabianSeaWindPreset(): ShockConfig {
  const base = createEmptyShockConfig()
  return {
    ...base,
    shape: 'box',
    boxRowMin: 2,
    boxRowMax: 13,
    boxColMin: 0,
    boxColMax: 9,
    centerRow: 7.5,
    centerCol: 4.5,
    gaussianSigma: 3.5,
    gaussianStroke: [{ row: 7.5, col: 4.5 }],
    strokeOrigin: null,
    channels: {
      ...base.channels,
      u10: { enabled: true, mode: 'percent', value: -40 },
      v10: { enabled: true, mode: 'percent', value: -40 },
    },
    presetName: 'Weaker Arabian Sea wind',
  }
}

export function getGaussianStrokePoints(config: ShockConfig): GridPoint[] {
  if (config.gaussianStroke && config.gaussianStroke.length > 0) {
    return config.gaussianStroke
  }
  return [{ row: config.centerRow, col: config.centerCol }]
}

/**
 * Spatial weight [0..1] for each cell (row, col), strictly 0 outside the shock region.
 */
export function computeSpatialWeights(config: ShockConfig): number[][] {
  const weights: number[][] = new Array(GRID_ROWS)
  const rMin = Math.max(0, Math.min(config.boxRowMin, config.boxRowMax))
  const rMax = Math.min(GRID_ROWS - 1, Math.max(config.boxRowMin, config.boxRowMax))
  const cMin = Math.max(0, Math.min(config.boxColMin, config.boxColMax))
  const cMax = Math.min(GRID_COLS - 1, Math.max(config.boxColMin, config.boxColMax))

  const sigma = Math.max(0.5, config.gaussianSigma)
  const cutoffRadius = sigma * 2.5
  const cutoffSq = cutoffRadius * cutoffRadius
  const twoSigmaSq = 2 * sigma * sigma
  const strokePts = config.shape === 'gaussian' ? getGaussianStrokePoints(config) : []

  for (let r = 0; r < GRID_ROWS; r++) {
    const row = new Array<number>(GRID_COLS).fill(0)
    for (let c = 0; c < GRID_COLS; c++) {
      if (config.shape === 'box') {
        if (r >= rMin && r <= rMax && c >= cMin && c <= cMax) {
          row[c] = 1
        }
      } else {
        let minD2 = Infinity
        for (let i = 0; i < strokePts.length; i++) {
          const dr = r - strokePts[i].row
          const dc = c - strokePts[i].col
          const d2 = dr * dr + dc * dc
          if (d2 < minD2) minD2 = d2
        }
        if (minD2 <= cutoffSq) {
          row[c] = Math.exp(-minD2 / twoSigmaSq)
        }
      }
    }
    weights[r] = row
  }
  return weights
}

/**
 * Build a [6][32][36] array of physical deltas, strictly 0 outside the shock.
 * Percentage shocks use the last input day (init_states[2]) per cell.
 */
export function buildPerturbation(config: ShockConfig, lastInputDay: State3D): State3D {
  setCachedLastInputDay(lastInputDay)
  const weights = computeSpatialWeights(config)
  const out: State3D = new Array(NUM_CHANNELS)

  for (const ch of CHANNELS) {
    const spec = config.channels[ch.key]
    const grid: number[][] = new Array(GRID_ROWS)
    const baseField = lastInputDay[ch.index]

    for (let r = 0; r < GRID_ROWS; r++) {
      const row = new Array<number>(GRID_COLS).fill(0)
      if (spec.enabled && spec.value !== 0) {
        for (let c = 0; c < GRID_COLS; c++) {
          const w = weights[r][c]
          if (w > 0) {
            if (spec.mode === 'percent') {
              row[c] = w * (spec.value / 100) * baseField[r][c]
            } else {
              row[c] = w * spec.value
            }
          }
        }
      }
      grid[r] = row
    }
    out[ch.index] = grid
  }

  return out
}

/**
 * Build an explicit [6][32][36] all-zero perturbation array for Prime vs Zero-Shock verification.
 */
export function buildZeroPerturbation(): State3D {
  const out: State3D = new Array(NUM_CHANNELS)
  for (let ch = 0; ch < NUM_CHANNELS; ch++) {
    const grid: number[][] = new Array(GRID_ROWS)
    for (let r = 0; r < GRID_ROWS; r++) {
      grid[r] = new Array<number>(GRID_COLS).fill(0)
    }
    out[ch] = grid
  }
  return out
}

/**
 * Extract a boolean [32][36] mask of cells belonging to the shock region.
 * Uses nonzero perturbation cells if present, otherwise falls back to the configured footprint.
 */
export function getShockMask(perturbation: State3D | null, fallbackConfig?: ShockConfig): boolean[][] {
  if (perturbation === null && fallbackConfig === undefined) {
    setCachedLastInputDay(null)
  }

  const mask: boolean[][] = Array.from({ length: GRID_ROWS }, () =>
    new Array<boolean>(GRID_COLS).fill(false),
  )

  let anyNonZero = false
  if (perturbation) {
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        for (let ch = 0; ch < NUM_CHANNELS; ch++) {
          if (perturbation[ch][r][c] !== 0) {
            mask[r][c] = true
            anyNonZero = true
            break
          }
        }
      }
    }
  }

  if (!anyNonZero && fallbackConfig) {
    const weights = computeSpatialWeights(fallbackConfig)
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        mask[r][c] = weights[r][c] > 0
      }
    }
  }

  return mask
}

export interface ChannelDeltaPreview {
  key: ChannelKey
  shortLabel: string
  wireUnit: string
  mode: AmplitudeMode
  inputValue: number
  hasPhysicalBase: boolean
  /** Signed delta with largest magnitude across the shock footprint */
  peakSignedDelta: number
  /** Mean signed delta across active shock cells (w > 0) */
  meanSignedDelta: number
}

export interface ShockFootprintPreview {
  weights: number[][]
  activeCells: number
  effectiveCells: number
  channelReadouts: ChannelDeltaPreview[]
}

/**
 * Compute live footprint weights and numeric delta readouts for the Shock Panel.
 * Percentage channels resolve exact physical deltas whenever init_states[2] is available.
 */
export function computeShockPreview(
  config: ShockConfig,
  lastInputDay: State3D | null,
): ShockFootprintPreview {
  const weights = computeSpatialWeights(config)
  let activeCells = 0
  let effectiveCells = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const w = weights[r][c]
      if (w > 0) {
        activeCells += 1
        effectiveCells += w
      }
    }
  }

  const channelReadouts: ChannelDeltaPreview[] = []

  for (const ch of CHANNELS) {
    const spec = config.channels[ch.key]
    if (!spec.enabled || spec.value === 0) continue

    if (spec.mode === 'absolute') {
      let peakSigned = 0
      let sum = 0
      for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
          const w = weights[r][c]
          if (w > 0) {
            const d = w * spec.value
            sum += d
            if (Math.abs(d) > Math.abs(peakSigned)) peakSigned = d
          }
        }
      }
      channelReadouts.push({
        key: ch.key,
        shortLabel: ch.shortLabel,
        wireUnit: ch.wireUnit,
        mode: 'absolute',
        inputValue: spec.value,
        hasPhysicalBase: true,
        peakSignedDelta: peakSigned,
        meanSignedDelta: activeCells > 0 ? sum / activeCells : 0,
      })
    } else if (lastInputDay) {
      const baseField = lastInputDay[ch.index]
      let peakSigned = 0
      let sum = 0
      for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
          const w = weights[r][c]
          if (w > 0) {
            const d = w * (spec.value / 100) * baseField[r][c]
            sum += d
            if (Math.abs(d) > Math.abs(peakSigned)) peakSigned = d
          }
        }
      }
      channelReadouts.push({
        key: ch.key,
        shortLabel: ch.shortLabel,
        wireUnit: ch.wireUnit,
        mode: 'percent',
        inputValue: spec.value,
        hasPhysicalBase: true,
        peakSignedDelta: peakSigned,
        meanSignedDelta: activeCells > 0 ? sum / activeCells : 0,
      })
    } else {
      // Before first simulation run when only relative % weight is known
      let peakPct = 0
      let sumPct = 0
      for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
          const w = weights[r][c]
          if (w > 0) {
            const p = w * spec.value
            sumPct += p
            if (Math.abs(p) > Math.abs(peakPct)) peakPct = p
          }
        }
      }
      channelReadouts.push({
        key: ch.key,
        shortLabel: ch.shortLabel,
        wireUnit: ch.wireUnit,
        mode: 'percent',
        inputValue: spec.value,
        hasPhysicalBase: false,
        peakSignedDelta: peakPct,
        meanSignedDelta: activeCells > 0 ? sumPct / activeCells : 0,
      })
    }
  }

  return {
    weights,
    activeCells,
    effectiveCells,
    channelReadouts,
  }
}

/**
 * Helper to summarize enabled channels in readable form.
 */
export function summarizeShockChannels(config: ShockConfig): string {
  const parts: string[] = []
  for (const ch of CHANNELS) {
    const spec = config.channels[ch.key]
    if (spec.enabled && spec.value !== 0) {
      const sign = spec.value > 0 ? '+' : ''
      if (spec.mode === 'percent') {
        parts.push(`${ch.shortLabel} ${sign}${spec.value}%`)
      } else {
        parts.push(`${ch.shortLabel} ${sign}${spec.value} ${CHANNEL_BY_KEY[ch.key].wireUnit}`)
      }
    }
  }
  return parts.length > 0 ? parts.join(', ') : 'No active channels (zero delta)'
}

function interpolateStrokeCells(from: GridPoint, to: GridPoint): GridPoint[] {
  const dr = to.row - from.row
  const dc = to.col - from.col
  const steps = Math.max(1, Math.ceil(Math.hypot(dr, dc) * 1.5))
  const pts: GridPoint[] = []
  for (let i = 1; i <= steps; i++) {
    const t = i / steps
    const r = Math.max(0, Math.min(GRID_ROWS - 1, Math.round(from.row + dr * t)))
    const c = Math.max(0, Math.min(GRID_COLS - 1, Math.round(from.col + dc * t)))
    const last = pts[pts.length - 1]
    if (!last || last.row !== r || last.col !== c) {
      pts.push({ row: r, col: c })
    }
  }
  return pts
}

export function moveShockCenter(config: ShockConfig, row: number, col: number): ShockConfig {
  const clampedRow = Math.max(0, Math.min(GRID_ROWS - 1, row))
  const clampedCol = Math.max(0, Math.min(GRID_COLS - 1, col))

  if (config.shape === 'gaussian') {
    return {
      ...config,
      centerRow: clampedRow,
      centerCol: clampedCol,
      gaussianStroke: [{ row: clampedRow, col: clampedCol }],
      strokeOrigin: null,
      presetName: null,
    }
  }

  const height = Math.max(1, Math.abs(config.boxRowMax - config.boxRowMin) + 1)
  const width = Math.max(1, Math.abs(config.boxColMax - config.boxColMin) + 1)
  let rMin = Math.round(clampedRow - (height - 1) / 2)
  let rMax = rMin + height - 1
  if (rMin < 0) {
    rMin = 0
    rMax = Math.min(GRID_ROWS - 1, height - 1)
  }
  if (rMax >= GRID_ROWS) {
    rMax = GRID_ROWS - 1
    rMin = Math.max(0, rMax - height + 1)
  }

  let cMin = Math.round(clampedCol - (width - 1) / 2)
  let cMax = cMin + width - 1
  if (cMin < 0) {
    cMin = 0
    cMax = Math.min(GRID_COLS - 1, width - 1)
  }
  if (cMax >= GRID_COLS) {
    cMax = GRID_COLS - 1
    cMin = Math.max(0, cMax - width + 1)
  }

  return {
    ...config,
    boxRowMin: rMin,
    boxRowMax: rMax,
    boxColMin: cMin,
    boxColMax: cMax,
    centerRow: (rMin + rMax) / 2,
    centerCol: (cMin + cMax) / 2,
    strokeOrigin: null,
    presetName: null,
  }
}

/**
 * Dragging on the map:
 * - In 'box' mode: drags out a bounding box [rowMin..rowMax, colMin..colMax].
 * - In 'gaussian' mode: paints a continuous Gaussian brush trail along the drag path.
 */
export function applyDragToShock(
  config: ShockConfig,
  startRow: number,
  startCol: number,
  endRow: number,
  endCol: number,
): ShockConfig {
  const r0 = Math.max(0, Math.min(GRID_ROWS - 1, startRow))
  const c0 = Math.max(0, Math.min(GRID_COLS - 1, startCol))
  const r1 = Math.max(0, Math.min(GRID_ROWS - 1, endRow))
  const c1 = Math.max(0, Math.min(GRID_COLS - 1, endCol))

  if (config.shape === 'gaussian') {
    const isSameGesture =
      config.strokeOrigin !== null &&
      config.strokeOrigin !== undefined &&
      config.strokeOrigin.row === r0 &&
      config.strokeOrigin.col === c0

    let nextStroke: GridPoint[]
    if (isSameGesture && config.gaussianStroke && config.gaussianStroke.length > 0) {
      const prevEnd = config.gaussianStroke[config.gaussianStroke.length - 1]
      if (prevEnd.row === r1 && prevEnd.col === c1) {
        return config
      }
      const added = interpolateStrokeCells(prevEnd, { row: r1, col: c1 })
      // Cap stroke buffer length so very long scribbles remain fast
      nextStroke = [...config.gaussianStroke, ...added].slice(-180)
    } else {
      const origin = { row: r0, col: c0 }
      nextStroke = [origin, ...interpolateStrokeCells(origin, { row: r1, col: c1 })]
    }

    return {
      ...config,
      centerRow: r1,
      centerCol: c1,
      gaussianStroke: nextStroke,
      strokeOrigin: { row: r0, col: c0 },
      presetName: null,
    }
  }

  const rowMin = Math.min(r0, r1)
  const rowMax = Math.max(r0, r1)
  const colMin = Math.min(c0, c1)
  const colMax = Math.max(c0, c1)
  return {
    ...config,
    boxRowMin: rowMin,
    boxRowMax: rowMax,
    boxColMin: colMin,
    boxColMax: colMax,
    centerRow: (rowMin + rowMax) / 2,
    centerCol: (colMin + colMax) / 2,
    strokeOrigin: null,
    presetName: null,
  }
}

/**
 * Marching-squares isoline contour extractor over a scalar vertex grid [ny + 1][nx + 1]
 * mapped into [0..GRID_COLS] x [0..GRID_ROWS] north-up map coordinates.
 */
export function extractIsoContourSegments(
  vertexField: number[][],
  nx: number,
  ny: number,
  level: number,
): Array<[number, number, number, number]> {
  const segments: Array<[number, number, number, number]> = []
  const dx = GRID_COLS / nx
  const dy = GRID_ROWS / ny

  const interp = (a: number, b: number) => {
    const denom = b - a
    if (Math.abs(denom) < 1e-12) return 0.5
    return Math.max(0, Math.min(1, (level - a) / denom))
  }

  for (let iy = 0; iy < ny; iy++) {
    const y0 = iy * dy
    const y1 = (iy + 1) * dy
    const rowTop = vertexField[iy]
    const rowBot = vertexField[iy + 1]

    for (let ix = 0; ix < nx; ix++) {
      const tl = rowTop[ix]
      const tr = rowTop[ix + 1]
      const br = rowBot[ix + 1]
      const bl = rowBot[ix]

      const code =
        (tl >= level ? 8 : 0) |
        (tr >= level ? 4 : 0) |
        (br >= level ? 2 : 0) |
        (bl >= level ? 1 : 0)

      if (code === 0 || code === 15) continue

      const x0 = ix * dx
      const x1 = (ix + 1) * dx

      const top: [number, number] = [x0 + dx * interp(tl, tr), y0]
      const right: [number, number] = [x1, y0 + dy * interp(tr, br)]
      const bottom: [number, number] = [x0 + dx * interp(bl, br), y1]
      const left: [number, number] = [x0, y0 + dy * interp(tl, bl)]

      switch (code) {
        case 1:
        case 14:
          segments.push([left[0], left[1], bottom[0], bottom[1]])
          break
        case 2:
        case 13:
          segments.push([bottom[0], bottom[1], right[0], right[1]])
          break
        case 3:
        case 12:
          segments.push([left[0], left[1], right[0], right[1]])
          break
        case 4:
        case 11:
          segments.push([top[0], top[1], right[0], right[1]])
          break
        case 5: {
          const center = 0.25 * (tl + tr + br + bl)
          if (center >= level) {
            segments.push([left[0], left[1], top[0], top[1]])
            segments.push([bottom[0], bottom[1], right[0], right[1]])
          } else {
            segments.push([left[0], left[1], bottom[0], bottom[1]])
            segments.push([top[0], top[1], right[0], right[1]])
          }
          break
        }
        case 6:
        case 9:
          segments.push([top[0], top[1], bottom[0], bottom[1]])
          break
        case 7:
        case 8:
          segments.push([left[0], left[1], top[0], top[1]])
          break
        case 10: {
          const center = 0.25 * (tl + tr + br + bl)
          if (center >= level) {
            segments.push([top[0], top[1], right[0], right[1]])
            segments.push([left[0], left[1], bottom[0], bottom[1]])
          } else {
            segments.push([left[0], left[1], top[0], top[1]])
            segments.push([bottom[0], bottom[1], right[0], right[1]])
          }
          break
        }
      }
    }
  }

  return segments
}
