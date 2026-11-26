import {
  GRID_COLS,
  GRID_ROWS,
  type ChannelIndex,
  type ChannelKey,
  type DisplayChannelId,
  type Grid2D,
  type State3D,
} from '../api/types'

export const LAT_ORIGIN = 6.5
export const LON_ORIGIN = 65.5
export const CELL_DEG = 1.0

export function rowToLat(row: number): number {
  return LAT_ORIGIN + row * CELL_DEG
}

export function colToLon(col: number): number {
  return LON_ORIGIN + col * CELL_DEG
}

export function latToRow(lat: number): number {
  return Math.max(0, Math.min(GRID_ROWS - 1, Math.round((lat - LAT_ORIGIN) / CELL_DEG)))
}

export function lonToCol(lon: number): number {
  return Math.max(0, Math.min(GRID_COLS - 1, Math.round((lon - LON_ORIGIN) / CELL_DEG)))
}

export interface ChannelMeta {
  index: ChannelIndex
  key: ChannelKey
  label: string
  shortLabel: string
  wireUnit: string
  displayUnit: string
  deltaUnit: string
  decimals: number
  toDisplay: (wireValue: number) => number
  deltaToDisplay: (wireDelta: number) => number
}

export const CHANNELS: readonly ChannelMeta[] = [
  {
    index: 0,
    key: 't2m',
    label: '2m Temperature',
    shortLabel: 't2m',
    wireUnit: 'K',
    displayUnit: '°C',
    deltaUnit: '°C',
    decimals: 1,
    toDisplay: (v) => v - 273.15,
    deltaToDisplay: (dv) => dv,
  },
  {
    index: 1,
    key: 'tp',
    label: 'Total Precipitation',
    shortLabel: 'tp',
    wireUnit: 'mm/day',
    displayUnit: 'mm/day',
    deltaUnit: 'mm/day',
    decimals: 1,
    toDisplay: (v) => v,
    deltaToDisplay: (dv) => dv,
  },
  {
    index: 2,
    key: 'u10',
    label: '10m Zonal Wind (u)',
    shortLabel: 'u10',
    wireUnit: 'm/s',
    displayUnit: 'm/s',
    deltaUnit: 'm/s',
    decimals: 1,
    toDisplay: (v) => v,
    deltaToDisplay: (dv) => dv,
  },
  {
    index: 3,
    key: 'v10',
    label: '10m Meridional Wind (v)',
    shortLabel: 'v10',
    wireUnit: 'm/s',
    displayUnit: 'm/s',
    deltaUnit: 'm/s',
    decimals: 1,
    toDisplay: (v) => v,
    deltaToDisplay: (dv) => dv,
  },
  {
    index: 4,
    key: 'msl',
    label: 'Mean Sea-Level Pressure',
    shortLabel: 'msl',
    wireUnit: 'Pa',
    displayUnit: 'hPa',
    deltaUnit: 'hPa',
    decimals: 1,
    toDisplay: (v) => v / 100,
    deltaToDisplay: (dv) => dv / 100,
  },
  {
    index: 5,
    key: 'tcwv',
    label: 'Total Column Water Vapor',
    shortLabel: 'tcwv',
    wireUnit: 'kg/m²',
    displayUnit: 'kg/m²',
    deltaUnit: 'kg/m²',
    decimals: 1,
    toDisplay: (v) => v,
    deltaToDisplay: (dv) => dv,
  },
] as const

export const CHANNEL_BY_KEY: Record<ChannelKey, ChannelMeta> = {
  t2m: CHANNELS[0],
  tp: CHANNELS[1],
  u10: CHANNELS[2],
  v10: CHANNELS[3],
  msl: CHANNELS[4],
  tcwv: CHANNELS[5],
}

export interface DisplayChannelMeta {
  id: DisplayChannelId
  label: string
  shortLabel: string
  displayUnit: string
  wireUnit: string
  decimals: number
  divergingBase?: boolean
}

export const DISPLAY_CHANNELS: readonly DisplayChannelMeta[] = [
  {
    id: 't2m',
    label: '2m Temperature',
    shortLabel: 't2m',
    displayUnit: '°C',
    wireUnit: 'K',
    decimals: 1,
  },
  {
    id: 'tp',
    label: 'Precipitation',
    shortLabel: 'tp',
    displayUnit: 'mm/day',
    wireUnit: 'mm/day',
    decimals: 1,
  },
  {
    id: 'wind',
    label: '10m Wind Speed',
    shortLabel: 'wind',
    displayUnit: 'm/s',
    wireUnit: 'm/s',
    decimals: 1,
  },
  {
    id: 'u10',
    label: '10m Zonal Wind (u10)',
    shortLabel: 'u10',
    displayUnit: 'm/s',
    wireUnit: 'm/s',
    decimals: 1,
    divergingBase: true,
  },
  {
    id: 'v10',
    label: '10m Meridional Wind (v10)',
    shortLabel: 'v10',
    displayUnit: 'm/s',
    wireUnit: 'm/s',
    decimals: 1,
    divergingBase: true,
  },
  {
    id: 'msl',
    label: 'Sea-Level Pressure',
    shortLabel: 'msl',
    displayUnit: 'hPa',
    wireUnit: 'Pa',
    decimals: 1,
  },
  {
    id: 'tcwv',
    label: 'Column Water Vapor',
    shortLabel: 'tcwv',
    displayUnit: 'kg/m²',
    wireUnit: 'kg/m²',
    decimals: 1,
  },
] as const

export function getDisplayChannelMeta(id: DisplayChannelId): DisplayChannelMeta {
  const found = DISPLAY_CHANNELS.find((c) => c.id === id)
  return found ?? DISPLAY_CHANNELS[0]
}

/**
 * Extract a 32x36 grid in display units for the selected channel from a 3D physical state.
 */
export function extractDisplayGrid(state: State3D, channel: DisplayChannelId): Grid2D {
  const out: Grid2D = new Array(GRID_ROWS)
  if (channel === 'wind') {
    const uGrid = state[2]
    const vGrid = state[3]
    for (let r = 0; r < GRID_ROWS; r++) {
      const row = new Array<number>(GRID_COLS)
      for (let c = 0; c < GRID_COLS; c++) {
        const u = uGrid[r][c]
        const v = vGrid[r][c]
        row[c] = Math.hypot(u, v)
      }
      out[r] = row
    }
    return out
  }

  const meta = CHANNEL_BY_KEY[channel]
  const src = state[meta.index]
  for (let r = 0; r < GRID_ROWS; r++) {
    const row = new Array<number>(GRID_COLS)
    for (let c = 0; c < GRID_COLS; c++) {
      row[c] = meta.toDisplay(src[r][c])
    }
    out[r] = row
  }
  return out
}

/**
 * Compute Branch - Prime in display units for the selected channel.
 */
export function computeDiffGrid(
  primeState: State3D,
  branchState: State3D,
  channel: DisplayChannelId,
): Grid2D {
  const primeGrid = extractDisplayGrid(primeState, channel)
  const branchGrid = extractDisplayGrid(branchState, channel)
  const out: Grid2D = new Array(GRID_ROWS)
  for (let r = 0; r < GRID_ROWS; r++) {
    const row = new Array<number>(GRID_COLS)
    for (let c = 0; c < GRID_COLS; c++) {
      row[c] = branchGrid[r][c] - primeGrid[r][c]
    }
    out[r] = row
  }
  return out
}

export function formatCoordLat(lat: number): string {
  return `${lat.toFixed(1)}°N`
}

export function formatCoordLon(lon: number): string {
  return `${lon.toFixed(1)}°E`
}
