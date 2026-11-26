export const GRID_ROWS = 32 as const
export const GRID_COLS = 36 as const
export const NUM_CHANNELS = 6 as const
export const INIT_DAYS = 3 as const

/**
 * Channel order on the wire (axis 1):
 * 0: t2m (K)
 * 1: tp (mm/day)
 * 2: u10 (m/s)
 * 3: v10 (m/s)
 * 4: msl (Pa)
 * 5: tcwv (kg/m2)
 */
export type ChannelIndex = 0 | 1 | 2 | 3 | 4 | 5

export type ChannelKey = 't2m' | 'tp' | 'u10' | 'v10' | 'msl' | 'tcwv'

export type DisplayChannelId = ChannelKey | 'wind'

/** 2D grid [32][36] (row 0 = lat 6.5, col 0 = lon 65.5) */
export type Grid2D = number[][]

/** 3D state [6][32][36] in physical wire units */
export type State3D = Grid2D[]

/** 4D sequence [T][6][32][36] in physical wire units */
export type StateSequence = State3D[]

/** [sin(2*pi*doy/365), cos(2*pi*doy/365)] for the last input day */
export type DoySinCos = [number, number]

/** Imported scenario held in memory only */
export interface ScenarioData {
  /** 3 consecutive days, oldest first: [3][6][32][36] */
  init_states: StateSequence
  /** Day-of-year sin/cos encoding for the last input day */
  doy_sincos: DoySinCos
  /** Optional source label (e.g. filename or "Pasted JSON") */
  sourceLabel?: string
}

/** POST /health response */
export interface HealthResponse {
  status: 'ready' | string
  model_name: string
}

/** POST /simulate request body */
export interface SimulateRequest {
  init_states: StateSequence
  doy_sincos: DoySinCos
  steps: number
  perturbation: State3D | null
}

/** POST /simulate response body */
export interface SimulateResponse {
  rollout: StateSequence
  inference_ms: number
}
