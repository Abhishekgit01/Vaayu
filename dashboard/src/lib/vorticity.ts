import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type Grid2D,
  type State3D,
} from '../api/types'
import { CELL_DEG, rowToLat } from './constants'

export type ViewFieldId = DisplayChannelId | 'vorticity'

export const VORTICITY_META = {
  id: 'vorticity' as const,
  label: 'Rotation (relative vorticity)',
  shortLabel: 'ζ',
  displayUnit: '×10⁻⁵ s⁻¹',
  wireUnit: 'computed curl(u10, v10)',
  decimals: 2,
  divergingBase: true,
  isComputed: true,
}

const EARTH_RADIUS_M = 6_371_000
const DEG_TO_RAD = Math.PI / 180
const D_LAT_RAD = CELL_DEG * DEG_TO_RAD
const D_LON_RAD = CELL_DEG * DEG_TO_RAD

const COS_LAT: readonly number[] = Array.from({ length: GRID_ROWS }, (_, r) =>
  Math.cos(rowToLat(r) * DEG_TO_RAD),
)

/**
 * Compute vertical relative vorticity ζ = (∇ × V)_z on the 1°×1° spherical grid
 * strictly from the rollout's (u10, v10) channels (in units of 10⁻⁵ s⁻¹):
 *
 *   ζ = [ 1 / (R cos φ) ] * [ ∂v10 / ∂λ − ∂(u10 cos φ) / ∂φ ]
 *
 * Positive values indicate cyclonic rotation (counter-clockwise in Northern Hemisphere);
 * negative values indicate anticyclonic rotation; exact 0 is irrotational.
 */
export function computeRelativeVorticityGrid(state: State3D): Grid2D {
  const uGrid = state[2]
  const vGrid = state[3]
  const out: Grid2D = new Array(GRID_ROWS)

  for (let r = 0; r < GRID_ROWS; r++) {
    const cosPhi = Math.max(0.05, COS_LAT[r])
    const rowOut = new Array<number>(GRID_COLS)

    const rSouth = Math.max(0, r - 1)
    const rNorth = Math.min(GRID_ROWS - 1, r + 1)
    const dPhi = (rNorth - rSouth) * D_LAT_RAD
    const cosSouth = COS_LAT[rSouth]
    const cosNorth = COS_LAT[rNorth]

    for (let c = 0; c < GRID_COLS; c++) {
      const cWest = Math.max(0, c - 1)
      const cEast = Math.min(GRID_COLS - 1, c + 1)
      const dLambda = (cEast - cWest) * D_LON_RAD

      const dvDlambda = (vGrid[r][cEast] - vGrid[r][cWest]) / dLambda
      const duCosDphi =
        (uGrid[rNorth][c] * cosNorth - uGrid[rSouth][c] * cosSouth) / dPhi

      const zetaPerSec = (dvDlambda - duCosDphi) / (EARTH_RADIUS_M * cosPhi)
      // Convert s⁻¹ to standard meteorological 10⁻⁵ s⁻¹
      const scaled = zetaPerSec * 1e5
      rowOut[c] = Math.abs(scaled) < 1e-12 ? 0 : scaled
    }

    out[r] = rowOut
  }

  return out
}
