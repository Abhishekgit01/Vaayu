import { GRID_COLS, GRID_ROWS, type Grid2D, type StateSequence } from '../api/types'
import { CELL_DEG, rowToLat } from './constants'

/** Mean spherical Earth radius (IUGG / WGS84) in km for exact 1°x1° quadrangle area */
const EARTH_RADIUS_KM = 6371.0
const DEG_TO_RAD = Math.PI / 180

/**
 * Exact spherical quadrangle surface area (km²) of a 1° x 1° grid cell at row r:
 * A(r) = R² * Δλ * (sin φ_north - sin φ_south)
 */
export function computeRowCellAreaKm2(row: number): number {
  const latCenter = rowToLat(row)
  const halfCell = CELL_DEG / 2
  const phiSouth = (latCenter - halfCell) * DEG_TO_RAD
  const phiNorth = (latCenter + halfCell) * DEG_TO_RAD
  const dLambda = CELL_DEG * DEG_TO_RAD
  return EARTH_RADIUS_KM * EARTH_RADIUS_KM * dLambda * (Math.sin(phiNorth) - Math.sin(phiSouth))
}

const ROW_CELL_AREAS_KM2: readonly number[] = Array.from({ length: GRID_ROWS }, (_, r) =>
  computeRowCellAreaKm2(r),
)

export const TOTAL_GRID_CELLS = GRID_ROWS * GRID_COLS

export const TOTAL_GRID_AREA_KM2: number = ROW_CELL_AREAS_KM2.reduce(
  (sum, rowArea) => sum + rowArea * GRID_COLS,
  0,
)

/**
 * Explicit, visible settings object controlling heatwave impact calculations.
 */
export interface ImpactSettings {
  /** Temperature threshold in °C; rollout days strictly above this value count as heatwave days */
  heatwaveThresholdC: number
}

export const DEFAULT_IMPACT_SETTINGS: ImpactSettings = {
  heatwaveThresholdC: 35,
}

export interface SingleImpactSummary {
  /** Per-cell count of rollout days with t2m > heatwaveThresholdC */
  daysGrid: Grid2D
  /** Count of grid cells experiencing >= 1 day above threshold */
  cellsAtRisk: number
  /** Alias kept for clarity */
  affectedCells: number
  /** Share of the 32x36 grid cells at risk (0..100) */
  cellsAtRiskPct: number
  /** Spherical surface area (km²) of all cells with >= 1 day above threshold */
  areaAtRiskKm2: number
  /** Share of total grid spherical area at risk (0..100) */
  areaAtRiskPct: number
  /** Sum of heatwave days across all 32x36 grid cells */
  totalCellDays: number
  /** Peak heatwave days observed in any single cell */
  maxDaysInCell: number
  /** Mean heatwave days across at-risk cells (or 0) */
  meanDaysAffected: number
  /** Highest 2m temperature (°C) observed anywhere in this rollout */
  rolloutMaxTempC: number
  /** Lowest 2m temperature (°C) observed anywhere in this rollout */
  rolloutMinTempC: number
}

export interface ImpactComparison {
  prime: SingleImpactSummary
  branch: SingleImpactSummary | null
  diffGrid: Grid2D | null
  deltaCellsAtRisk: number | null
  deltaAffectedCells: number | null
  deltaAreaAtRiskKm2: number | null
  deltaCellDays: number | null
}

export function computeHeatwaveImpact(
  rollout: StateSequence,
  settings: ImpactSettings,
): SingleImpactSummary {
  const thresholdK = settings.heatwaveThresholdC + 273.15
  const daysGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  let maxTempK = -Infinity
  let minTempK = Infinity

  for (let k = 0; k < rollout.length; k++) {
    const t2m = rollout[k][0]
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const tempK = t2m[r][c]
        if (tempK > maxTempK) maxTempK = tempK
        if (tempK < minTempK) minTempK = tempK
        if (tempK > thresholdK) {
          daysGrid[r][c] += 1
        }
      }
    }
  }

  let totalCellDays = 0
  let cellsAtRisk = 0
  let areaAtRiskKm2 = 0
  let maxDaysInCell = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    const cellArea = ROW_CELL_AREAS_KM2[r]
    for (let c = 0; c < GRID_COLS; c++) {
      const d = daysGrid[r][c]
      totalCellDays += d
      if (d > 0) {
        cellsAtRisk += 1
        areaAtRiskKm2 += cellArea
      }
      if (d > maxDaysInCell) maxDaysInCell = d
    }
  }

  return {
    daysGrid,
    cellsAtRisk,
    affectedCells: cellsAtRisk,
    cellsAtRiskPct: TOTAL_GRID_CELLS > 0 ? (cellsAtRisk / TOTAL_GRID_CELLS) * 100 : 0,
    areaAtRiskKm2,
    areaAtRiskPct: TOTAL_GRID_AREA_KM2 > 0 ? (areaAtRiskKm2 / TOTAL_GRID_AREA_KM2) * 100 : 0,
    totalCellDays,
    maxDaysInCell,
    meanDaysAffected: cellsAtRisk > 0 ? totalCellDays / cellsAtRisk : 0,
    rolloutMaxTempC: Number.isFinite(maxTempK) ? maxTempK - 273.15 : NaN,
    rolloutMinTempC: Number.isFinite(minTempK) ? minTempK - 273.15 : NaN,
  }
}

export function compareHeatwaveImpact(
  primeRollout: StateSequence,
  branchRollout: StateSequence | null,
  settings: ImpactSettings,
): ImpactComparison {
  const prime = computeHeatwaveImpact(primeRollout, settings)
  if (!branchRollout) {
    return {
      prime,
      branch: null,
      diffGrid: null,
      deltaCellsAtRisk: null,
      deltaAffectedCells: null,
      deltaAreaAtRiskKm2: null,
      deltaCellDays: null,
    }
  }

  const branch = computeHeatwaveImpact(branchRollout, settings)
  const diffGrid: Grid2D = Array.from({ length: GRID_ROWS }, (_, r) =>
    Array.from({ length: GRID_COLS }, (_, c) => branch.daysGrid[r][c] - prime.daysGrid[r][c]),
  )

  const deltaCells = branch.cellsAtRisk - prime.cellsAtRisk

  return {
    prime,
    branch,
    diffGrid,
    deltaCellsAtRisk: deltaCells,
    deltaAffectedCells: deltaCells,
    deltaAreaAtRiskKm2: branch.areaAtRiskKm2 - prime.areaAtRiskKm2,
    deltaCellDays: branch.totalCellDays - prime.totalCellDays,
  }
}
