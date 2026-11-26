import { GRID_COLS, GRID_ROWS, type Grid2D, type State3D, type StateSequence } from '../api/types'

/** Standard agronomic approximation: 1 grid cell (1° × 1°) ≈ 10,000 sq km */
export const CELL_AREA_SQ_KM = 10_000
export const TOTAL_GRID_CELLS = GRID_ROWS * GRID_COLS
export const TOTAL_DOMAIN_SQ_KM = TOTAL_GRID_CELLS * CELL_AREA_SQ_KM

/** 2m Temperature (t2m, K) wire channel index */
export const T2M_CHANNEL_INDEX = 0
/** Rainfall (tp, mm/day) wire channel index */
export const TP_CHANNEL_INDEX = 1
/** Total Column Water Vapor (tcwv, kg/m²) wire channel index */
export const TCWV_CHANNEL_INDEX = 5

/** Lethal Wet-Bulb Equivalent threshold in Kelvin (approx 37°C) for Severe Human Danger */
export const WET_BULB_DANGER_THRESHOLD_K = 310.0
export const WET_BULB_TCWV_WEIGHT = 0.15

/**
 * Agronomic & Human Risk Decision Support System settings.
 */
export interface ImpactSettings {
  /** Daily rainfall threshold in mm/day; days with tp < threshold count toward a dry spell (official standard: 2.5 mm/day) */
  drySpellThresholdMmDay: number
  /** Minimum consecutive dry days required to flag a cell as Severe Crop Stress (official standard: 5 days) */
  severeStressConsecutiveDays: number
  /** Wet-Bulb Equivalent threshold in K (Wet_Bulb_Eq > threshold flags Severe Human Danger; standard: 310.0 K) */
  wetBulbDangerThresholdK?: number
}

export const DEFAULT_IMPACT_SETTINGS: ImpactSettings = {
  drySpellThresholdMmDay: 2.5,
  severeStressConsecutiveDays: 5,
  wetBulbDangerThresholdK: WET_BULB_DANGER_THRESHOLD_K,
}

export interface DrySpellImpactSummary {
  /** Number of rollout steps evaluated */
  evaluatedSteps: number
  /** [32][36] maximum consecutive steps per cell where tp < drySpellThresholdMmDay */
  maxDrySpellGrid: Grid2D
  /** [32][36] binary mask (1 = Severe Crop Stress where maxDrySpell >= severeStressConsecutiveDays, 0 = otherwise) */
  severeStressMask: Grid2D
  /** [32][36] cumulative rainfall (mm) over the rollout per cell */
  totalRainfallGrid: Grid2D
  /** Count of cells flagged with Severe Crop Stress */
  severeStressCells: number
  /** Total area (sq km) under Severe Crop Stress (1 cell = 10,000 sq km) */
  severeStressAreaSqKm: number
  /** Percentage of the 32x36 domain under Severe Crop Stress (0..100) */
  severeStressPct: number
  /** Longest consecutive dry spell (days) observed in any cell */
  maxConsecutiveDryDays: number
  /** Mean max dry-spell length across stressed cells (or 0 if none) */
  meanStressedSpellDays: number
}

export interface DrySpellComparison {
  prime: DrySpellImpactSummary
  branch: DrySpellImpactSummary | null
  /**
   * [32][36] stress transition grid when Branch exists:
   * +1 = newly stressed in Branch, -1 = stress relieved in Branch, 0 = unchanged
   */
  stressShiftGrid: Grid2D | null
  /** [32][36] difference in max consecutive dry days (Branch - Prime) */
  spellDiffGrid: Grid2D | null
  deltaSevereCells: number | null
  deltaSevereAreaSqKm: number | null
  newlyStressedCells: number | null
  relievedCells: number | null
}

/**
 * Compute the Kharif Crop Dry-Spell Index entirely in the browser from a rollout [steps][6][32][36].
 * Ensures 0 errors and 0 Severe Crop Stress when steps < severeStressConsecutiveDays (e.g. steps < 5).
 */
export function computeKharifDrySpellImpact(
  rollout: StateSequence | null | undefined,
  settings: ImpactSettings = DEFAULT_IMPACT_SETTINGS,
): DrySpellImpactSummary {
  const steps = Array.isArray(rollout) ? rollout.length : 0
  const dryThreshold = settings.drySpellThresholdMmDay
  const stressDaysRequired = Math.max(1, Math.round(settings.severeStressConsecutiveDays))

  const maxDrySpellGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )
  const severeStressMask: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )
  const totalRainfallGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  if (!rollout || steps === 0) {
    return {
      evaluatedSteps: 0,
      maxDrySpellGrid,
      severeStressMask,
      totalRainfallGrid,
      severeStressCells: 0,
      severeStressAreaSqKm: 0,
      severeStressPct: 0,
      maxConsecutiveDryDays: 0,
      meanStressedSpellDays: 0,
    }
  }

  let severeStressCells = 0
  let maxConsecutiveDryDays = 0
  let stressedSpellSum = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      let currentConsecutive = 0
      let cellMaxConsecutive = 0
      let rainSum = 0

      for (let s = 0; s < steps; s++) {
        const tp = rollout[s]?.[TP_CHANNEL_INDEX]?.[r]?.[c]
        const validTp = typeof tp === 'number' && Number.isFinite(tp) ? tp : 0
        rainSum += validTp

        if (validTp < dryThreshold) {
          currentConsecutive += 1
          if (currentConsecutive > cellMaxConsecutive) {
            cellMaxConsecutive = currentConsecutive
          }
        } else {
          currentConsecutive = 0
        }
      }

      maxDrySpellGrid[r][c] = cellMaxConsecutive
      totalRainfallGrid[r][c] = rainSum

      if (cellMaxConsecutive > maxConsecutiveDryDays) {
        maxConsecutiveDryDays = cellMaxConsecutive
      }

      // When steps < stressDaysRequired (e.g. steps < 5), cellMaxConsecutive <= steps < 5,
      // so severeStressMask cleanly remains 0 across all cells with zero errors.
      if (steps >= stressDaysRequired && cellMaxConsecutive >= stressDaysRequired) {
        severeStressMask[r][c] = 1
        severeStressCells += 1
        stressedSpellSum += cellMaxConsecutive
      }
    }
  }

  const severeStressAreaSqKm = severeStressCells * CELL_AREA_SQ_KM

  return {
    evaluatedSteps: steps,
    maxDrySpellGrid,
    severeStressMask,
    totalRainfallGrid,
    severeStressCells,
    severeStressAreaSqKm,
    severeStressPct: TOTAL_GRID_CELLS > 0 ? (severeStressCells / TOTAL_GRID_CELLS) * 100 : 0,
    maxConsecutiveDryDays,
    meanStressedSpellDays: severeStressCells > 0 ? stressedSpellSum / severeStressCells : 0,
  }
}

export function compareKharifDrySpellImpact(
  primeRollout: StateSequence | null | undefined,
  branchRollout: StateSequence | null | undefined,
  settings: ImpactSettings = DEFAULT_IMPACT_SETTINGS,
): DrySpellComparison | null {
  if (!primeRollout) return null

  const prime = computeKharifDrySpellImpact(primeRollout, settings)
  if (!branchRollout) {
    return {
      prime,
      branch: null,
      stressShiftGrid: null,
      spellDiffGrid: null,
      deltaSevereCells: null,
      deltaSevereAreaSqKm: null,
      newlyStressedCells: null,
      relievedCells: null,
    }
  }

  const branch = computeKharifDrySpellImpact(branchRollout, settings)
  const stressShiftGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )
  const spellDiffGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  let newlyStressedCells = 0
  let relievedCells = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const pStress = prime.severeStressMask[r][c]
      const bStress = branch.severeStressMask[r][c]
      const shift = bStress - pStress
      stressShiftGrid[r][c] = shift
      spellDiffGrid[r][c] = branch.maxDrySpellGrid[r][c] - prime.maxDrySpellGrid[r][c]
      if (shift > 0) newlyStressedCells += 1
      if (shift < 0) relievedCells += 1
    }
  }

  const deltaSevereCells = branch.severeStressCells - prime.severeStressCells

  return {
    prime,
    branch,
    stressShiftGrid,
    spellDiffGrid,
    deltaSevereCells,
    deltaSevereAreaSqKm: deltaSevereCells * CELL_AREA_SQ_KM,
    newlyStressedCells,
    relievedCells,
  }
}

export interface WetBulbStepSummary {
  /** 0-based step index evaluated */
  stepIndex: number
  /** [32][36] Wet_Bulb_Eq = t2m + (tcwv * 0.15) in Kelvin */
  wetBulbGrid: Grid2D
  /** [32][36] binary mask (1 = Severe Human Danger where Wet_Bulb_Eq > thresholdK, 0 = otherwise) */
  severeDangerMask: Grid2D
  /** Count of cells flagged as Severe Human Danger */
  severeDangerCells: number
  /** Total area (sq km) in Severe Human Danger (1 cell = 10,000 sq km) */
  severeDangerAreaSqKm: number
  /** Percentage of the 32x36 domain in Severe Human Danger (0..100) */
  severeDangerPct: number
  /** Peak Wet_Bulb_Eq (K) observed across all cells at this step */
  maxWetBulbK: number
  /** Domain-mean Wet_Bulb_Eq (K) across all cells at this step */
  meanWetBulbK: number
  /** Domain-minimum Wet_Bulb_Eq (K) across all cells at this step */
  minWetBulbK: number
}

export interface WetBulbStepComparison {
  stepIndex: number
  totalSteps: number
  thresholdK: number
  prime: WetBulbStepSummary
  branch: WetBulbStepSummary | null
  /**
   * [32][36] danger transition grid when Branch exists at stepIndex:
   * +1 = newly flagged Severe Human Danger in Branch, -1 = danger relieved in Branch, 0 = unchanged
   */
  dangerShiftGrid: Grid2D | null
  /** [32][36] Wet_Bulb_Eq difference (Branch - Prime) in K */
  wetBulbDiffGrid: Grid2D | null
  deltaDangerCells: number | null
  deltaDangerAreaSqKm: number | null
  newlyEndangeredCells: number | null
  relievedCells: number | null
}

/**
 * Compute the Lethal Wet-Bulb Index statically for a single timestep state [6][32][36]:
 *   Wet_Bulb_Eq = t2m + (tcwv * 0.15)
 * Cells with Wet_Bulb_Eq > thresholdK (default 310.0 K, approx 37°C) are flagged as Severe Human Danger.
 */
export function computeWetBulbStepImpact(
  state: State3D | null | undefined,
  stepIndex = 0,
  thresholdK: number = WET_BULB_DANGER_THRESHOLD_K,
): WetBulbStepSummary {
  const wetBulbGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )
  const severeDangerMask: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  if (!state) {
    return {
      stepIndex,
      wetBulbGrid,
      severeDangerMask,
      severeDangerCells: 0,
      severeDangerAreaSqKm: 0,
      severeDangerPct: 0,
      maxWetBulbK: 0,
      meanWetBulbK: 0,
      minWetBulbK: 0,
    }
  }

  const t2mGrid = state[T2M_CHANNEL_INDEX]
  const tcwvGrid = state[TCWV_CHANNEL_INDEX]

  let severeDangerCells = 0
  let maxWetBulbK = -Infinity
  let minWetBulbK = Infinity
  let sumWetBulbK = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const rawT2m = t2mGrid?.[r]?.[c]
      const rawTcwv = tcwvGrid?.[r]?.[c]
      const t2m = typeof rawT2m === 'number' && Number.isFinite(rawT2m) ? rawT2m : 0
      const tcwv = typeof rawTcwv === 'number' && Number.isFinite(rawTcwv) ? rawTcwv : 0

      const wetBulbEq = t2m + tcwv * WET_BULB_TCWV_WEIGHT
      wetBulbGrid[r][c] = wetBulbEq
      sumWetBulbK += wetBulbEq

      if (wetBulbEq > maxWetBulbK) maxWetBulbK = wetBulbEq
      if (wetBulbEq < minWetBulbK) minWetBulbK = wetBulbEq

      if (wetBulbEq > thresholdK) {
        severeDangerMask[r][c] = 1
        severeDangerCells += 1
      }
    }
  }

  return {
    stepIndex,
    wetBulbGrid,
    severeDangerMask,
    severeDangerCells,
    severeDangerAreaSqKm: severeDangerCells * CELL_AREA_SQ_KM,
    severeDangerPct: TOTAL_GRID_CELLS > 0 ? (severeDangerCells / TOTAL_GRID_CELLS) * 100 : 0,
    maxWetBulbK: Number.isFinite(maxWetBulbK) ? maxWetBulbK : 0,
    meanWetBulbK: TOTAL_GRID_CELLS > 0 ? sumWetBulbK / TOTAL_GRID_CELLS : 0,
    minWetBulbK: Number.isFinite(minWetBulbK) ? minWetBulbK : 0,
  }
}

/**
 * Compare Prime vs Branch Lethal Wet-Bulb Index on-the-fly for the active timeline step.
 */
export function compareWetBulbStepImpact(
  primeRollout: StateSequence | null | undefined,
  branchRollout: StateSequence | null | undefined,
  stepIndex: number,
  thresholdK: number = WET_BULB_DANGER_THRESHOLD_K,
): WetBulbStepComparison | null {
  if (!primeRollout || primeRollout.length === 0) return null

  const safePrimeIdx = Math.max(0, Math.min(primeRollout.length - 1, stepIndex))
  const prime = computeWetBulbStepImpact(primeRollout[safePrimeIdx], safePrimeIdx, thresholdK)

  if (!branchRollout || branchRollout.length === 0) {
    return {
      stepIndex: safePrimeIdx,
      totalSteps: primeRollout.length,
      thresholdK,
      prime,
      branch: null,
      dangerShiftGrid: null,
      wetBulbDiffGrid: null,
      deltaDangerCells: null,
      deltaDangerAreaSqKm: null,
      newlyEndangeredCells: null,
      relievedCells: null,
    }
  }

  const safeBranchIdx = Math.max(0, Math.min(branchRollout.length - 1, stepIndex))
  const branch = computeWetBulbStepImpact(branchRollout[safeBranchIdx], safeBranchIdx, thresholdK)

  const dangerShiftGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )
  const wetBulbDiffGrid: Grid2D = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  let newlyEndangeredCells = 0
  let relievedCells = 0

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const pDanger = prime.severeDangerMask[r][c]
      const bDanger = branch.severeDangerMask[r][c]
      const shift = bDanger - pDanger
      dangerShiftGrid[r][c] = shift
      wetBulbDiffGrid[r][c] = branch.wetBulbGrid[r][c] - prime.wetBulbGrid[r][c]
      if (shift > 0) newlyEndangeredCells += 1
      if (shift < 0) relievedCells += 1
    }
  }

  const deltaDangerCells = branch.severeDangerCells - prime.severeDangerCells

  return {
    stepIndex: safePrimeIdx,
    totalSteps: Math.max(primeRollout.length, branchRollout.length),
    thresholdK,
    prime,
    branch,
    dangerShiftGrid,
    wetBulbDiffGrid,
    deltaDangerCells,
    deltaDangerAreaSqKm: deltaDangerCells * CELL_AREA_SQ_KM,
    newlyEndangeredCells,
    relievedCells,
  }
}
