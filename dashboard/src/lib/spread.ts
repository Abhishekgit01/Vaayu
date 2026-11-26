import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type StateSequence,
} from '../api/types'
import { computeDiffGrid } from './constants'

export const DAY1_LOCALITY_CELLS = 3
export const REFERENCE_FRONT_CELLS_PER_DAY = 8
export const FRONT_THRESHOLD_FRACTION = 0.05

export interface SpreadDayProfile {
  /** 1-indexed forecast day */
  day: number
  /** Mean |Branch - Prime| per integer cell-distance bin (index = distance in cells) */
  magnitudeByDistance: number[]
  /** Farthest distance bin exceeding the front threshold (in cells), or 0 */
  frontDistanceCells: number
  /** Peak magnitude on this day */
  peakMagnitude: number
}

export interface SpreadAnalysis {
  maxDistance: number
  globalPeak: number
  frontThresholdValue: number
  hasShockRegion: boolean
  profiles: SpreadDayProfile[]
}

/**
 * Compute Euclidean distance in cells from each grid cell to the nearest cell in shockMask.
 * Cells inside the shock have distance 0.
 */
export function computeDistanceField(shockMask: boolean[][]): {
  distances: number[][]
  maxDistance: number
  hasShockRegion: boolean
} {
  const shockCells: Array<[number, number]> = []
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (shockMask[r][c]) {
        shockCells.push([r, c])
      }
    }
  }

  const distances: number[][] = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  if (shockCells.length === 0) {
    return { distances, maxDistance: 0, hasShockRegion: false }
  }

  let maxDist = 0
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (shockMask[r][c]) {
        distances[r][c] = 0
        continue
      }
      let bestSq = Infinity
      for (let i = 0; i < shockCells.length; i++) {
        const dr = r - shockCells[i][0]
        const dc = c - shockCells[i][1]
        const d2 = dr * dr + dc * dc
        if (d2 < bestSq) bestSq = d2
      }
      const bin = Math.round(Math.sqrt(bestSq))
      distances[r][c] = bin
      if (bin > maxDist) maxDist = bin
    }
  }

  return { distances, maxDistance: maxDist, hasShockRegion: true }
}

/**
 * Compute response magnitude vs distance in cells across all rollout steps,
 * plus day-by-day propagation front distance.
 */
export function computeSpreadAnalysis(
  primeRollout: StateSequence,
  branchRollout: StateSequence,
  shockMask: boolean[][],
  channel: DisplayChannelId,
): SpreadAnalysis {
  const steps = Math.min(primeRollout.length, branchRollout.length)
  const { distances, maxDistance, hasShockRegion } = computeDistanceField(shockMask)

  if (!hasShockRegion || steps === 0) {
    return {
      maxDistance: 0,
      globalPeak: 0,
      frontThresholdValue: 0,
      hasShockRegion,
      profiles: [],
    }
  }

  const numBins = maxDistance + 1
  const binCounts = new Array<number>(numBins).fill(0)
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      binCounts[distances[r][c]]++
    }
  }

  let globalPeak = 0
  const rawProfiles: Array<{ day: number; mags: number[]; peak: number }> = []

  for (let k = 0; k < steps; k++) {
    const diff = computeDiffGrid(primeRollout[k], branchRollout[k], channel)
    const sums = new Array<number>(numBins).fill(0)
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        sums[distances[r][c]] += Math.abs(diff[r][c])
      }
    }
    const mags = new Array<number>(numBins).fill(0)
    let dayPeak = 0
    for (let b = 0; b < numBins; b++) {
      const avg = binCounts[b] > 0 ? sums[b] / binCounts[b] : 0
      mags[b] = avg
      if (avg > dayPeak) dayPeak = avg
    }
    if (dayPeak > globalPeak) globalPeak = dayPeak
    rawProfiles.push({ day: k + 1, mags, peak: dayPeak })
  }

  const frontThresholdValue = globalPeak > 0 ? globalPeak * FRONT_THRESHOLD_FRACTION : 0

  const profiles: SpreadDayProfile[] = rawProfiles.map(({ day, mags, peak }) => {
    let frontDist = 0
    if (frontThresholdValue > 0) {
      for (let b = numBins - 1; b >= 0; b--) {
        if (mags[b] >= frontThresholdValue) {
          frontDist = b
          break
        }
      }
    }
    return {
      day,
      magnitudeByDistance: mags,
      frontDistanceCells: frontDist,
      peakMagnitude: peak,
    }
  })

  return {
    maxDistance,
    globalPeak,
    frontThresholdValue,
    hasShockRegion: true,
    profiles,
  }
}
