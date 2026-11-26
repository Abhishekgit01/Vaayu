import {
  GRID_COLS,
  GRID_ROWS,
  INIT_DAYS,
  NUM_CHANNELS,
  type ScenarioData,
  type State3D,
  type StateSequence,
} from '../api/types'

/**
 * Returns true only when the current URL query string includes ?dev=1.
 */
export function isDevPreviewEnabled(): boolean {
  if (typeof window === 'undefined') return false
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

function buildProceduralDayState(dayOffset: number): State3D {
  const state: State3D = Array.from({ length: NUM_CHANNELS }, () =>
    Array.from({ length: GRID_ROWS }, () => new Array<number>(GRID_COLS).fill(0)),
  )

  const phase = dayOffset * 0.28

  for (let r = 0; r < GRID_ROWS; r++) {
    const ny = r / (GRID_ROWS - 1) // 0 (south) .. 1 (north)
    for (let c = 0; c < GRID_COLS; c++) {
      const nx = c / (GRID_COLS - 1) // 0 (west) .. 1 (east)

      // 0: t2m (K) — smooth synthetic thermal dome + gentle meridional wave
      const warmCore = Math.exp(-((ny - 0.58) ** 2 / 0.08 + (nx - 0.35) ** 2 / 0.12))
      const northCool = Math.max(0, ny - 0.78) * 38
      const t2m =
        298.5 +
        12.5 * warmCore -
        northCool +
        2.2 * Math.sin(nx * Math.PI * 2 + phase) * Math.cos(ny * Math.PI)

      // 1: tp (mm/day) — smooth synthetic precipitation lobes crossing 10 mm/day
      const rainLobeA = Math.exp(-((ny - 0.32) ** 2 / 0.045 + (nx - 0.68) ** 2 / 0.06))
      const rainLobeB = Math.exp(-((ny - 0.38) ** 2 / 0.05 + (nx - 0.24) ** 2 / 0.03))
      const rainWave = 0.5 + 0.5 * Math.sin(nx * 5.2 - phase + ny * 3.1)
      const tp = Math.max(0, 22 * rainLobeA + 15 * rainLobeB * (0.7 + 0.3 * rainWave))

      // 2: u10 (m/s) & 3: v10 (m/s) — smooth synthetic circulation
      const u10 =
        8.5 * Math.sin((ny + 0.15) * Math.PI) * (0.8 + 0.25 * Math.cos(nx * Math.PI * 2 - phase)) -
        2.0 * Math.cos(ny * Math.PI * 2 + phase * 0.5)
      const v10 =
        5.5 * Math.cos(nx * Math.PI * 1.8 - phase * 0.7) * Math.sin(ny * Math.PI) +
        2.2 * Math.sin((nx - ny) * Math.PI * 2)

      // 4: msl (Pa) — smooth synthetic pressure trough
      const msl =
        100950 -
        520 * warmCore +
        260 * Math.cos(nx * Math.PI * 1.5 + phase * 0.5) * Math.sin(ny * Math.PI)

      // 5: tcwv (kg/m2) — smooth synthetic column moisture
      const tcwv = Math.max(
        12,
        44 +
          18 * Math.exp(-((ny - 0.28) ** 2 / 0.14)) * (0.75 + 0.25 * Math.sin(nx * Math.PI + phase)) -
          24 * Math.max(0, ny - 0.65),
      )

      state[0][r][c] = t2m
      state[1][r][c] = tp
      state[2][r][c] = u10
      state[3][r][c] = v10
      state[4][r][c] = msl
      state[5][r][c] = tcwv
    }
  }

  return state
}

/**
 * Generates a clearly labelled SYNTHETIC scenario [3][6][32][36] at runtime
 * from smooth procedural functions (not real weather).
 */
export function generateSyntheticScenario(): ScenarioData {
  const init_states: StateSequence = []
  for (let d = 0; d < INIT_DAYS; d++) {
    init_states.push(buildProceduralDayState(d))
  }
  const syntheticAngle = (2 * Math.PI * 195) / 365
  return {
    init_states,
    doy_sincos: [Math.sin(syntheticAngle), Math.cos(syntheticAngle)],
    sourceLabel: 'SYNTHETIC - DEV ONLY',
  }
}

function diffuseStep(grid: number[][], driftCol: number, driftRow: number, decay: number): number[][] {
  const next: number[][] = Array.from({ length: GRID_ROWS }, () =>
    new Array<number>(GRID_COLS).fill(0),
  )

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const sr = Math.max(0, Math.min(GRID_ROWS - 1, Math.round(r - driftRow)))
      const sc = Math.max(0, Math.min(GRID_COLS - 1, Math.round(c - driftCol)))
      const center = grid[sr][sc]
      const n = grid[Math.min(GRID_ROWS - 1, sr + 1)][sc]
      const s = grid[Math.max(0, sr - 1)][sc]
      const e = grid[sr][Math.min(GRID_COLS - 1, sc + 1)]
      const w = grid[sr][Math.max(0, sc - 1)]
      const ne = grid[Math.min(GRID_ROWS - 1, sr + 1)][Math.min(GRID_COLS - 1, sc + 1)]
      const nw = grid[Math.min(GRID_ROWS - 1, sr + 1)][Math.max(0, sc - 1)]
      const se = grid[Math.max(0, sr - 1)][Math.min(GRID_COLS - 1, sc + 1)]
      const sw = grid[Math.max(0, sr - 1)][Math.max(0, sc - 1)]
      next[r][c] =
        decay * (0.4 * center + 0.1 * (n + s + e + w) + 0.05 * (ne + nw + se + sw))
    }
  }
  return next
}

function hasAnyNonZeroPerturbation(perturbation: State3D | null): boolean {
  if (!perturbation) return false
  for (let ch = 0; ch < NUM_CHANNELS; ch++) {
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        if (perturbation[ch][r][c] !== 0) return true
      }
    }
  }
  return false
}

/**
 * Generates a procedural fake rollout [steps][6][32][36] for ?dev=1 UI preview without a server.
 * An all-zero perturbation produces an identical rollout to perturbation === null.
 */
export function generateSyntheticRollout(
  initStates: StateSequence,
  steps: number,
  perturbation: State3D | null,
): StateSequence {
  const baseLast = initStates[INIT_DAYS - 1]
  const activePert = hasAnyNonZeroPerturbation(perturbation)

  // Working perturbation field propagated day by day (identical zero when inactive)
  let pertField: State3D | null = activePert
    ? perturbation!.map((grid) => grid.map((row) => row.slice()))
    : null

  const rollout: StateSequence = []

  for (let k = 0; k < steps; k++) {
    const dayNum = k + 1
    const baseNext = buildProceduralDayState(INIT_DAYS - 1 + dayNum)

    // Blend smoothly from baseLast on early steps for continuity
    const blend = Math.min(1, dayNum / 3)
    const state: State3D = Array.from({ length: NUM_CHANNELS }, (_, ch) =>
      Array.from({ length: GRID_ROWS }, (_, r) =>
        Array.from({ length: GRID_COLS }, (_, c) => {
          return baseLast[ch][r][c] * (1 - blend) + baseNext[ch][r][c] * blend
        }),
      ),
    )

    if (pertField) {
      // Diffuse & gently advect perturbation outward each step
      pertField = pertField.map((grid, ch) => {
        const driftC = ch === 2 || ch === 3 ? 0.6 : 0.45
        const driftR = 0.35
        // Two light passes per day so the front expands smoothly across cells
        const p1 = diffuseStep(grid, driftC, driftR, 0.96)
        return diffuseStep(p1, driftC * 0.5, driftR * 0.5, 0.96)
      })

      for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
          const dt = pertField[0][r][c]
          const dtp = pertField[1][r][c]
          const du = pertField[2][r][c]
          const dv = pertField[3][r][c]
          const dmsl = pertField[4][r][c]
          const dtcwv = pertField[5][r][c]

          // Direct delta plus modest cross-channel procedural coupling
          state[0][r][c] += dt - 0.04 * du
          state[1][r][c] = Math.max(0, state[1][r][c] + dtp + 0.55 * dt + 0.35 * dtcwv + 0.4 * Math.hypot(du, dv) * Math.sign(du + dv || 1))
          state[2][r][c] += du + 0.18 * dt
          state[3][r][c] += dv + 0.14 * dt
          state[4][r][c] += dmsl - 28 * dt
          state[5][r][c] = Math.max(0, state[5][r][c] + dtcwv + 0.8 * dt)
        }
      }
    }

    rollout.push(state)
  }

  return rollout
}
