import { GRID_COLS, GRID_ROWS, type Grid2D, type State3D } from '../api/types'
import type { WindArrow } from './heatmap'

export const MAX_WIND_PARTICLES_2D = 380
export const MAX_WIND_STREAMLINES_3D = 168

export interface WindParticle {
  gx: number
  gy: number
  age: number
  maxAge: number
}

/**
 * Bilinearly interpolate (u10, v10) at continuous north-up map coordinates
 * gx in [0, GRID_COLS], gy in [0, GRID_ROWS] (gy = 0 is North, gy = GRID_ROWS is South).
 */
export function sampleWindBilinear(
  uGrid: Grid2D,
  vGrid: Grid2D,
  gx: number,
  gy: number,
): [number, number] {
  const gc = Math.max(0, Math.min(GRID_COLS - 1, gx - 0.5))
  const gr = Math.max(0, Math.min(GRID_ROWS - 1, GRID_ROWS - 1 - (gy - 0.5)))

  const c0 = Math.floor(gc)
  const c1 = Math.min(GRID_COLS - 1, c0 + 1)
  const fc = gc - c0

  const r0 = Math.floor(gr)
  const r1 = Math.min(GRID_ROWS - 1, r0 + 1)
  const fr = gr - r0

  const u0 = uGrid[r0][c0] + (uGrid[r0][c1] - uGrid[r0][c0]) * fc
  const u1 = uGrid[r1][c0] + (uGrid[r1][c1] - uGrid[r1][c0]) * fc
  const u = u0 + (u1 - u0) * fr

  const v0 = vGrid[r0][c0] + (vGrid[r0][c1] - vGrid[r0][c0]) * fc
  const v1 = vGrid[r1][c0] + (vGrid[r1][c1] - vGrid[r1][c0]) * fc
  const v = v0 + (v1 - v0) * fr

  return [u, v]
}

export function computeMaxWindSpeed(state: State3D | null | undefined): number {
  if (!state) return 1
  const uGrid = state[2]
  const vGrid = state[3]
  let maxS = 0
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const s = Math.hypot(uGrid[r][c], vGrid[r][c])
      if (s > maxS) maxS = s
    }
  }
  return Math.max(1e-3, maxS)
}

// Perceptual speed stops: rain #7FB5B0 -> pale sage #D6E8DC -> bone #ECE7DA -> lightning #F0B84B
const SPEED_STOPS: ReadonlyArray<[number, number, number]> = [
  [127, 181, 176],
  [214, 232, 220],
  [236, 231, 218],
  [240, 184, 75],
]

export function sampleWindSpeedRgb(normSpeed: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, Number.isFinite(normSpeed) ? normSpeed : 0))
  const scaled = t * (SPEED_STOPS.length - 1)
  const idx = Math.min(SPEED_STOPS.length - 2, Math.floor(scaled))
  const frac = scaled - idx
  const a = SPEED_STOPS[idx]
  const b = SPEED_STOPS[idx + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ]
}

export function createInitialParticles(count = MAX_WIND_PARTICLES_2D): WindParticle[] {
  const capped = Math.min(MAX_WIND_PARTICLES_2D, Math.max(1, count))
  const particles: WindParticle[] = new Array(capped)
  for (let i = 0; i < capped; i++) {
    const maxAge = 55 + Math.floor(Math.random() * 45)
    particles[i] = {
      gx: Math.random() * GRID_COLS,
      gy: Math.random() * GRID_ROWS,
      age: Math.floor(Math.random() * maxAge),
      maxAge,
    }
  }
  return particles
}

export function respawnParticle(p: WindParticle): void {
  p.gx = Math.random() * GRID_COLS
  p.gy = Math.random() * GRID_ROWS
  p.age = 0
  p.maxAge = 55 + Math.floor(Math.random() * 45)
}

/**
 * Render static speed-colored streamlines when prefers-reduced-motion is enabled.
 */
export function drawStaticWindStreamlines2D(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  getFieldAtX: (normX: number) => State3D | null | undefined,
  maxSpeed: number,
): void {
  ctx.clearRect(0, 0, width, height)
  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineWidth = 1.15 * dpr

  const stepDt = 0.08
  const numSteps = 10

  for (let sy = 1.2; sy < GRID_ROWS; sy += 2.2) {
    for (let sx = 1.2; sx < GRID_COLS; sx += 2.2) {
      let gx = sx
      let gy = sy
      for (let s = 0; s < numSteps; s++) {
        const state = getFieldAtX(gx / GRID_COLS)
        if (!state) break
        const [u, v] = sampleWindBilinear(state[2], state[3], gx, gy)
        const speed = Math.hypot(u, v)
        if (speed < maxSpeed * 0.03) break
        const nx = gx + u * stepDt
        const ny = gy - v * stepDt // +v is North = negative gy
        if (nx < 0 || nx > GRID_COLS || ny < 0 || ny > GRID_ROWS) break

        const normS = Math.min(1, speed / maxSpeed)
        const [r, g, b] = sampleWindSpeedRgb(normS)
        const alpha = 0.28 + 0.58 * normS
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`
        ctx.beginPath()
        ctx.moveTo((gx / GRID_COLS) * width, (gy / GRID_ROWS) * height)
        ctx.lineTo((nx / GRID_COLS) * width, (ny / GRID_ROWS) * height)
        ctx.stroke()

        gx = nx
        gy = ny
      }
    }
  }
  ctx.restore()
}

/**
 * Build vector arrows of (Branch − Prime) wind (du10, dv10) for Difference mode.
 */
export function buildDiffWindArrows(
  primeState: State3D,
  branchState: State3D,
): WindArrow[] {
  const pu = primeState[2]
  const pv = primeState[3]
  const bu = branchState[2]
  const bv = branchState[3]

  let maxDiffSpeed = 0
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const s = Math.hypot(bu[r][c] - pu[r][c], bv[r][c] - pv[r][c])
      if (s > maxDiffSpeed) maxDiffSpeed = s
    }
  }
  if (maxDiffSpeed <= 1e-6) return []

  const arrows: WindArrow[] = []
  for (let r = 1; r < GRID_ROWS; r += 2) {
    const cy = GRID_ROWS - 1 - r + 0.5
    for (let c = 1; c < GRID_COLS; c += 2) {
      const du = bu[r][c] - pu[r][c]
      const dv = bv[r][c] - pv[r][c]
      const speed = Math.hypot(du, dv)
      if (speed < maxDiffSpeed * 0.05) continue

      const len = 0.45 + 1.15 * Math.min(1, speed / maxDiffSpeed)
      const ux = du / speed
      const uy = -dv / speed
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

export interface Streamline3DPath {
  /** Sequence of [gx, gy, normSpeed] points along the advected streamline */
  points: Array<[number, number, number]>
}

/**
 * Pre-trace capped bilinearly advected streamlines in [0..36] x [0..32] for 3D rendering.
 */
export function traceWindStreamlinesFor3D(state: State3D): Streamline3DPath[] {
  const uGrid = state[2]
  const vGrid = state[3]
  const maxSpeed = computeMaxWindSpeed(state)
  if (maxSpeed <= 1e-5) return []

  const paths: Streamline3DPath[] = []
  const stepDt = 0.085
  const numSegs = 14

  for (let sy = 1.2; sy < GRID_ROWS && paths.length < MAX_WIND_STREAMLINES_3D; sy += 2.5) {
    for (let sx = 1.4; sx < GRID_COLS && paths.length < MAX_WIND_STREAMLINES_3D; sx += 2.5) {
      let gx = sx
      let gy = sy
      const pts: Array<[number, number, number]> = []

      for (let s = 0; s <= numSegs; s++) {
        const [u, v] = sampleWindBilinear(uGrid, vGrid, gx, gy)
        const speed = Math.hypot(u, v)
        if (speed < maxSpeed * 0.03) break
        pts.push([gx, gy, Math.min(1, speed / maxSpeed)])
        gx += u * stepDt
        gy -= v * stepDt
        if (gx < 0.5 || gx > GRID_COLS - 0.5 || gy < 0.5 || gy > GRID_ROWS - 0.5) {
          break
        }
      }

      if (pts.length >= 3) {
        paths.push({ points: pts })
      }
    }
  }

  return paths
}
