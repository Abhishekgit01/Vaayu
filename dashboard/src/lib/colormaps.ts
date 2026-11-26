import type { DisplayChannelId, Grid2D } from '../api/types'

export type PaletteKind =
  | 'temperature'
  | 'rain'
  | 'wind'
  | 'pressure'
  | 'vapor'
  | 'diverging'
  | 'heatwave'

type RGB = [number, number, number]

const PALETTE_STOPS: Record<PaletteKind, readonly RGB[]> = {
  // Temperature: #2B3A67 → #3F7C85 → #A9C8A0 → #F2E3B3 → #F0B84B → #D9593D → #7A2B22
  temperature: [
    [43, 58, 103],
    [63, 124, 133],
    [169, 200, 160],
    [242, 227, 179],
    [240, 184, 75],
    [217, 89, 61],
    [122, 43, 34],
  ],
  // Rain: #0D1210 → #17383A → #2F6F73 → #7FB5B0 → #D6E8DC → #FFFFFF
  rain: [
    [13, 18, 16],
    [23, 56, 58],
    [47, 111, 115],
    [127, 181, 176],
    [214, 232, 220],
    [255, 255, 255],
  ],
  // Wind: perceptual sage-lichen ramp (#0D1210 → #ECE7DA)
  wind: [
    [13, 18, 16],
    [27, 43, 36],
    [53, 86, 71],
    [98, 140, 117],
    [169, 200, 160],
    [236, 231, 218],
  ],
  // Pressure (msl): perceptual warm-slate to bone ramp
  pressure: [
    [16, 21, 19],
    [38, 49, 43],
    [74, 90, 81],
    [126, 142, 132],
    [186, 194, 184],
    [236, 231, 218],
  ],
  // Water vapor (tcwv): perceptual deep-petrichor to pale rain ramp
  vapor: [
    [13, 18, 16],
    [23, 47, 48],
    [44, 88, 91],
    [88, 140, 139],
    [156, 198, 192],
    [236, 231, 218],
  ],
  // Difference (diverging, center EXACTLY zero = bone #ECE7DA): #3F7C85 ← #ECE7DA → #D9593D
  diverging: [
    [63, 124, 133],
    [150, 178, 176],
    [236, 231, 218],
    [227, 160, 140],
    [217, 89, 61],
  ],
  // Heatwave risk days: surface #141B18 (0 days) -> ochre -> lightning #F0B84B -> ember #D9593D -> #7A2B22
  heatwave: [
    [20, 27, 24],
    [72, 55, 34],
    [148, 96, 42],
    [240, 184, 75],
    [217, 89, 61],
    [122, 43, 34],
  ],
}

export function paletteForChannel(channel: DisplayChannelId): PaletteKind {
  switch (channel) {
    case 't2m':
      return 'temperature'
    case 'tp':
      return 'rain'
    case 'wind':
      return 'wind'
    case 'u10':
    case 'v10':
      return 'diverging'
    case 'msl':
      return 'pressure'
    case 'tcwv':
      return 'vapor'
  }
}

export function samplePalette(palette: PaletteKind, t: number): RGB {
  const stops = PALETTE_STOPS[palette]
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0.5))
  const scaled = clamped * (stops.length - 1)
  const idx = Math.min(stops.length - 2, Math.floor(scaled))
  const frac = scaled - idx
  const a = stops[idx]
  const b = stops[idx + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ]
}

// Odd LUT size (1025) guarantees t = 0.5 (exact 0 on diverging scales) lands on exact index 512.
const LUT_SIZE = 1025
const LUT_MAX_INDEX = LUT_SIZE - 1

const PALETTE_LUTS: Record<PaletteKind, Uint8ClampedArray> = (() => {
  const out = {} as Record<PaletteKind, Uint8ClampedArray>
  for (const kind of Object.keys(PALETTE_STOPS) as PaletteKind[]) {
    const lut = new Uint8ClampedArray(LUT_SIZE * 3)
    for (let i = 0; i < LUT_SIZE; i++) {
      const [r, g, b] = samplePalette(kind, i / LUT_MAX_INDEX)
      const o = i * 3
      lut[o] = r
      lut[o + 1] = g
      lut[o + 2] = b
    }
    out[kind] = lut
  }
  return out
})()

/**
 * Write RGB bytes for normalized t in [0, 1] directly into an RGBA ImageData buffer.
 */
export function writePaletteColor(
  palette: PaletteKind,
  t: number,
  target: Uint8ClampedArray,
  rgbaOffset: number,
): void {
  const lut = PALETTE_LUTS[palette]
  const clamped = t <= 0 ? 0 : t >= 1 ? 1 : Number.isFinite(t) ? t : 0.5
  const lutOffset = Math.round(clamped * LUT_MAX_INDEX) * 3
  target[rgbaOffset] = lut[lutOffset]
  target[rgbaOffset + 1] = lut[lutOffset + 1]
  target[rgbaOffset + 2] = lut[lutOffset + 2]
  target[rgbaOffset + 3] = 255
}

export function paletteToCssGradient(palette: PaletteKind): string {
  const stops = PALETTE_STOPS[palette]
  const parts = stops.map((rgb, i) => {
    const pct = (i / (stops.length - 1)) * 100
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}) ${pct.toFixed(1)}%`
  })
  return `linear-gradient(90deg, ${parts.join(', ')})`
}

export interface ColorScale {
  min: number
  max: number
  diverging: boolean
}

/**
 * Compute min/max range across one or more 2D grids.
 * For diverging scales, the range is symmetric around 0: [-absMax, +absMax].
 */
export function computeScale(grids: Grid2D[], diverging: boolean, clampMinZero = false): ColorScale {
  let min = Infinity
  let max = -Infinity

  for (const grid of grids) {
    for (let r = 0; r < grid.length; r++) {
      const row = grid[r]
      for (let c = 0; c < row.length; c++) {
        const v = row[c]
        if (v < min) min = v
        if (v > max) max = v
      }
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return diverging ? { min: -1, max: 1, diverging: true } : { min: 0, max: 1, diverging: false }
  }

  if (diverging) {
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    // Keep exact-zero difference maps cleanly centered at 0 with a small symmetric display span
    const span = absMax === 0 ? 1e-6 : absMax
    return { min: -span, max: span, diverging: true }
  }

  if (clampMinZero && min > 0) {
    min = 0
  }
  if (max === min) {
    return { min, max: min + 1, diverging: false }
  }
  return { min, max, diverging: false }
}

export function normalizeToScale(value: number, scale: ColorScale): number {
  if (scale.diverging) {
    if (value === 0) return 0.5
    const span = scale.max - scale.min
    if (span <= 0) return 0.5
    return Math.max(0, Math.min(1, (value - scale.min) / span))
  }
  const span = scale.max - scale.min
  if (span <= 0) return 0
  return Math.max(0, Math.min(1, (value - scale.min) / span))
}
