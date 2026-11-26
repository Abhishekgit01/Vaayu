import { useEffect, useRef } from 'react'

const MAX_FPS = 15
const FRAME_INTERVAL_MS = 1000 / MAX_FPS
const FONT_PX = 14
const CELL_W = 9
const CELL_H = 16
const MAX_DPR = 2

const BG_COLOR = '#020617'
const SLATE = '#475569'
const CLOUD_CHARS = ['@', '#', '%', '&', 'W'] as const
const CLOUD_EDGE_CHARS = ['%', '&', '#', '(', ')'] as const
const GROUND_FILL_CHARS = ['-', '_', '-', '/', '\\', '-'] as const
const RAIN_TAIL_CHARS = [':', "'", '.'] as const

const CLOUD_FRACTION = 0.3
const GROUND_FRACTION = 0.15

interface RainDrop {
  col: number
  y: number
  speed: number
  length: number
}

function shouldRenderStatic(): boolean {
  if (typeof window === 'undefined') return true
  if (window.navigator?.webdriver === true) return true
  return Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)
}

/** Deterministic per-cell hash in [0, 1) so the static layers look the same on every resize. */
function hash01(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453
  return s - Math.floor(s)
}

/** Small seeded PRNG so drop placement is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Cumulonimbus underside: overlapping billow lobes plus a fine ragged edge, in rows. */
function cloudBottomRows(cols: number, canopyRows: number): Int16Array {
  const out = new Int16Array(cols)
  const lobes = [
    { c: 0.08, w: 0.1, h: 0.78 },
    { c: 0.24, w: 0.13, h: 0.95 },
    { c: 0.42, w: 0.11, h: 0.82 },
    { c: 0.58, w: 0.15, h: 1.0 },
    { c: 0.76, w: 0.12, h: 0.88 },
    { c: 0.93, w: 0.1, h: 0.8 },
  ]
  for (let x = 0; x < cols; x++) {
    const nx = x / Math.max(1, cols - 1)
    let billow = 0
    for (const lobe of lobes) {
      const d = (nx - lobe.c) / lobe.w
      billow = Math.max(billow, lobe.h * Math.exp(-d * d))
    }
    const ragged = Math.sin(x * 0.9) * 0.04 + (hash01(x, 7) - 0.5) * 0.08
    const depth = 0.5 + 0.5 * billow + ragged
    out[x] = Math.max(2, Math.min(canopyRows, Math.round(canopyRows * depth)))
  }
  return out
}

/** Uneven terrain surface row per column (rolling hills plus a few sharper ridges). */
function groundTopRows(cols: number, rows: number, groundRows: number): Int16Array {
  const out = new Int16Array(cols)
  for (let x = 0; x < cols; x++) {
    const hills =
      0.5 +
      0.28 * Math.sin(x * 0.045 + 0.6) +
      0.16 * Math.sin(x * 0.13 + 2.1) +
      0.1 * Math.sin(x * 0.31 + 4.2)
    const h = Math.max(0.12, Math.min(1, hills))
    out[x] = Math.max(0, rows - 1 - Math.round((groundRows - 1) * h))
  }
  return out
}

interface AsciiRainscapeBgProps {
  /** When true the canvas is not rendered and the effect cleanup cancels the rain loop. */
  destroyed?: boolean
}

/**
 * Full-screen retro ASCII weather scene on one 2D canvas: a static cumulonimbus canopy (top 30%),
 * static uneven terrain (bottom 15%), and rain columns falling between them, throttled to 15 FPS.
 * Decorative only, not weather data.
 */
export function AsciiRainscapeBg({ destroyed = false }: AsciiRainscapeBgProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (destroyed) return
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const staticMode = shouldRenderStatic()
    const font = `${FONT_PX}px 'Geist Mono', 'Courier New', ui-monospace, monospace`
    const staticLayer = document.createElement('canvas')

    let cssW = 0
    let cssH = 0
    let dpr = 1
    let cols = 0
    let rows = 0
    let cloudBottom: Int16Array = new Int16Array(0)
    let groundTop: Int16Array = new Int16Array(0)
    let drops: RainDrop[] = []
    let rafId: number | null = null
    let lastDraw = 0
    let stopped = false

    const spawnDrop = (drop: RainDrop, rand: () => number, anywhere: boolean) => {
      const col = Math.floor(rand() * cols)
      const start = cloudBottom[col] ?? 0
      const end = groundTop[col] ?? rows
      drop.col = col
      drop.y = anywhere ? start + rand() * Math.max(1, end - start) : start - rand() * 4
      drop.speed = 0.55 + rand() * 0.75
      drop.length = 2 + Math.floor(rand() * 3)
    }

    const paintStaticLayer = () => {
      staticLayer.width = canvas.width
      staticLayer.height = canvas.height
      const sctx = staticLayer.getContext('2d')
      if (!sctx) return
      sctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      sctx.fillStyle = BG_COLOR
      sctx.fillRect(0, 0, cssW, cssH)
      sctx.font = font
      sctx.textBaseline = 'top'

      // Cloud canopy: densest near the top, thinning toward the ragged underside
      for (let x = 0; x < cols; x++) {
        const bottom = cloudBottom[x]
        for (let y = 0; y < bottom; y++) {
          const h = hash01(x, y)
          const depthFromEdge = bottom - 1 - y
          let ch: string
          let alpha: number
          if (depthFromEdge === 0) {
            if (h < 0.35) continue
            ch = CLOUD_EDGE_CHARS[Math.floor(h * CLOUD_EDGE_CHARS.length)]
            alpha = 0.45
          } else if (depthFromEdge === 1) {
            if (h < 0.12) continue
            ch = CLOUD_CHARS[Math.floor(h * CLOUD_CHARS.length)]
            alpha = 0.62
          } else {
            ch = CLOUD_CHARS[Math.floor(h * CLOUD_CHARS.length)]
            alpha = 0.72 + 0.28 * Math.min(1, depthFromEdge / 6)
          }
          sctx.globalAlpha = alpha
          sctx.fillStyle = SLATE
          sctx.fillText(ch, x * CELL_W, y * CELL_H)
        }
      }

      // Terrain: slope-aware surface line, then sparse structural fill below it
      for (let x = 0; x < cols; x++) {
        const top = groundTop[x]
        const left = groundTop[Math.max(0, x - 1)]
        const right = groundTop[Math.min(cols - 1, x + 1)]
        let surface = '_'
        if (top < left && top < right) surface = '^'
        else if (right < top) surface = '/'
        else if (left < top) surface = '\\'
        else if (top === left && top === right) surface = hash01(x, 3) < 0.25 ? '-' : '_'

        sctx.globalAlpha = 0.95
        sctx.fillStyle = SLATE
        sctx.fillText(surface, x * CELL_W, top * CELL_H)

        for (let y = top + 1; y < rows; y++) {
          const h = hash01(x + 101, y)
          if (h < 0.45) continue
          sctx.globalAlpha = 0.35 + 0.25 * Math.min(1, (y - top) / 4)
          sctx.fillText(GROUND_FILL_CHARS[Math.floor(h * GROUND_FILL_CHARS.length)], x * CELL_W, y * CELL_H)
        }
      }
      sctx.globalAlpha = 1
    }

    const resize = () => {
      const w = wrap.clientWidth
      const h = wrap.clientHeight
      if (w <= 0 || h <= 0) return
      cssW = w
      cssH = h
      dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR)
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      cols = Math.ceil(w / CELL_W)
      rows = Math.ceil(h / CELL_H)

      const canopyRows = Math.max(3, Math.round(rows * CLOUD_FRACTION))
      const groundRows = Math.max(3, Math.round(rows * GROUND_FRACTION))
      cloudBottom = cloudBottomRows(cols, canopyRows)
      groundTop = groundTopRows(cols, rows, groundRows)

      const rand = mulberry32(cols * 7919 + rows)
      const dropCount = Math.round(cols * 0.55)
      drops = Array.from({ length: dropCount }, () => {
        const d: RainDrop = { col: 0, y: 0, speed: 0, length: 0 }
        spawnDrop(d, rand, true)
        return d
      })

      paintStaticLayer()
    }

    const respawnRand = mulberry32(20_240_611)

    const draw = () => {
      if (cols === 0 || rows === 0) return
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.drawImage(staticLayer, 0, 0)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.font = font
      ctx.textBaseline = 'top'

      // Rain heads
      ctx.fillStyle = 'rgba(34, 211, 238, 0.42)'
      for (const d of drops) {
        const headRow = Math.floor(d.y)
        if (headRow < cloudBottom[d.col] || headRow >= groundTop[d.col]) continue
        ctx.fillText('|', d.col * CELL_W, headRow * CELL_H)
      }

      // Fading tails
      ctx.fillStyle = 'rgba(34, 211, 238, 0.2)'
      for (const d of drops) {
        const headRow = Math.floor(d.y)
        const top = cloudBottom[d.col]
        const ground = groundTop[d.col]
        for (let i = 1; i < d.length; i++) {
          const row = headRow - i
          if (row < top || row >= ground) continue
          ctx.fillText(RAIN_TAIL_CHARS[Math.min(RAIN_TAIL_CHARS.length - 1, i - 1)], d.col * CELL_W, row * CELL_H)
        }
      }
    }

    const advance = () => {
      for (const d of drops) {
        d.y += d.speed
        if (d.y - d.length >= groundTop[d.col]) {
          spawnDrop(d, respawnRand, false)
        }
      }
    }

    const tick = (now: number) => {
      rafId = null
      if (stopped) return
      // Hard 15 FPS cap; skip work entirely while the tab is hidden
      if (!document.hidden && now - lastDraw >= FRAME_INTERVAL_MS) {
        lastDraw = now
        advance()
        draw()
      }
      rafId = window.requestAnimationFrame(tick)
    }

    resize()
    draw()
    if (!staticMode) {
      rafId = window.requestAnimationFrame(tick)
    }

    const observer = new ResizeObserver(() => {
      resize()
      draw()
    })
    observer.observe(wrap)

    return () => {
      stopped = true
      if (rafId !== null) window.cancelAnimationFrame(rafId)
      observer.disconnect()
    }
  }, [destroyed])

  if (destroyed) return null

  return (
    <div
      ref={wrapRef}
      aria-hidden="true"
      data-ascii-rainscape="true"
      className="pointer-events-none absolute inset-0 overflow-hidden"
      style={{ zIndex: -1, pointerEvents: 'none', backgroundColor: BG_COLOR }}
    >
      <canvas ref={canvasRef} className="block h-full w-full" style={{ pointerEvents: 'none' }} />
    </div>
  )
}
