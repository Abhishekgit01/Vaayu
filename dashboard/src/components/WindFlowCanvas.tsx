import { useEffect, useRef } from 'react'
import { GRID_COLS, GRID_ROWS, type State3D } from '../api/types'
import {
  computeMaxWindSpeed,
  createInitialParticles,
  drawStaticWindStreamlines2D,
  MAX_WIND_PARTICLES_2D,
  respawnParticle,
  sampleWindBilinear,
  sampleWindSpeedRgb,
  type WindParticle,
} from '../lib/windFlow'

interface WindFlowCanvasProps {
  active: boolean
  primaryState?: State3D | null
  secondaryState?: State3D | null
  /** Optional split position in [0..1]; left uses primaryState, right uses secondaryState */
  splitRatio?: number
}

export function WindFlowCanvas({
  active,
  primaryState,
  secondaryState,
  splitRatio,
}: WindFlowCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const particlesRef = useRef<WindParticle[] | null>(null)
  const fieldsRef = useRef<{
    primary: State3D | null | undefined
    secondary: State3D | null | undefined
    splitRatio: number | undefined
    maxSpeed: number
  }>({
    primary: primaryState,
    secondary: secondaryState,
    splitRatio,
    maxSpeed: 1,
  })

  const maxPrimary = computeMaxWindSpeed(primaryState)
  const maxSecondary = computeMaxWindSpeed(secondaryState)
  fieldsRef.current = {
    primary: primaryState,
    secondary: secondaryState,
    splitRatio,
    maxSpeed: Math.max(maxPrimary, maxSecondary),
  }

  useEffect(() => {
    if (!active) return
    const canvas = canvasRef.current
    const parent = canvas?.parentElement
    if (!canvas || !parent) return

    if (!particlesRef.current) {
      particlesRef.current = createInitialParticles(MAX_WIND_PARTICLES_2D)
    }

    const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    let prefersReduced = reducedQuery.matches

    const syncSize = () => {
      const rect = parent.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return false
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const w = Math.max(1, Math.round(rect.width * dpr))
      const h = Math.max(1, Math.round(rect.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      return true
    }

    const getFieldAtNormX = (nx: number): State3D | null | undefined => {
      const { primary, secondary, splitRatio: sr } = fieldsRef.current
      if (sr !== undefined) {
        return nx < sr ? primary : secondary
      }
      return primary
    }

    const renderReducedStatic = () => {
      if (!syncSize()) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      drawStaticWindStreamlines2D(
        ctx,
        canvas.width,
        canvas.height,
        dpr,
        getFieldAtNormX,
        fieldsRef.current.maxSpeed,
      )
    }

    let rafId: number | null = null
    let lastMs = performance.now()

    const stepFrame = (now: number) => {
      rafId = null
      if (document.hidden || prefersReduced) return
      if (!syncSize()) {
        rafId = window.requestAnimationFrame(stepFrame)
        return
      }

      const dtSec = Math.min(0.05, Math.max(0.008, (now - lastMs) / 1000))
      lastMs = now

      const ctx = canvas.getContext('2d')
      const particles = particlesRef.current
      if (!ctx || !particles) return

      const w = canvas.width
      const h = canvas.height
      const dpr = Math.min(window.devicePixelRatio || 1, 2)

      // Fade existing particle trails cleanly toward transparent without darkening the underlying map
      ctx.save()
      ctx.globalCompositeOperation = 'destination-in'
      ctx.fillStyle = 'rgba(0, 0, 0, 0.90)'
      ctx.fillRect(0, 0, w, h)
      ctx.restore()

      const maxSpeed = fieldsRef.current.maxSpeed
      // Velocity scaling in grid cells per second so moderate 10 m/s winds traverse ~2.4 cells/sec
      const advectScale = (2.8 / Math.max(6, maxSpeed)) * (dtSec * 60) * 0.08

      ctx.save()
      ctx.lineCap = 'round'
      ctx.lineWidth = 1.25 * dpr

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i]
        if (p.age >= p.maxAge) {
          respawnParticle(p)
        }

        const state = getFieldAtNormX(p.gx / GRID_COLS)
        if (!state) {
          p.age += 2
          continue
        }

        const [u, v] = sampleWindBilinear(state[2], state[3], p.gx, p.gy)
        const speed = Math.hypot(u, v)
        if (speed < maxSpeed * 0.015) {
          p.age += 3
          continue
        }

        const nextGx = p.gx + u * advectScale
        const nextGy = p.gy - v * advectScale // +v is North = negative gy

        if (nextGx < 0 || nextGx > GRID_COLS || nextGy < 0 || nextGy > GRID_ROWS) {
          respawnParticle(p)
          continue
        }

        const normS = Math.min(1, speed / maxSpeed)
        const [r, g, b] = sampleWindSpeedRgb(normS)
        const lifeFade = Math.sin((p.age / p.maxAge) * Math.PI)
        const alpha = (0.25 + 0.68 * normS) * (0.35 + 0.65 * lifeFade)

        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`
        ctx.beginPath()
        ctx.moveTo((p.gx / GRID_COLS) * w, (p.gy / GRID_ROWS) * h)
        ctx.lineTo((nextGx / GRID_COLS) * w, (nextGy / GRID_ROWS) * h)
        ctx.stroke()

        p.gx = nextGx
        p.gy = nextGy
        p.age += 1
      }

      ctx.restore()
      rafId = window.requestAnimationFrame(stepFrame)
    }

    const start = () => {
      if (prefersReduced) {
        renderReducedStatic()
      } else if (rafId === null && !document.hidden) {
        lastMs = performance.now()
        rafId = window.requestAnimationFrame(stepFrame)
      }
    }

    const stop = () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId)
        rafId = null
      }
    }

    const onVisChange = () => {
      if (document.hidden) {
        stop()
      } else {
        start()
      }
    }

    const onReducedChange = (e: MediaQueryListEvent) => {
      prefersReduced = e.matches
      stop()
      start()
    }

    const resizeObs = new ResizeObserver(() => {
      if (prefersReduced) {
        renderReducedStatic()
      } else {
        syncSize()
      }
    })
    resizeObs.observe(parent)

    document.addEventListener('visibilitychange', onVisChange)
    reducedQuery.addEventListener('change', onReducedChange)
    start()

    return () => {
      stop()
      resizeObs.disconnect()
      document.removeEventListener('visibilitychange', onVisChange)
      reducedQuery.removeEventListener('change', onReducedChange)
    }
  }, [active, primaryState, secondaryState, splitRatio])

  if (!active || (!primaryState && !secondaryState)) return null

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-[4] block h-full w-full"
    />
  )
}
