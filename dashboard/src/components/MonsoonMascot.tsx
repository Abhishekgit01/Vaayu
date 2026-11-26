import { useEffect, useRef } from 'react'
import type { HealthState } from './StatusBar'
import {
  MEGH_IDLE, MEGH_BLINK, MEGH_SLEEPY,
  MEGH_MOUTH_OPEN, MEGH_SHOCKED,
  RAIN_FRAMES, RAIN_HEAVY_FRAMES,
  ALL_LIGHTNING,
  UMBRELLA_WALK, UMBRELLA_SPLASH,
  FROG_SIT, FROG_RIBBIT,
  SCENE_WIDTH,
} from './mascotFrames'

// ---- timing constants ----
const TICK_MS         = 100
const SINE_PERIOD     = 6000
const BLINK_INTERVAL  = 3000
const BLINK_DURATION  = 300
const THUNDER_TOTAL   = 900
const FROG_MIN_MS     = 8000
const FROG_MAX_MS     = 12000
const FROG_SHOW_DUR   = 3200
const PERSON_STEP_MS  = 400
const PERSON_COLS     = 40

function randBetween(a: number, b: number) {
  return a + Math.random() * (b - a)
}

interface MonsoonMascotProps {
  health: HealthState
  isDragging: boolean
}

export function MonsoonMascot({ health, isDragging }: MonsoonMascotProps) {
  const preRef = useRef<HTMLPreElement | null>(null)
  const propsRef = useRef({ health, isDragging })
  propsRef.current = { health, isDragging }

  useEffect(() => {
    const pre = preRef.current
    if (!pre) return

    // prefers-reduced-motion → single static frame, no timer
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      pre.textContent = MEGH_IDLE.map(l => l.padEnd(SCENE_WIDTH)).join('\n')
      return
    }

    // ---- mutable state ----
    let tick = 0
    let blinkUntil = 0
    let nextBlink = Date.now() + BLINK_INTERVAL

    let thunderOn = false
    let thunderStart = 0
    let wakeFlashDone = false
    let prevReady = propsRef.current.health.kind === 'ready'

    let personCol = 0
    let personFrame = 0
    let lastPersonStep = Date.now()

    let frogShow = false
    let frogStart = 0
    let frogCol = 10
    let nextFrog = Date.now() + randBetween(FROG_MIN_MS, FROG_MAX_MS)

    let stopped = false

    function frame() {
      if (stopped || document.hidden) return
      const now = Date.now()
      tick++

      const { health: hp, isDragging: drag } = propsRef.current
      const ready = hp.kind === 'ready'

      // ---- blink ----
      if (ready && !thunderOn && now >= nextBlink && now > blinkUntil) {
        blinkUntil = now + BLINK_DURATION
        nextBlink = now + BLINK_INTERVAL + randBetween(-500, 500)
      }
      const isBlinking = now < blinkUntil

      // ---- wake-up thunder ----
      if (ready && !prevReady && !wakeFlashDone) {
        thunderOn = true
        thunderStart = now
        wakeFlashDone = true
      }
      if (thunderOn && now - thunderStart >= THUNDER_TOTAL) thunderOn = false
      prevReady = ready

      // ---- person ----
      if (ready && now - lastPersonStep > PERSON_STEP_MS) {
        lastPersonStep = now
        personCol = (personCol + 1) % PERSON_COLS
        personFrame = (personFrame + 1) % UMBRELLA_WALK.length
      }

      // ---- frog ----
      if (ready && !frogShow && now >= nextFrog) {
        frogShow = true
        frogStart = now
        frogCol = 5 + Math.floor(Math.random() * 30)
      }
      if (frogShow && now - frogStart > FROG_SHOW_DUR) {
        frogShow = false
        nextFrog = now + randBetween(FROG_MIN_MS, FROG_MAX_MS)
      }

      // ---- compose ----
      const lines: string[] = []
      const drift = Math.round(Math.sin((now / SINE_PERIOD) * Math.PI * 2) * 3)
      const pad = (s: string) => ' '.repeat(Math.max(0, drift + 4)) + s

      // cloud face
      let face: readonly string[]
      if (!ready) {
        face = MEGH_SLEEPY[Math.floor((tick / 10) % MEGH_SLEEPY.length)]
      } else if (thunderOn) {
        face = MEGH_SHOCKED
      } else if (drag) {
        face = MEGH_MOUTH_OPEN
      } else if (isBlinking) {
        face = MEGH_BLINK
      } else {
        face = MEGH_IDLE
      }
      for (const l of face) lines.push(pad(l))

      // rain / lightning / blank
      let hasLightning = false
      if (ready && thunderOn) {
        const fi = Math.min(2, Math.floor((now - thunderStart) / 300))
        for (const l of ALL_LIGHTNING[fi]) lines.push(pad(l))
        hasLightning = true
      } else if (ready) {
        const src = drag ? RAIN_HEAVY_FRAMES : RAIN_FRAMES
        for (const l of src[tick % src.length]) lines.push(pad(l))
      } else {
        lines.push('', '', '')
      }

      lines.push('')

      // umbrella person
      if (ready) {
        for (const l of UMBRELLA_WALK[personFrame])
          lines.push(' '.repeat(personCol) + l)
        for (const l of UMBRELLA_SPLASH)
          lines.push(' '.repeat(personCol) + l)
      } else {
        for (let i = 0; i < 7; i++) lines.push('')
      }

      // frog
      if (frogShow) {
        const fFr = (now - frogStart > FROG_SHOW_DUR * 0.5) ? FROG_RIBBIT : FROG_SIT
        for (const l of fFr) lines.push(' '.repeat(frogCol) + l)
      } else {
        lines.push('', '')
      }

      pre.textContent = lines.map(l => l.padEnd(SCENE_WIDTH)).join('\n')
      if (hasLightning) pre.classList.add('mascot-flash')
      else pre.classList.remove('mascot-flash')
    }

    const onVis = () => { if (!document.hidden) nextBlink = Date.now() + BLINK_INTERVAL }
    document.addEventListener('visibilitychange', onVis)
    const id = setInterval(frame, TICK_MS)
    frame()

    return () => {
      stopped = true
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  return (
    <>
      <style>{`
        .mascot-flash { text-shadow: 0 0 8px rgba(240,184,75,0.7); }
        @media (max-width: 900px) { .monsoon-mascot-wrap { display: none !important; } }
      `}</style>
      <div
        className="monsoon-mascot-wrap"
        aria-hidden="true"
        style={{
          position: 'absolute',
          right: '5%',
          top: '50%',
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      >
        <pre
          ref={preRef}
          style={{
            fontFamily: "'Geist Mono','Courier New',ui-monospace,monospace",
            fontSize: 13,
            lineHeight: 1.3,
            color: 'rgba(127,181,176,0.7)',
            margin: 0,
            whiteSpace: 'pre',
            userSelect: 'none',
          }}
        />
      </div>
    </>
  )
}
