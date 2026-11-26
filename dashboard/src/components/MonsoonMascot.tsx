import { useEffect, useRef } from 'react'
import type { HealthState } from './StatusBar'
import {
  ALL_LIGHTNING,
  DUCK_SWIM,
  FROG_RIBBIT,
  FROG_SIT,
  MEGH_BLINK,
  MEGH_IDLE,
  MEGH_MOUTH_OPEN,
  MEGH_SHOCKED,
  MEGH_SLEEPY,
  PEACOCK_DANCE,
  PEACOCK_REST,
  RAIN_FRAMES,
  RAIN_HEAVY_FRAMES,
  SCENE_HEIGHT,
  SCENE_WIDTH,
  SNAIL_FRAMES,
  UMBRELLA_SPLASH,
  UMBRELLA_WALK,
} from './mascotFrames'

const TICK_MS = 100
const SINE_PERIOD_MS = 6000
const BLINK_INTERVAL_MS = 3200
const BLINK_DURATION_MS = 280
const THUNDER_TOTAL_MS = 900
const FROG_MIN_MS = 8000
const FROG_MAX_MS = 12000
const FROG_SHOW_MS = 3200
const PERSON_STEP_MS = 380
const PERSON_MAX_COL = 24

function randBetween(min: number, max: number) {
  return min + Math.random() * (max - min)
}

/** Overlay `sprite` onto `canvas` starting at (row, col), skipping spaces so sprites compose cleanly. */
function blit(canvas: string[][], sprite: readonly string[], row: number, col: number) {
  for (let r = 0; r < sprite.length; r++) {
    const targetRow = canvas[row + r]
    if (!targetRow) continue
    const line = sprite[r]
    for (let c = 0; c < line.length; c++) {
      const ch = line[c]
      if (ch !== ' ' && col + c >= 0 && col + c < SCENE_WIDTH) {
        targetRow[col + c] = ch
      }
    }
  }
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

    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      const rows = Array.from({ length: SCENE_HEIGHT }, () => Array(SCENE_WIDTH).fill(' '))
      blit(rows, MEGH_IDLE, 0, 2)
      blit(rows, PEACOCK_REST, 28, 64)
      pre.textContent = rows.map((r) => r.join('')).join('\n')
      return
    }

    let tick = 0
    let blinkUntil = 0
    let nextBlink = Date.now() + BLINK_INTERVAL_MS

    let thunderOn = false
    let thunderStart = 0
    let wakeFlashDone = false
    let prevReady = propsRef.current.health.kind === 'ready'

    let personCol = 2
    let personFrame = 0
    let lastPersonStep = Date.now()

    let frogShow = false
    let frogHopping = false
    let frogStart = 0
    let frogCol = 22
    let nextFrog = Date.now() + randBetween(FROG_MIN_MS, FROG_MAX_MS)

    let snailCol = 43
    let stopped = false

    const frame = () => {
      if (stopped || document.hidden) return
      const now = Date.now()
      tick++

      const { health: hp, isDragging: drag } = propsRef.current
      const ready = hp.kind === 'ready'

      if (ready && !thunderOn && now >= nextBlink && now > blinkUntil) {
        blinkUntil = now + BLINK_DURATION_MS
        nextBlink = now + BLINK_INTERVAL_MS + randBetween(-500, 500)
      }
      const isBlinking = now < blinkUntil

      if (ready && !prevReady && !wakeFlashDone) {
        thunderOn = true
        thunderStart = now
        wakeFlashDone = true
      }
      if (thunderOn && now - thunderStart >= THUNDER_TOTAL_MS) thunderOn = false
      prevReady = ready

      if (ready && now - lastPersonStep > PERSON_STEP_MS) {
        lastPersonStep = now
        personCol = (personCol + 1) % PERSON_MAX_COL
        personFrame = (personFrame + 1) % UMBRELLA_WALK.length
      }

      if (ready && tick % 18 === 0) {
        snailCol = snailCol <= 40 ? 44 : snailCol - 1
      }

      if (ready && !frogShow && now >= nextFrog) {
        frogShow = true
        frogHopping = true
        frogStart = now
        frogCol = 18 + Math.floor(Math.random() * 8)
      }
      if (frogShow) {
        const elapsed = now - frogStart
        frogHopping = elapsed < 1000
        if (elapsed > FROG_SHOW_MS) {
          frogShow = false
          nextFrog = now + randBetween(FROG_MIN_MS, FROG_MAX_MS)
        }
      }

      const buf = Array.from({ length: SCENE_HEIGHT }, () => Array<string>(SCENE_WIDTH).fill(' '))
      const cloudCol = 2 + Math.round(Math.sin((now / SINE_PERIOD_MS) * Math.PI * 2) * 2)

      // 1. Exact pasted Megh cloud (rows 0..26)
      let face: readonly string[]
      if (!ready) {
        face = MEGH_SLEEPY[Math.floor((tick / 8) % MEGH_SLEEPY.length)]
      } else if (thunderOn) {
        face = MEGH_SHOCKED
      } else if (drag) {
        face = MEGH_MOUTH_OPEN
      } else if (isBlinking) {
        face = MEGH_BLINK
      } else {
        face = MEGH_IDLE
      }
      blit(buf, face, 0, cloudCol)

      // 2. Rain or 3-frame lightning under Megh (rows 19..22)
      let hasLightning = false
      if (ready && thunderOn) {
        const fi = Math.min(2, Math.floor((now - thunderStart) / 300))
        blit(buf, ALL_LIGHTNING[fi], 19, cloudCol + 26)
        hasLightning = true
      } else if (ready) {
        const rainSrc = drag ? RAIN_HEAVY_FRAMES : RAIN_FRAMES
        blit(buf, rainSrc[tick % rainSrc.length], 19, cloudCol + 34)
      }

      // 3. Ground flora/animals (rows 28..34)
      if (ready) {
        blit(buf, UMBRELLA_WALK[personFrame], 28, personCol)
        blit(buf, [UMBRELLA_SPLASH[tick % UMBRELLA_SPLASH.length]], 33, personCol)

        const duckFrame = DUCK_SWIM[Math.floor(tick / 6) % DUCK_SWIM.length]
        blit(buf, duckFrame, 31, 38)

        blit(buf, SNAIL_FRAMES[Math.floor(tick / 9) % SNAIL_FRAMES.length], 32, snailCol + 8)

        const peacockFrame = PEACOCK_DANCE[Math.floor(tick / 5) % PEACOCK_DANCE.length]
        blit(buf, peacockFrame, 28, 64)

        if (frogShow) {
          const frogRow = frogHopping ? 31 : 32
          const frogSprite = now - frogStart > FROG_SHOW_MS * 0.45 ? FROG_RIBBIT : FROG_SIT
          blit(buf, frogSprite, frogRow, frogCol + 6)
        }
      } else {
        blit(buf, PEACOCK_REST, 28, 64)
      }

      // Subtle wet ridge floor (row 34)
      const floor = '  ..---..___..---..~~~~~~..---..___..---..___..---..___..---..___..---..      '
      blit(buf, [floor], 34, 0)

      pre.textContent = buf.map((r) => r.join('')).join('\n')
      pre.classList.toggle('mascot-flash', hasLightning)
    }

    const onVis = () => {
      if (!document.hidden) nextBlink = Date.now() + BLINK_INTERVAL_MS
    }
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
        .mascot-flash { color: rgba(240, 184, 75, 0.88) !important; text-shadow: 0 0 10px rgba(240, 184, 75, 0.65); }
        @media (max-width: 900px) { .monsoon-mascot-wrap { display: none !important; } }
      `}</style>
      <div
        className="monsoon-mascot-wrap pointer-events-none absolute right-[3%] top-[56%] z-[5] -translate-y-1/2 select-none"
        aria-hidden="true"
      >
        <pre
          ref={preRef}
          className="m-0 whitespace-pre font-mono leading-[1.15]"
          style={{
            fontSize: 'clamp(8px, 0.72vw, 11px)',
            color: 'rgba(127, 181, 176, 0.72)',
          }}
        />
      </div>
    </>
  )
}
