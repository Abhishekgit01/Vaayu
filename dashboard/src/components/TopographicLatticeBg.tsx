import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { GRID_COLS, GRID_ROWS } from '../api/types'

/**
 * Purely decorative 32x36 mathematical lattice for the empty landing screen.
 * Driven strictly by analytic sine/cosine geometry — NOT weather data or fake values.
 */
export function TopographicLatticeBg() {
  const mountRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    let prefersReducedMotion = reducedMotionQuery.matches

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(
      34,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      200,
    )
    camera.position.set(2.5, 16.5, 29.5)
    camera.lookAt(-2.5, -1.2, 0)

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' })
    } catch {
      return
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    const group = new THREE.Group()
    group.position.set(4.2, -1.5, 0)
    scene.add(group)

    const numVerts = GRID_ROWS * GRID_COLS
    const positions = new Float32Array(numVerts * 3)
    const pointColors = new Float32Array(numVerts * 3)

    const hairlineColor = new THREE.Color('#26312B')
    const rainColor = new THREE.Color('#7FB5B0')

    const spacing = 0.92
    const halfCols = ((GRID_COLS - 1) * spacing) / 2
    const halfRows = ((GRID_ROWS - 1) * spacing) / 2

    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const idx = r * GRID_COLS + c
        const o = idx * 3
        positions[o] = c * spacing - halfCols
        positions[o + 1] = 0
        positions[o + 2] = (GRID_ROWS - 1 - r) * spacing - halfRows

        // Deterministic sparse accent points in rain #7FB5B0; remainder in hairline #26312B
        const isRainNode = (r * 7 + c * 13) % 19 === 0
        const col = isRainNode ? rainColor : hairlineColor
        pointColors[o] = col.r
        pointColors[o + 1] = col.g
        pointColors[o + 2] = col.b
      }
    }

    // Build grid line index pairs along rows and columns
    const segmentIndices: number[] = []
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const idx = r * GRID_COLS + c
        if (c + 1 < GRID_COLS) {
          segmentIndices.push(idx, idx + 1)
        }
        if (r + 1 < GRID_ROWS) {
          segmentIndices.push(idx, idx + GRID_COLS)
        }
      }
    }

    const geometry = new THREE.BufferGeometry()
    const posAttr = new THREE.BufferAttribute(positions, 3)
    posAttr.setUsage(THREE.DynamicDrawUsage)
    geometry.setAttribute('position', posAttr)
    geometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3))
    geometry.setIndex(segmentIndices)

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x26312b,
      transparent: true,
      opacity: 0.58,
    })
    const lines = new THREE.LineSegments(geometry, lineMaterial)
    group.add(lines)

    const pointsMaterial = new THREE.PointsMaterial({
      size: 0.14,
      vertexColors: true,
      transparent: true,
      opacity: 0.78,
      sizeAttenuation: true,
    })
    const points = new THREE.Points(geometry, pointsMaterial)
    group.add(points)

    const updateWave = (timeSec: number) => {
      for (let r = 0; r < GRID_ROWS; r++) {
        const ny = r / (GRID_ROWS - 1)
        for (let c = 0; c < GRID_COLS; c++) {
          const nx = c / (GRID_COLS - 1)
          const idx = r * GRID_COLS + c
          const h =
            Math.sin(nx * 5.2 + timeSec * 0.42) * Math.cos(ny * 4.1 - timeSec * 0.31) * 1.15 +
            Math.sin((nx + ny) * 3.4 - timeSec * 0.22) * 0.55
          positions[idx * 3 + 1] = h
        }
      }
      posAttr.needsUpdate = true
    }

    let pointerX = 0
    let pointerY = 0
    let currentRotX = 0
    let currentRotY = -0.18

    const onPointerMove = (e: PointerEvent) => {
      if (prefersReducedMotion) return
      pointerX = (e.clientX / Math.max(1, window.innerWidth) - 0.5) * 2
      pointerY = (e.clientY / Math.max(1, window.innerHeight) - 0.5) * 2
    }

    let rafId: number | null = null
    const startMs = performance.now()

    const renderFrame = () => {
      rafId = null
      if (document.hidden) return

      const elapsed = (performance.now() - startMs) / 1000
      updateWave(prefersReducedMotion ? 1.4 : elapsed)

      const targetY = -0.18 + pointerX * 0.06
      const targetX = pointerY * 0.04
      currentRotY += (targetY - currentRotY) * 0.04
      currentRotX += (targetX - currentRotX) * 0.04
      group.rotation.y = currentRotY
      group.rotation.x = currentRotX

      renderer.render(scene, camera)

      if (!prefersReducedMotion && !document.hidden) {
        rafId = window.requestAnimationFrame(renderFrame)
      }
    }

    const startLoop = () => {
      if (rafId === null && !document.hidden) {
        rafId = window.requestAnimationFrame(renderFrame)
      }
    }

    const stopLoop = () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId)
        rafId = null
      }
    }

    const onVisibilityChange = () => {
      if (document.hidden) {
        stopLoop()
      } else {
        startLoop()
      }
    }

    const onReducedMotionChange = (e: MediaQueryListEvent) => {
      prefersReducedMotion = e.matches
      if (prefersReducedMotion) {
        stopLoop()
        renderFrame()
      } else {
        startLoop()
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (w <= 0 || h <= 0) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(w, h)
      if (prefersReducedMotion) {
        renderer.render(scene, camera)
      }
    })
    resizeObserver.observe(container)

    window.addEventListener('pointermove', onPointerMove, { passive: true })
    document.addEventListener('visibilitychange', onVisibilityChange)
    reducedMotionQuery.addEventListener('change', onReducedMotionChange)

    // Initial render + loop
    renderFrame()

    return () => {
      stopLoop()
      resizeObserver.disconnect()
      window.removeEventListener('pointermove', onPointerMove)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      reducedMotionQuery.removeEventListener('change', onReducedMotionChange)
      geometry.dispose()
      lineMaterial.dispose()
      pointsMaterial.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div
      ref={mountRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 overflow-hidden opacity-65"
    />
  )
}
