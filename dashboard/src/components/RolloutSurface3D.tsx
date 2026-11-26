import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import * as THREE from 'three'
import {
  GRID_COLS,
  GRID_ROWS,
  type DisplayChannelId,
  type Grid2D,
  type State3D,
} from '../api/types'
import {
  normalizeToScale,
  paletteToCssGradient,
  samplePalette,
  type ColorScale,
  type PaletteKind,
} from '../lib/colormaps'
import {
  colToLon,
  formatCoordLat,
  formatCoordLon,
  rowToLat,
} from '../lib/constants'
import { GRATICULE_LATS, GRATICULE_LONS } from '../lib/heatmap'
import type { HoveredCell } from './GridMap'

interface RolloutSurface3DProps {
  title: string
  accentRule: 'rain' | 'lightning' | 'hairline'
  subtitle?: string
  grid: Grid2D | null
  physicalState?: State3D | null
  compareGrids?: {
    prime?: Grid2D | null
    branch?: Grid2D | null
    diff?: Grid2D | null
  }
  channel?: DisplayChannelId
  palette: PaletteKind
  scale: ColorScale
  unit: string
  decimals?: number
  isDifference?: boolean
  hoveredCell?: HoveredCell | null
  onHoverCell?: (cell: HoveredCell | null) => void
  emptyMessage?: string
}

const DEFAULT_AZIMUTH = 0
const DEFAULT_ELEVATION = (35 * Math.PI) / 180 // ~35° tilt, North up
const DEFAULT_DISTANCE = 84

export function RolloutSurface3D({
  title,
  accentRule,
  subtitle,
  grid,
  physicalState,
  compareGrids,
  channel,
  palette,
  scale,
  unit,
  decimals = 1,
  isDifference = false,
  hoveredCell,
  onHoverCell,
  emptyMessage = 'Awaiting rollout data',
}: RolloutSurface3DProps) {
  const stageRef = useRef<HTMLDivElement | null>(null)
  const [autoRotate, setAutoRotate] = useState(false)
  const [localHover, setLocalHover] = useState<HoveredCell | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [isPointerInside, setIsPointerInside] = useState(false)

  const activeHover = hoveredCell !== undefined ? hoveredCell : localHover

  // Mutable scene handles so data/scrubbing updates never recreate the WebGL context
  const threeRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    mesh: THREE.InstancedMesh
    hoverBox: THREE.LineSegments
    raycaster: THREE.Raycaster
    azimuth: number
    elevation: number
    distance: number
    renderOnce: () => void
  } | null>(null)

  const orbitDragRef = useRef<{ startX: number; startY: number; az0: number; el0: number } | null>(
    null,
  )

  // Mount WebGL scene ONLY when real rollout grid exists
  const hasGrid = grid !== null

  useEffect(() => {
    if (!hasGrid) return
    const container = stageRef.current
    if (!container) return

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true })
    } catch {
      return
    }

    const w = Math.max(1, container.clientWidth)
    const h = Math.max(1, container.clientHeight)

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(w, h)
    renderer.setClearColor(0x0d1210, 1)
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    // Narrow FOV gives a clean orthographic-like axonometric projection
    const camera = new THREE.PerspectiveCamera(22, w / h, 1, 300)

    const ambient = new THREE.AmbientLight(0xffffff, 0.84)
    scene.add(ambient)
    const dirLight = new THREE.DirectionalLight(0xfff8ea, 0.34)
    dirLight.position.set(16, 38, 24)
    scene.add(dirLight)

    // Base 5-degree graticule frame at y = 0 (North = -Z, South = +Z, West = -X, East = +X)
    const graticuleVerts: number[] = []
    const xWest = -GRID_COLS / 2
    const xEast = GRID_COLS / 2
    const zNorth = -GRID_ROWS / 2
    const zSouth = GRID_ROWS / 2

    // Outer frame
    graticuleVerts.push(
      xWest, 0, zNorth, xEast, 0, zNorth,
      xEast, 0, zNorth, xEast, 0, zSouth,
      xEast, 0, zSouth, xWest, 0, zSouth,
      xWest, 0, zSouth, xWest, 0, zNorth,
    )
    for (const lat of GRATICULE_LATS) {
      const z = zNorth + ((38 - lat) / GRID_ROWS) * GRID_ROWS
      graticuleVerts.push(xWest, 0, z, xEast, 0, z)
    }
    for (const lon of GRATICULE_LONS) {
      const x = xWest + (lon - 65)
      graticuleVerts.push(x, 0, zNorth, x, 0, zSouth)
    }

    const gratGeom = new THREE.BufferGeometry()
    gratGeom.setAttribute('position', new THREE.Float32BufferAttribute(graticuleVerts, 3))
    const gratMat = new THREE.LineBasicMaterial({ color: 0x26312b })
    const gratLines = new THREE.LineSegments(gratGeom, gratMat)
    scene.add(gratLines)

    // Extruded 32x36 InstancedMesh (unit height box shifted +0.5 in Y so scaling Y extrudes from y=0)
    const cellGeom = new THREE.BoxGeometry(0.88, 1, 0.88)
    cellGeom.translate(0, 0.5, 0)
    const cellMat = new THREE.MeshLambertMaterial({ color: 0xffffff })
    const mesh = new THREE.InstancedMesh(cellGeom, cellMat, GRID_ROWS * GRID_COLS)
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(mesh)

    // Hover highlight wireframe box
    const hoverEdges = new THREE.EdgesGeometry(cellGeom)
    const hoverMat = new THREE.LineBasicMaterial({ color: 0xf0b84b })
    const hoverBox = new THREE.LineSegments(hoverEdges, hoverMat)
    hoverBox.visible = false
    scene.add(hoverBox)

    const state = {
      renderer,
      scene,
      camera,
      mesh,
      hoverBox,
      raycaster: new THREE.Raycaster(),
      azimuth: DEFAULT_AZIMUTH,
      elevation: DEFAULT_ELEVATION,
      distance: DEFAULT_DISTANCE,
      renderOnce: () => {
        const s = state
        const cosEl = Math.cos(s.elevation)
        s.camera.position.set(
          s.distance * cosEl * Math.sin(s.azimuth),
          s.distance * Math.sin(s.elevation),
          s.distance * cosEl * Math.cos(s.azimuth),
        )
        s.camera.lookAt(0, 1.4, 0)
        s.renderer.render(s.scene, s.camera)
      },
    }

    threeRef.current = state
    state.renderOnce()

    const resizeObserver = new ResizeObserver(() => {
      const cw = container.clientWidth
      const ch = container.clientHeight
      if (cw <= 0 || ch <= 0) return
      state.camera.aspect = cw / ch
      state.camera.updateProjectionMatrix()
      state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      state.renderer.setSize(cw, ch)
      state.renderOnce()
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      threeRef.current = null
      gratGeom.dispose()
      gratMat.dispose()
      cellGeom.dispose()
      cellMat.dispose()
      hoverEdges.dispose()
      hoverMat.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [hasGrid])

  // Update extruded column heights & palette colors whenever rollout step/channel/scale changes
  useEffect(() => {
    const ctx = threeRef.current
    if (!ctx || !grid) return

    const dummy = new THREE.Object3D()
    const color = new THREE.Color()
    const maxHeight = 6.2

    for (let r = 0; r < GRID_ROWS; r++) {
      const z = (GRID_ROWS - 1) / 2 - r // row 31 (North) -> -15.5 (away/up), row 0 (South) -> +15.5 (near/down)
      for (let c = 0; c < GRID_COLS; c++) {
        const idx = r * GRID_COLS + c
        const x = c - (GRID_COLS - 1) / 2
        const val = grid[r][c]
        const norm = normalizeToScale(val, scale)
        const heightNorm = scale.diverging ? Math.abs(norm - 0.5) * 2 : norm
        const colHeight = 0.08 + heightNorm * maxHeight

        dummy.position.set(x, 0, z)
        dummy.scale.set(1, colHeight, 1)
        dummy.updateMatrix()
        ctx.mesh.setMatrixAt(idx, dummy.matrix)

        const [red, green, blue] = samplePalette(palette, norm)
        color.setRGB(red / 255, green / 255, blue / 255, THREE.SRGBColorSpace)
        ctx.mesh.setColorAt(idx, color)
      }
    }

    ctx.mesh.instanceMatrix.needsUpdate = true
    if (ctx.mesh.instanceColor) {
      ctx.mesh.instanceColor.needsUpdate = true
    }
    ctx.mesh.computeBoundingSphere()
    ctx.renderOnce()
  }, [grid, scale, palette, hasGrid])

  // Sync hovered cell wireframe column
  useEffect(() => {
    const ctx = threeRef.current
    if (!ctx || !grid) return
    if (!activeHover) {
      ctx.hoverBox.visible = false
      ctx.renderOnce()
      return
    }
    const { row, col } = activeHover
    const val = grid[row]?.[col]
    if (val === undefined) {
      ctx.hoverBox.visible = false
      ctx.renderOnce()
      return
    }
    const norm = normalizeToScale(val, scale)
    const heightNorm = scale.diverging ? Math.abs(norm - 0.5) * 2 : norm
    const colHeight = 0.08 + heightNorm * 6.2 + 0.04
    ctx.hoverBox.position.set(col - (GRID_COLS - 1) / 2, 0, (GRID_ROWS - 1) / 2 - row)
    ctx.hoverBox.scale.set(1.04, colHeight, 1.04)
    ctx.hoverBox.visible = true
    ctx.renderOnce()
  }, [activeHover, grid, scale])

  // Optional slow auto-rotate (off by default, respects prefers-reduced-motion & tab visibility)
  useEffect(() => {
    if (!autoRotate || !hasGrid) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    let rafId: number | null = null
    let lastTime = performance.now()

    const tick = (now: number) => {
      rafId = null
      if (document.hidden) return
      const dt = Math.min(0.05, (now - lastTime) / 1000)
      lastTime = now
      const ctx = threeRef.current
      if (ctx) {
        // Gentle bounded oscillation within [-0.45, +0.45] rad so North stays oriented upward
        ctx.azimuth = Math.sin(now * 0.00035) * 0.42
        ctx.renderOnce()
      }
      rafId = window.requestAnimationFrame(tick)
    }

    const onVis = () => {
      if (!document.hidden && rafId === null) {
        lastTime = performance.now()
        rafId = window.requestAnimationFrame(tick)
      }
    }

    rafId = window.requestAnimationFrame(tick)
    document.addEventListener('visibilitychange', onVis)
    return () => {
      if (rafId !== null) window.cancelAnimationFrame(rafId)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [autoRotate, hasGrid])

  const updateHover = (cell: HoveredCell | null) => {
    if (onHoverCell) {
      onHoverCell(cell)
    } else {
      setLocalHover(cell)
    }
  }

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    const ctx = threeRef.current
    if (!ctx) return
    e.currentTarget.setPointerCapture(e.pointerId)
    setAutoRotate(false)
    orbitDragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      az0: ctx.azimuth,
      el0: ctx.elevation,
    }
  }

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const ctx = threeRef.current
    const container = stageRef.current
    if (!ctx || !container) return

    const rect = container.getBoundingClientRect()
    setIsPointerInside(true)
    setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })

    if (orbitDragRef.current) {
      const dx = e.clientX - orbitDragRef.current.startX
      const dy = e.clientY - orbitDragRef.current.startY
      // Orbit limited around North-up
      ctx.azimuth = Math.max(-0.65, Math.min(0.65, orbitDragRef.current.az0 - dx * 0.0055))
      ctx.elevation = Math.max(0.34, Math.min(1.24, orbitDragRef.current.el0 + dy * 0.0055))
      ctx.renderOnce()
      return
    }

    // Raycast cell under cursor
    const ndcX = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
    const ndcY = -(((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1)
    ctx.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), ctx.camera)
    const hits = ctx.raycaster.intersectObject(ctx.mesh, false)
    if (hits.length > 0 && hits[0].instanceId !== undefined) {
      const id = hits[0].instanceId
      const row = Math.floor(id / GRID_COLS)
      const col = id % GRID_COLS
      updateHover({ row, col })
    } else {
      updateHover(null)
    }
  }

  const handlePointerUp = () => {
    orbitDragRef.current = null
  }

  const handlePointerLeave = () => {
    orbitDragRef.current = null
    setIsPointerInside(false)
    setTooltipPos(null)
    updateHover(null)
  }

  const handleWheel = (e: ReactWheelEvent<HTMLDivElement>) => {
    const ctx = threeRef.current
    if (!ctx) return
    e.preventDefault()
    ctx.distance = Math.max(54, Math.min(125, ctx.distance + e.deltaY * 0.04))
    ctx.renderOnce()
  }

  const resetCamera = () => {
    const ctx = threeRef.current
    if (!ctx) return
    setAutoRotate(false)
    ctx.azimuth = DEFAULT_AZIMUTH
    ctx.elevation = DEFAULT_ELEVATION
    ctx.distance = DEFAULT_DISTANCE
    ctx.renderOnce()
  }

  const hoverReadout = useMemo(() => {
    if (!activeHover || !grid) return null
    const { row, col } = activeHover
    const val = grid[row]?.[col]
    if (val === undefined) return null
    const lat = rowToLat(row)
    const lon = colToLon(col)
    const sign = isDifference && val > 0 ? '+' : ''
    const formatted = `${sign}${val.toFixed(decimals)} ${unit}`

    let wireNote: string | null = null
    if (!isDifference && physicalState && channel) {
      if (channel === 't2m') {
        wireNote = `${physicalState[0][row][col].toFixed(2)} K`
      } else if (channel === 'msl') {
        wireNote = `${Math.round(physicalState[4][row][col])} Pa`
      } else if (channel === 'wind') {
        wireNote = `u=${physicalState[2][row][col].toFixed(1)}, v=${physicalState[3][row][col].toFixed(1)}`
      }
    }

    const diffVal = compareGrids?.diff?.[row]?.[col]
    const diffNote =
      !isDifference && diffVal !== undefined
        ? `Δ ${diffVal > 0 ? '+' : ''}${diffVal.toFixed(
            Math.abs(diffVal) > 0 && Math.abs(diffVal) < 0.1 ? 3 : decimals,
          )} ${unit}`
        : null

    return { row, col, lat, lon, formatted, wireNote, diffNote }
  }, [activeHover, grid, decimals, unit, isDifference, physicalState, compareGrids, channel])

  const formatTick = (v: number) => {
    if (Math.abs(v) < 1e-5) return '0'
    const sign = scale.diverging && v > 0 ? '+' : ''
    if (Math.abs(v) < 0.1) return `${sign}${v.toFixed(3)}`
    return `${sign}${v.toFixed(decimals)}`
  }

  const ruleColor =
    accentRule === 'rain' ? '#7FB5B0' : accentRule === 'lightning' ? '#F0B84B' : '#26312B'

  return (
    <div className="flex flex-col pt-2.5" style={{ borderTop: `1px solid ${ruleColor}` }}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-baseline gap-2.5">
            <h3 className="font-serif-display italic text-xl leading-none text-[#ECE7DA]">
              {title}
            </h3>
            {subtitle && (
              <span className="truncate font-mono text-[11px] text-[#8C958D]">{subtitle}</span>
            )}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-3 font-mono text-[11px] text-[#8C958D]">
          <button
            type="button"
            onClick={() => setAutoRotate((v) => !v)}
            className={`ui-transition cursor-pointer ${
              autoRotate ? 'text-[#F0B84B]' : 'hover:text-[#ECE7DA]'
            }`}
          >
            {autoRotate ? 'Drift: on' : 'Drift: off'}
          </button>
          <button
            type="button"
            onClick={resetCamera}
            className="ui-transition hover:text-[#ECE7DA] cursor-pointer"
          >
            Reset 35° N↑
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
        onWheel={handleWheel}
        className="relative aspect-[36/32] w-full cursor-grab select-none overflow-hidden rounded-[2px] border border-[#26312B] bg-[#0D1210] active:cursor-grabbing"
      >
        {!grid ? (
          <div className="flex h-full w-full items-center justify-center p-4 text-xs text-[#5A645D]">
            {emptyMessage}
          </div>
        ) : (
          <>
            <span className="pointer-events-none absolute bottom-2 left-2.5 font-mono text-[10px] text-[#5A645D]">
              N ↑ · Drag to tilt
            </span>
            {isPointerInside && tooltipPos && hoverReadout && (
              <div
                style={{
                  left: Math.min(
                    Math.max(8, tooltipPos.x + 12),
                    (stageRef.current?.clientWidth ?? 260) - 184,
                  ),
                  top: tooltipPos.y > 82 ? tooltipPos.y - 74 : tooltipPos.y + 14,
                }}
                className="tooltip-float pointer-events-none absolute z-20 w-[176px] rounded-[3px] px-2.5 py-1.5 font-mono text-[11px] tabular-nums"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-medium text-[#ECE7DA]">{hoverReadout.formatted}</span>
                  {hoverReadout.wireNote && (
                    <span className="text-[10px] text-[#8C958D]">{hoverReadout.wireNote}</span>
                  )}
                </div>
                {hoverReadout.diffNote && (
                  <div className="text-[10px] text-[#F0B84B]">{hoverReadout.diffNote}</div>
                )}
                <div className="mt-0.5 text-[10px] text-[#8C958D]">
                  r{hoverReadout.row} c{hoverReadout.col} · {formatCoordLat(hoverReadout.lat)},{' '}
                  {formatCoordLon(hoverReadout.lon)}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <div className="mt-2">
        <div
          className="h-1.5 w-full rounded-[1px] border border-[#26312B]"
          style={{ background: paletteToCssGradient(palette) }}
        />
        <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-[#8C958D] tabular-nums">
          <span>{formatTick(scale.min)}</span>
          <span>
            {scale.diverging ? `0 ${unit}` : `${formatTick((scale.min + scale.max) / 2)} ${unit}`}
          </span>
          <span>{formatTick(scale.max)}</span>
        </div>
      </div>
    </div>
  )
}
