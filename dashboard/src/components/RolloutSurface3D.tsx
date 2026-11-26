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
import {
  GRATICULE_LATS,
  GRATICULE_LONS,
  type ContourSegment,
  type WindArrow,
} from '../lib/heatmap'
import type { ViewFieldId } from '../lib/vorticity'
import {
  sampleWindSpeedRgb,
  traceWindStreamlinesFor3D,
} from '../lib/windFlow'
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
  channel?: ViewFieldId
  palette: PaletteKind
  scale: ColorScale
  unit: string
  decimals?: number
  landSeaMask?: Grid2D | null
  coastlineSegments?: ContourSegment[]
  windFlowEnabled?: boolean
  diffWindArrows?: WindArrow[]
  isDifference?: boolean
  hoveredCell?: HoveredCell | null
  onHoverCell?: (cell: HoveredCell | null) => void
  emptyMessage?: string
}

// Camera tilted 50° from horizontal, North strictly up
const DEFAULT_AZIMUTH = 0
const DEFAULT_ELEVATION = (50 * Math.PI) / 180
const DEFAULT_DISTANCE = 76
const BASE_RELIEF = 3.8
const SEA_DARKEN_3D = 0.24

/** Bilinearly sample terrain height on the 3D heightfield at north-up [0..36] x [0..32] */
function sampleTerrainY(
  grid: Grid2D,
  scale: ColorScale,
  maxHeight: number,
  gx: number,
  gy: number,
): number {
  const gc = Math.max(0, Math.min(GRID_COLS - 1, gx - 0.5))
  const gr = Math.max(0, Math.min(GRID_ROWS - 1, GRID_ROWS - 1 - (gy - 0.5)))
  const c0 = Math.floor(gc)
  const c1 = Math.min(GRID_COLS - 1, c0 + 1)
  const fc = gc - c0
  const r0 = Math.floor(gr)
  const r1 = Math.min(GRID_ROWS - 1, r0 + 1)
  const fr = gr - r0

  const v0 = grid[r0][c0] + (grid[r0][c1] - grid[r0][c0]) * fc
  const v1 = grid[r1][c0] + (grid[r1][c1] - grid[r1][c0]) * fc
  const val = v0 + (v1 - v0) * fr
  const norm = normalizeToScale(val, scale)
  return scale.diverging ? (norm - 0.5) * 2 * maxHeight : norm * maxHeight
}

/** Convert north-up [0..36] x [0..32] map coords to 3D world X, Z */
function mapToWorldXZ(gx: number, gy: number): [number, number] {
  const x = gx - 0.5 - (GRID_COLS - 1) / 2
  const z = gy - 0.5 - (GRID_ROWS - 1) / 2
  return [x, z]
}

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
  landSeaMask,
  coastlineSegments,
  windFlowEnabled = false,
  diffWindArrows,
  isDifference = false,
  hoveredCell,
  onHoverCell,
  emptyMessage = 'Awaiting rollout data',
}: RolloutSurface3DProps) {
  const stageRef = useRef<HTMLDivElement | null>(null)
  const [exaggeration, setExaggeration] = useState<number>(1.0)
  const [localHover, setLocalHover] = useState<HoveredCell | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [isPointerInside, setIsPointerInside] = useState(false)

  const activeHover = hoveredCell !== undefined ? hoveredCell : localHover
  const hasGrid = grid !== null

  const threeRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    terrainGeom: THREE.PlaneGeometry
    terrainMesh: THREE.Mesh
    coastlineLines: THREE.LineSegments
    windFlowLines: THREE.LineSegments
    hoverMarker: THREE.Mesh
    raycaster: THREE.Raycaster
    azimuth: number
    elevation: number
    distance: number
    renderOnce: () => void
  } | null>(null)

  const orbitDragRef = useRef<{ startX: number; startY: number; az0: number; el0: number } | null>(
    null,
  )

  // Mount WebGL scene ONLY when real rollout data exists
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
    const camera = new THREE.PerspectiveCamera(24, w / h, 1, 300)

    const hemiLight = new THREE.HemisphereLight(0xf6f2e8, 0x1a231f, 0.78)
    scene.add(hemiLight)
    const dirLight = new THREE.DirectionalLight(0xfff6e5, 0.45)
    dirLight.position.set(-18, 36, 20)
    scene.add(dirLight)

    // Flat sea-level / zero-datum reference plane at y = 0
    const seaGeom = new THREE.PlaneGeometry(GRID_COLS, GRID_ROWS, 1, 1)
    seaGeom.rotateX(-Math.PI / 2)
    const seaMat = new THREE.MeshBasicMaterial({
      color: 0x141b18,
      transparent: true,
      opacity: 0.62,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 2,
    })
    const seaPlane = new THREE.Mesh(seaGeom, seaMat)
    seaPlane.position.y = 0
    scene.add(seaPlane)

    // Crisp 5-degree graticule lines & perimeter frame on the sea-level reference plane
    const graticuleVerts: number[] = []
    const xWest = -GRID_COLS / 2
    const xEast = GRID_COLS / 2
    const zNorth = -GRID_ROWS / 2
    const zSouth = GRID_ROWS / 2
    const yGrat = 0.015

    graticuleVerts.push(
      xWest, yGrat, zNorth, xEast, yGrat, zNorth,
      xEast, yGrat, zNorth, xEast, yGrat, zSouth,
      xEast, yGrat, zSouth, xWest, yGrat, zSouth,
      xWest, yGrat, zSouth, xWest, yGrat, zNorth,
    )
    for (const lat of GRATICULE_LATS) {
      const z = zNorth + ((38 - lat) / GRID_ROWS) * GRID_ROWS
      graticuleVerts.push(xWest, yGrat, z, xEast, yGrat, z)
    }
    for (const lon of GRATICULE_LONS) {
      const x = xWest + (lon - 65)
      graticuleVerts.push(x, yGrat, zNorth, x, yGrat, zSouth)
    }

    const gratGeom = new THREE.BufferGeometry()
    gratGeom.setAttribute('position', new THREE.Float32BufferAttribute(graticuleVerts, 3))
    const gratMat = new THREE.LineBasicMaterial({
      color: 0x26312b,
      transparent: true,
      opacity: 0.9,
    })
    const gratLines = new THREE.LineSegments(gratGeom, gratMat)
    scene.add(gratLines)

    // Smooth 32x36 heightfield mesh
    const terrainGeom = new THREE.PlaneGeometry(
      GRID_COLS - 1,
      GRID_ROWS - 1,
      GRID_COLS - 1,
      GRID_ROWS - 1,
    )
    terrainGeom.rotateX(-Math.PI / 2)
    const numVerts = GRID_COLS * GRID_ROWS
    const colorBuffer = new Float32Array(numVerts * 3)
    const colorAttr = new THREE.BufferAttribute(colorBuffer, 3)
    colorAttr.setUsage(THREE.DynamicDrawUsage)
    terrainGeom.setAttribute('color', colorAttr)
    ;(terrainGeom.attributes.position as THREE.BufferAttribute).setUsage(THREE.DynamicDrawUsage)

    const terrainMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.88,
      metalness: 0.03,
      side: THREE.DoubleSide,
    })
    const terrainMesh = new THREE.Mesh(terrainGeom, terrainMat)
    scene.add(terrainMesh)

    // 1px bone #ECE7DA coastline overlay draped on the 3D mesh
    const coastGeom = new THREE.BufferGeometry()
    const coastMat = new THREE.LineBasicMaterial({
      color: 0xece7da,
      transparent: true,
      opacity: 0.92,
    })
    const coastlineLines = new THREE.LineSegments(coastGeom, coastMat)
    coastlineLines.visible = false
    scene.add(coastlineLines)

    // Speed-colored 3D wind flow lines (or Difference wind vector arrows)
    const windGeom = new THREE.BufferGeometry()
    const windMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
    })
    const windFlowLines = new THREE.LineSegments(windGeom, windMat)
    windFlowLines.visible = false
    scene.add(windFlowLines)

    const hoverGeom = new THREE.RingGeometry(0.32, 0.52, 24)
    hoverGeom.rotateX(-Math.PI / 2)
    const hoverMat = new THREE.MeshBasicMaterial({
      color: 0xf0b84b,
      side: THREE.DoubleSide,
      depthTest: false,
    })
    const hoverMarker = new THREE.Mesh(hoverGeom, hoverMat)
    hoverMarker.visible = false
    hoverMarker.renderOrder = 10
    scene.add(hoverMarker)

    const state = {
      renderer,
      scene,
      camera,
      terrainGeom,
      terrainMesh,
      coastlineLines,
      windFlowLines,
      hoverMarker,
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
        s.camera.lookAt(0, 0.9, 0)
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
      seaGeom.dispose()
      seaMat.dispose()
      gratGeom.dispose()
      gratMat.dispose()
      terrainGeom.dispose()
      terrainMat.dispose()
      coastGeom.dispose()
      coastMat.dispose()
      windGeom.dispose()
      windMat.dispose()
      hoverGeom.dispose()
      hoverMat.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [hasGrid])

  // Update heightfield vertices, sea-cell tint, and draped 1px bone coastlines
  useEffect(() => {
    const ctx = threeRef.current
    if (!ctx || !grid) return

    const posAttr = ctx.terrainGeom.attributes.position as THREE.BufferAttribute
    const colAttr = ctx.terrainGeom.attributes.color as THREE.BufferAttribute
    const color = new THREE.Color()
    const maxHeight = BASE_RELIEF * exaggeration

    for (let iy = 0; iy < GRID_ROWS; iy++) {
      const row = GRID_ROWS - 1 - iy
      const rowData = grid[row]
      const maskRow = landSeaMask ? landSeaMask[row] : null

      for (let ix = 0; ix < GRID_COLS; ix++) {
        const vIdx = iy * GRID_COLS + ix
        const norm = normalizeToScale(rowData[ix], scale)
        const y = scale.diverging ? (norm - 0.5) * 2 * maxHeight : norm * maxHeight
        posAttr.setY(vIdx, y)

        let [red, green, blue] = samplePalette(palette, norm)
        if (maskRow) {
          const sea = (1 - Math.max(0, Math.min(1, maskRow[ix]))) * SEA_DARKEN_3D
          if (sea > 0) {
            const keep = 1 - sea
            red = Math.round(red * keep + 13 * sea)
            green = Math.round(green * keep + 18 * sea)
            blue = Math.round(blue * keep + 16 * sea)
          }
        }
        color.setRGB(red / 255, green / 255, blue / 255, THREE.SRGBColorSpace)
        colAttr.setXYZ(vIdx, color.r, color.g, color.b)
      }
    }

    posAttr.needsUpdate = true
    colAttr.needsUpdate = true
    ctx.terrainGeom.computeVertexNormals()
    ctx.terrainGeom.computeBoundingSphere()

    // Update draped 3D coastline outline if geography is loaded
    if (coastlineSegments && coastlineSegments.length > 0) {
      const verts = new Float32Array(coastlineSegments.length * 6)
      for (let i = 0; i < coastlineSegments.length; i++) {
        const [x1, y1, x2, y2] = coastlineSegments[i]
        const [wx1, wz1] = mapToWorldXZ(x1, y1)
        const [wx2, wz2] = mapToWorldXZ(x2, y2)
        const wy1 = sampleTerrainY(grid, scale, maxHeight, x1, y1) + 0.045
        const wy2 = sampleTerrainY(grid, scale, maxHeight, x2, y2) + 0.045
        const o = i * 6
        verts[o] = wx1
        verts[o + 1] = wy1
        verts[o + 2] = wz1
        verts[o + 3] = wx2
        verts[o + 4] = wy2
        verts[o + 5] = wz2
      }
      ctx.coastlineLines.geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3))
      ctx.coastlineLines.visible = true
    } else {
      ctx.coastlineLines.visible = false
    }

    ctx.renderOnce()
  }, [grid, scale, palette, exaggeration, landSeaMask, coastlineSegments, hasGrid])

  // Update 3D wind flow lines (animated particle pulses along streamlines, or Difference arrows)
  useEffect(() => {
    const ctx = threeRef.current
    if (!ctx || !grid) return

    if (!windFlowEnabled) {
      ctx.windFlowLines.visible = false
      ctx.renderOnce()
      return
    }

    const maxHeight = BASE_RELIEF * exaggeration
    const cHelper = new THREE.Color()

    // In Difference mode, draw 3D arrows of (Branch − Prime) wind
    if (isDifference) {
      if (!diffWindArrows || diffWindArrows.length === 0) {
        ctx.windFlowLines.visible = false
        ctx.renderOnce()
        return
      }

      const arrowVerts: number[] = []
      const arrowColors: number[] = []
      cHelper.setRGB(20 / 255, 27 / 255, 24 / 255, THREE.SRGBColorSpace)

      const pushSeg = (ax: number, ay: number, bx: number, by: number) => {
        const [wx1, wz1] = mapToWorldXZ(ax, ay)
        const [wx2, wz2] = mapToWorldXZ(bx, by)
        const wy1 = sampleTerrainY(grid, scale, maxHeight, ax, ay) + 0.09
        const wy2 = sampleTerrainY(grid, scale, maxHeight, bx, by) + 0.09
        arrowVerts.push(wx1, wy1, wz1, wx2, wy2, wz2)
        arrowColors.push(cHelper.r, cHelper.g, cHelper.b, cHelper.r, cHelper.g, cHelper.b)
      }

      for (const a of diffWindArrows) {
        pushSeg(a.x1, a.y1, a.x2, a.y2)
        pushSeg(a.headA[0], a.headA[1], a.x2, a.y2)
        pushSeg(a.headB[0], a.headB[1], a.x2, a.y2)
      }

      ctx.windFlowLines.geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(arrowVerts, 3),
      )
      ctx.windFlowLines.geometry.setAttribute(
        'color',
        new THREE.Float32BufferAttribute(arrowColors, 3),
      )
      ctx.windFlowLines.visible = true
      ctx.renderOnce()
      return
    }

    // Prime / Branch 3D wind streamlines colored by speed
    if (!physicalState) {
      ctx.windFlowLines.visible = false
      ctx.renderOnce()
      return
    }

    const streamlines = traceWindStreamlinesFor3D(physicalState)
    if (streamlines.length === 0) {
      ctx.windFlowLines.visible = false
      ctx.renderOnce()
      return
    }

    // Count total segments across capped streamlines
    let totalSegs = 0
    for (const sl of streamlines) {
      totalSegs += sl.points.length - 1
    }

    const positions = new Float32Array(totalSegs * 6)
    const baseRgb = new Float32Array(totalSegs * 6)
    const segPhase = new Float32Array(totalSegs)
    const colors = new Float32Array(totalSegs * 6)

    let sIdx = 0
    streamlines.forEach((sl, pathIdx) => {
      const pts = sl.points
      const pathOffset = (pathIdx * 0.37) % 1
      for (let i = 0; i < pts.length - 1; i++) {
        const [gx1, gy1, ns1] = pts[i]
        const [gx2, gy2, ns2] = pts[i + 1]
        const [wx1, wz1] = mapToWorldXZ(gx1, gy1)
        const [wx2, wz2] = mapToWorldXZ(gx2, gy2)
        const wy1 = sampleTerrainY(grid, scale, maxHeight, gx1, gy1) + 0.11
        const wy2 = sampleTerrainY(grid, scale, maxHeight, gx2, gy2) + 0.11

        const o = sIdx * 6
        positions[o] = wx1
        positions[o + 1] = wy1
        positions[o + 2] = wz1
        positions[o + 3] = wx2
        positions[o + 4] = wy2
        positions[o + 5] = wz2

        const [r1, g1, b1] = sampleWindSpeedRgb(ns1)
        cHelper.setRGB(r1 / 255, g1 / 255, b1 / 255, THREE.SRGBColorSpace)
        baseRgb[o] = cHelper.r
        baseRgb[o + 1] = cHelper.g
        baseRgb[o + 2] = cHelper.b

        const [r2, g2, b2] = sampleWindSpeedRgb(ns2)
        cHelper.setRGB(r2 / 255, g2 / 255, b2 / 255, THREE.SRGBColorSpace)
        baseRgb[o + 3] = cHelper.r
        baseRgb[o + 4] = cHelper.g
        baseRgb[o + 5] = cHelper.b

        segPhase[sIdx] = (pathOffset + i / pts.length) % 1
        sIdx++
      }
    })

    ctx.windFlowLines.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const colAttr = new THREE.BufferAttribute(colors, 3)
    colAttr.setUsage(THREE.DynamicDrawUsage)
    ctx.windFlowLines.geometry.setAttribute('color', colAttr)
    ctx.windFlowLines.visible = true

    const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    let prefersReduced = reducedQuery.matches

    const applyPulsePhase = (phase: number) => {
      for (let i = 0; i < totalSegs; i++) {
        const o = i * 6
        const w = prefersReduced
          ? 0.85
          : 0.25 + 0.75 * Math.pow(0.5 + 0.5 * Math.sin((segPhase[i] - phase) * Math.PI * 2), 2)
        for (let k = 0; k < 6; k++) {
          colors[o + k] = baseRgb[o + k] * w
        }
      }
      colAttr.needsUpdate = true
    }

    let rafId: number | null = null
    const startMs = performance.now()

    const animate = (now: number) => {
      rafId = null
      if (document.hidden || prefersReduced) return
      const phase = ((now - startMs) * 0.00075) % 1
      applyPulsePhase(phase)
      ctx.renderOnce()
      rafId = window.requestAnimationFrame(animate)
    }

    if (prefersReduced) {
      applyPulsePhase(0)
      ctx.renderOnce()
    } else {
      rafId = window.requestAnimationFrame(animate)
    }

    const onVis = () => {
      if (document.hidden) {
        if (rafId !== null) {
          window.cancelAnimationFrame(rafId)
          rafId = null
        }
      } else if (!prefersReduced && rafId === null) {
        rafId = window.requestAnimationFrame(animate)
      }
    }

    const onReduced = (e: MediaQueryListEvent) => {
      prefersReduced = e.matches
      if (prefersReduced) {
        if (rafId !== null) {
          window.cancelAnimationFrame(rafId)
          rafId = null
        }
        applyPulsePhase(0)
        ctx.renderOnce()
      } else if (rafId === null && !document.hidden) {
        rafId = window.requestAnimationFrame(animate)
      }
    }

    document.addEventListener('visibilitychange', onVis)
    reducedQuery.addEventListener('change', onReduced)

    return () => {
      if (rafId !== null) window.cancelAnimationFrame(rafId)
      document.removeEventListener('visibilitychange', onVis)
      reducedQuery.removeEventListener('change', onReduced)
    }
  }, [windFlowEnabled, isDifference, diffWindArrows, physicalState, grid, scale, exaggeration])

  // Sync hovered vertex marker on the smooth surface
  useEffect(() => {
    const ctx = threeRef.current
    if (!ctx || !grid) return
    if (!activeHover) {
      ctx.hoverMarker.visible = false
      ctx.renderOnce()
      return
    }
    const { row, col } = activeHover
    const val = grid[row]?.[col]
    if (val === undefined) {
      ctx.hoverMarker.visible = false
      ctx.renderOnce()
      return
    }
    const norm = normalizeToScale(val, scale)
    const maxHeight = BASE_RELIEF * exaggeration
    const y = scale.diverging ? (norm - 0.5) * 2 * maxHeight : norm * maxHeight
    const x = col - (GRID_COLS - 1) / 2
    const z = (GRID_ROWS - 1) / 2 - row
    ctx.hoverMarker.position.set(x, y + 0.04, z)
    ctx.hoverMarker.visible = true
    ctx.renderOnce()
  }, [activeHover, grid, scale, exaggeration])

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
      ctx.azimuth = Math.max(-0.55, Math.min(0.55, orbitDragRef.current.az0 - dx * 0.0055))
      ctx.elevation = Math.max(0.42, Math.min(1.38, orbitDragRef.current.el0 + dy * 0.0055))
      ctx.renderOnce()
      return
    }

    const ndcX = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
    const ndcY = -(((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1)
    ctx.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), ctx.camera)
    const hits = ctx.raycaster.intersectObject(ctx.terrainMesh, false)
    if (hits.length > 0) {
      const pt = hits[0].point
      const col = Math.max(0, Math.min(GRID_COLS - 1, Math.round(pt.x + (GRID_COLS - 1) / 2)))
      const row = Math.max(0, Math.min(GRID_ROWS - 1, Math.round((GRID_ROWS - 1) / 2 - pt.z)))
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
    ctx.distance = Math.max(48, Math.min(118, ctx.distance + e.deltaY * 0.04))
    ctx.renderOnce()
  }

  const resetCamera = () => {
    const ctx = threeRef.current
    if (!ctx) return
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
    if (channel === 'vorticity') {
      wireNote = 'computed'
    } else if (!isDifference && physicalState && channel) {
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
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <h3 className="font-serif-display shrink-0 text-xl italic leading-none text-[#ECE7DA]">
            {title}
          </h3>
          {subtitle && (
            <span className="truncate font-mono text-[11px] text-[#8C958D]">{subtitle}</span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-4 font-mono text-[11px] text-[#8C958D]">
          <label className="inline-flex items-center gap-2 tabular-nums">
            <span>Exaggeration {exaggeration.toFixed(1)}×</span>
            <input
              type="range"
              min={0.2}
              max={2.5}
              step={0.1}
              value={exaggeration}
              onChange={(e) => setExaggeration(Number.parseFloat(e.target.value))}
              aria-label="3D height exaggeration"
              className="instrument-range w-20"
            />
          </label>

          <button
            type="button"
            onClick={resetCamera}
            className="ui-transition hover:text-[#ECE7DA] cursor-pointer"
          >
            Reset 50° N↑
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
              N ↑ · 50° heightfield · y=0 datum plane
            </span>
            {isPointerInside && tooltipPos && hoverReadout && (
              <div
                style={{
                  left: Math.min(
                    Math.max(6, tooltipPos.x + 12),
                    (stageRef.current?.clientWidth ?? 240) - 178,
                  ),
                  top: tooltipPos.y > 76 ? tooltipPos.y - 68 : tooltipPos.y + 12,
                }}
                className="tooltip-float pointer-events-none absolute z-20 w-[170px] rounded-[3px] px-2.5 py-1.5 font-mono text-[11px] tabular-nums"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-medium text-[#ECE7DA]">{hoverReadout.formatted}</span>
                  {hoverReadout.wireNote && (
                    <span className="text-[10px] text-[#8C958D]">{hoverReadout.wireNote}</span>
                  )}
                </div>
                <div className="mt-0.5 text-[10px] text-[#8C958D]">
                  row {hoverReadout.row}, col {hoverReadout.col}
                </div>
                <div className="text-[10px] text-[#5A645D]">
                  {formatCoordLat(hoverReadout.lat)}, {formatCoordLon(hoverReadout.lon)}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <div className="mt-1.5">
        <div
          className="h-1.5 w-full rounded-[1px] border border-[#26312B]"
          style={{ background: paletteToCssGradient(palette) }}
        />
        <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-[#8C958D] tabular-nums">
          <span>{formatTick(scale.min)}</span>
          <span className="text-[#5A645D]">
            {scale.diverging ? `0 ${unit}` : `${formatTick((scale.min + scale.max) / 2)} ${unit}`}
          </span>
          <span>{formatTick(scale.max)}</span>
        </div>
      </div>
    </div>
  )
}
