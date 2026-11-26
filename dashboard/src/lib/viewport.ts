import { GRID_COLS, GRID_ROWS } from '../api/types'
import { LAT_ORIGIN, LON_ORIGIN } from './constants'

export const MIN_MAP_ZOOM = 1
export const MAX_MAP_ZOOM = 6

/**
 * Shared north-up continuous viewport over [0..GRID_COLS] x [0..GRID_ROWS],
 * where gx = 0 is 65°E (west), gx = 36 is 101°E (east),
 * gy = 0 is 38°N (north/top), gy = 32 is 6°N (south/bottom).
 */
export interface MapViewport {
  zoom: number
  centerGx: number
  centerGy: number
}

export interface ViewportBounds {
  zoom: number
  gxMin: number
  gxMax: number
  gyMin: number
  gyMax: number
  spanGx: number
  spanGy: number
}

export const DEFAULT_MAP_VIEWPORT: MapViewport = {
  zoom: 1,
  centerGx: GRID_COLS / 2,
  centerGy: GRID_ROWS / 2,
}

export function clampMapViewport(viewport: MapViewport): MapViewport {
  const zoom = Math.max(
    MIN_MAP_ZOOM,
    Math.min(MAX_MAP_ZOOM, Number.isFinite(viewport.zoom) ? viewport.zoom : 1),
  )
  const halfSpanX = GRID_COLS / (2 * zoom)
  const halfSpanY = GRID_ROWS / (2 * zoom)

  const centerGx = Math.max(
    halfSpanX,
    Math.min(GRID_COLS - halfSpanX, Number.isFinite(viewport.centerGx) ? viewport.centerGx : GRID_COLS / 2),
  )
  const centerGy = Math.max(
    halfSpanY,
    Math.min(GRID_ROWS - halfSpanY, Number.isFinite(viewport.centerGy) ? viewport.centerGy : GRID_ROWS / 2),
  )

  return { zoom, centerGx, centerGy }
}

export function getViewportBounds(viewport?: MapViewport | null): ViewportBounds {
  const vp = viewport ? clampMapViewport(viewport) : DEFAULT_MAP_VIEWPORT
  const spanGx = GRID_COLS / vp.zoom
  const spanGy = GRID_ROWS / vp.zoom
  const gxMin = vp.centerGx - spanGx / 2
  const gyMin = vp.centerGy - spanGy / 2
  return {
    zoom: vp.zoom,
    gxMin,
    gxMax: gxMin + spanGx,
    gyMin,
    gyMax: gyMin + spanGy,
    spanGx,
    spanGy,
  }
}

export function zoomViewportAtNorm(
  viewport: MapViewport,
  nextZoomRaw: number,
  normX = 0.5,
  normY = 0.5,
): MapViewport {
  const curBounds = getViewportBounds(viewport)
  const nx = Math.max(0, Math.min(1, normX))
  const ny = Math.max(0, Math.min(1, normY))
  const nextZoom = Math.max(MIN_MAP_ZOOM, Math.min(MAX_MAP_ZOOM, nextZoomRaw))

  const focusGx = curBounds.gxMin + nx * curBounds.spanGx
  const focusGy = curBounds.gyMin + ny * curBounds.spanGy

  const nextSpanGx = GRID_COLS / nextZoom
  const nextSpanGy = GRID_ROWS / nextZoom
  const nextGxMin = focusGx - nx * nextSpanGx
  const nextGyMin = focusGy - ny * nextSpanGy

  return clampMapViewport({
    zoom: nextZoom,
    centerGx: nextGxMin + nextSpanGx / 2,
    centerGy: nextGyMin + nextSpanGy / 2,
  })
}

export function panViewportByNormDelta(
  viewport: MapViewport,
  deltaNormX: number,
  deltaNormY: number,
): MapViewport {
  const bounds = getViewportBounds(viewport)
  return clampMapViewport({
    zoom: bounds.zoom,
    centerGx: viewport.centerGx - deltaNormX * bounds.spanGx,
    centerGy: viewport.centerGy - deltaNormY * bounds.spanGy,
  })
}

/**
 * Compute a zoomed viewport framing a boolean [32][36] shock mask plus ~3-cell locality margin.
 */
export function computeViewportForShockMask(mask: boolean[][] | null | undefined): MapViewport {
  if (!mask) return DEFAULT_MAP_VIEWPORT
  let rMin: number = GRID_ROWS
  let rMax = -1
  let cMin: number = GRID_COLS
  let cMax = -1

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      if (mask[r]?.[c]) {
        if (r < rMin) rMin = r
        if (r > rMax) rMax = r
        if (c < cMin) cMin = c
        if (c > cMax) cMax = c
      }
    }
  }
  if (rMax < rMin || cMax < cMin) return DEFAULT_MAP_VIEWPORT

  const pad = 4
  const spanCols = Math.min(GRID_COLS, cMax - cMin + 1 + pad * 2)
  const spanRows = Math.min(GRID_ROWS, rMax - rMin + 1 + pad * 2)
  const zoom = Math.max(
    MIN_MAP_ZOOM,
    Math.min(4.5, Math.min(GRID_COLS / spanCols, GRID_ROWS / spanRows)),
  )

  const centerGx = (cMin + cMax + 1) / 2
  const gyTop = GRID_ROWS - 1 - rMax
  const gyBot = GRID_ROWS - rMin
  const centerGy = (gyTop + gyBot) / 2

  return clampMapViewport({ zoom, centerGx, centerGy })
}

/**
 * Map continuous north-up grid coords (gx, gy) into viewport-relative [0..1] fractions.
 */
export function gxToViewportNormX(gx: number, bounds: ViewportBounds): number {
  return (gx - bounds.gxMin) / bounds.spanGx
}

export function gyToViewportNormY(gy: number, bounds: ViewportBounds): number {
  return (gy - bounds.gyMin) / bounds.spanGy
}

/**
 * Convert pointer position [0..1] inside a zoomed map container to model [row, col].
 */
export function viewportNormToCell(
  normX: number,
  normY: number,
  bounds: ViewportBounds,
): { row: number; col: number } | null {
  if (normX < 0 || normX > 1 || normY < 0 || normY > 1) return null
  const gx = bounds.gxMin + normX * bounds.spanGx
  const gy = bounds.gyMin + normY * bounds.spanGy
  const col = Math.max(0, Math.min(GRID_COLS - 1, Math.floor(gx)))
  const visualY = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor(gy)))
  return { row: GRID_ROWS - 1 - visualY, col }
}

export interface GraticuleTick {
  value: number
  normPos: number
}

/**
 * Dynamically generate latitude & longitude graticule lines/labels for any zoom level.
 * Cell center row 0 = 6.5°N (south edge 6.0°N), row 31 = 37.5°N (north edge 38.0°N).
 * Cell center col 0 = 65.5°E (west edge 65.0°E), col 35 = 100.5°E (east edge 101.0°E).
 */
export function getViewportGraticuleTicks(bounds: ViewportBounds): {
  lats: GraticuleTick[]
  lons: GraticuleTick[]
} {
  const step = bounds.zoom >= 3.8 ? 1 : bounds.zoom >= 1.9 ? 2 : 5
  const northEdgeLat = LAT_ORIGIN - 0.5 + GRID_ROWS // 38.0
  const westEdgeLon = LON_ORIGIN - 0.5 // 65.0

  const latMax = northEdgeLat - bounds.gyMin
  const latMin = northEdgeLat - bounds.gyMax
  const lonMin = westEdgeLon + bounds.gxMin
  const lonMax = westEdgeLon + bounds.gxMax

  const lats: GraticuleTick[] = []
  const firstLat = Math.ceil((latMin + 1e-6) / step) * step
  for (let lat = firstLat; lat <= latMax - 1e-6; lat += step) {
    if (lat <= 6.1 || lat >= 37.9) continue
    const gy = northEdgeLat - lat
    const normPos = gyToViewportNormY(gy, bounds)
    if (normPos >= 0.02 && normPos <= 0.98) {
      lats.push({ value: lat, normPos })
    }
  }

  const lons: GraticuleTick[] = []
  const firstLon = Math.ceil((lonMin + 1e-6) / step) * step
  for (let lon = firstLon; lon <= lonMax - 1e-6; lon += step) {
    if (lon <= 65.1 || lon >= 100.9) continue
    const gx = lon - westEdgeLon
    const normPos = gxToViewportNormX(gx, bounds)
    if (normPos >= 0.02 && normPos <= 0.98) {
      lons.push({ value: lon, normPos })
    }
  }

  return { lats, lons }
}
