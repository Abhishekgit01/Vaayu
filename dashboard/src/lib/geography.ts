import { GRID_COLS, GRID_ROWS, type Grid2D } from '../api/types'
import { LAT_ORIGIN, LON_ORIGIN } from './constants'
import { buildNorthUpScalarVertexField, type ContourSegment } from './heatmap'
import { extractIsoContourSegments } from './shock'

/**
 * In-memory geography overlay built from a user-imported GeoJSON (.geojson)
 * or 32x36 land_sea_mask JSON. Never saved to localStorage or disk; no geographic data is bundled.
 */
export interface GeographyOverlayData {
  /** [32][36] land fraction in [0..1] (row 0 = south 6.5°N, col 0 = west 65.5°E; >= 0.5 = land) */
  landSeaMask: Grid2D
  /** Coastline segments in [0..36] x [0..32] north-up map coordinates */
  coastlineSegments: ContourSegment[]
  sourceLabel: string
}

export type GeographyValidationResult =
  | { ok: true; data: GeographyOverlayData }
  | { ok: false; error: string }

const WEST_EDGE_LON = LON_ORIGIN - 0.5 // 65.0
const NORTH_EDGE_LAT = LAT_ORIGIN - 0.5 + GRID_ROWS // 38.0

function lonLatToGridXY(lon: number, lat: number): [number, number] {
  return [lon - WEST_EDGE_LON, NORTH_EDGE_LAT - lat]
}

function pointInRing(lon: number, lat: number, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0]
    const yi = ring[i][1]
    const xj = ring[j][0]
    const yj = ring[j][1]
    const intersect =
      yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

function pointInPolygonRings(lon: number, lat: number, rings: number[][][]): boolean {
  if (rings.length === 0 || !pointInRing(lon, lat, rings[0])) return false
  for (let h = 1; h < rings.length; h++) {
    if (pointInRing(lon, lat, rings[h])) return false
  }
  return true
}

function collectGeoJsonGeometries(
  node: unknown,
  polygons: number[][][][],
  lines: number[][][],
): boolean {
  if (!node || typeof node !== 'object' || Array.isArray(node)) return false
  const obj = node as Record<string, unknown>
  const gType = obj.type

  if (gType === 'FeatureCollection' && Array.isArray(obj.features)) {
    let any = false
    for (const f of obj.features) {
      if (collectGeoJsonGeometries(f, polygons, lines)) any = true
    }
    return any
  }
  if (gType === 'Feature') {
    return collectGeoJsonGeometries(obj.geometry, polygons, lines)
  }
  if (gType === 'GeometryCollection' && Array.isArray(obj.geometries)) {
    let any = false
    for (const g of obj.geometries) {
      if (collectGeoJsonGeometries(g, polygons, lines)) any = true
    }
    return any
  }
  if (gType === 'Polygon' && Array.isArray(obj.coordinates)) {
    const rings = obj.coordinates as number[][][]
    polygons.push(rings)
    for (const ring of rings) {
      if (Array.isArray(ring) && ring.length >= 2) lines.push(ring)
    }
    return true
  }
  if (gType === 'MultiPolygon' && Array.isArray(obj.coordinates)) {
    const polys = obj.coordinates as number[][][][]
    for (const rings of polys) {
      if (!Array.isArray(rings)) continue
      polygons.push(rings)
      for (const ring of rings) {
        if (Array.isArray(ring) && ring.length >= 2) lines.push(ring)
      }
    }
    return true
  }
  if (gType === 'LineString' && Array.isArray(obj.coordinates)) {
    lines.push(obj.coordinates as number[][])
    return true
  }
  if (gType === 'MultiLineString' && Array.isArray(obj.coordinates)) {
    for (const line of obj.coordinates as number[][][]) {
      if (Array.isArray(line) && line.length >= 2) lines.push(line)
    }
    return true
  }
  return false
}

export function isGeographyPayload(parsed: unknown): boolean {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return false
  const obj = parsed as Record<string, unknown>
  if (Array.isArray(obj.init_states)) return false
  if (Array.isArray(obj.land_sea_mask)) return true
  const t = obj.type
  return (
    t === 'FeatureCollection' ||
    t === 'Feature' ||
    t === 'GeometryCollection' ||
    t === 'Polygon' ||
    t === 'MultiPolygon' ||
    t === 'LineString' ||
    t === 'MultiLineString'
  )
}

export function parseAndValidateGeographyJson(
  rawText: string,
  sourceLabel = 'geography.geojson',
): GeographyValidationResult {
  const trimmed = rawText.trim()
  if (!trimmed) {
    return {
      ok: false,
      error: 'Geography file is empty. Expected a .geojson FeatureCollection or a 32×36 "land_sea_mask" JSON.',
    }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch (err) {
    return {
      ok: false,
      error: `Invalid geography JSON syntax: ${err instanceof Error ? err.message : String(err)}`,
    }
  }

  // 1. Standard GeoJSON (FeatureCollection / Feature / Polygon / MultiPolygon / LineString)
  const polygons: number[][][][] = []
  const lines: number[][][] = []
  if (collectGeoJsonGeometries(parsed, polygons, lines) && (polygons.length > 0 || lines.length > 0)) {
    const coastlineSegments: ContourSegment[] = []
    for (const line of lines) {
      for (let i = 0; i < line.length - 1; i++) {
        const p1 = line[i]
        const p2 = line[i + 1]
        if (
          !Array.isArray(p1) ||
          !Array.isArray(p2) ||
          typeof p1[0] !== 'number' ||
          typeof p1[1] !== 'number' ||
          typeof p2[0] !== 'number' ||
          typeof p2[1] !== 'number'
        ) {
          continue
        }
        const [x1, y1] = lonLatToGridXY(p1[0], p1[1])
        const [x2, y2] = lonLatToGridXY(p2[0], p2[1])
        coastlineSegments.push([x1, y1, x2, y2])
      }
    }

    const rawMask =
      typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>).land_sea_mask
        : undefined

    let mask: Grid2D | null = null
    if (Array.isArray(rawMask) && rawMask.length === GRID_ROWS) {
      const candidate: Grid2D = []
      let valid = true
      for (let r = 0; r < GRID_ROWS; r++) {
        const row = rawMask[r]
        if (!Array.isArray(row) || row.length !== GRID_COLS) {
          valid = false
          break
        }
        candidate.push(
          row.map((cell) =>
            typeof cell === 'boolean'
              ? cell
                ? 1
                : 0
              : typeof cell === 'number' && Number.isFinite(cell)
                ? Math.max(0, Math.min(1, cell))
                : 0,
          ),
        )
      }
      if (valid) mask = candidate
    }

    if (!mask) {
      // Rasterize polygons onto 32x36 grid with 2x2 sub-cell sampling for accurate land/sea fractions
      mask = Array.from({ length: GRID_ROWS }, (_, r) => {
        const rowLatCenter = LAT_ORIGIN + r
        return Array.from({ length: GRID_COLS }, (__, c) => {
          if (polygons.length === 0) return 1
          const colLonCenter = LON_ORIGIN + c
          const offsets = [-0.25, 0.25]
          let hits = 0
          for (const dLat of offsets) {
            for (const dLon of offsets) {
              const lat = rowLatCenter + dLat
              const lon = colLonCenter + dLon
              for (let p = 0; p < polygons.length; p++) {
                if (pointInPolygonRings(lon, lat, polygons[p])) {
                  hits += 1
                  break
                }
              }
            }
          }
          return hits / 4
        })
      })
    }

    return {
      ok: true,
      data: {
        landSeaMask: mask,
        coastlineSegments,
        sourceLabel,
      },
    }
  }

  // 2. Legacy 32x36 land_sea_mask JSON
  const rawMask =
    typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>).land_sea_mask
      : undefined

  if (!Array.isArray(rawMask) || rawMask.length !== GRID_ROWS) {
    return {
      ok: false,
      error: `Invalid geography file: expected a GeoJSON FeatureCollection or { "land_sea_mask": number[${GRID_ROWS}][${GRID_COLS}] }.`,
    }
  }

  const mask: Grid2D = new Array(GRID_ROWS)
  for (let r = 0; r < GRID_ROWS; r++) {
    const row = rawMask[r]
    if (!Array.isArray(row) || row.length !== GRID_COLS) {
      return {
        ok: false,
        error: `Invalid land_sea_mask[${r}]: expected ${GRID_COLS} columns, got ${Array.isArray(row) ? row.length : typeof row
          }.`,
      }
    }
    const cleanRow = new Array<number>(GRID_COLS)
    for (let c = 0; c < GRID_COLS; c++) {
      const cell = row[c]
      if (typeof cell === 'boolean') {
        cleanRow[c] = cell ? 1 : 0
      } else if (typeof cell === 'number' && Number.isFinite(cell)) {
        cleanRow[c] = Math.max(0, Math.min(1, cell))
      } else {
        return {
          ok: false,
          error: `Invalid value at land_sea_mask[${r}][${c}]: expected finite number or boolean.`,
        }
      }
    }
    mask[r] = cleanRow
  }

  // Compute coastline outlines via marching squares at the 0.5 land/sea threshold
  const nx = GRID_COLS * 2
  const ny = GRID_ROWS * 2
  const vertexField = buildNorthUpScalarVertexField(mask, nx, ny, false)
  const coastlineSegments = extractIsoContourSegments(vertexField, nx, ny, 0.5)

  return {
    ok: true,
    data: {
      landSeaMask: mask,
      coastlineSegments,
      sourceLabel,
    },
  }
}
