import { GRID_COLS, GRID_ROWS, type Grid2D } from '../api/types'
import { buildNorthUpScalarVertexField, type ContourSegment } from './heatmap'
import { extractIsoContourSegments } from './shock'

/**
 * In-memory geography overlay built from a user-imported 32x36 land_sea_mask JSON.
 * Never saved to localStorage or disk; no geographic data is bundled with the app.
 */
export interface GeographyOverlayData {
  /** [32][36] land fraction in [0..1] (row 0 = south 6.5°N, col 0 = west 65.5°E; >= 0.5 = land) */
  landSeaMask: Grid2D
  /** Marching-squares coastline segments at mask = 0.5 in [0..36] x [0..32] north-up coords */
  coastlineSegments: ContourSegment[]
  sourceLabel: string
}

export type GeographyValidationResult =
  | { ok: true; data: GeographyOverlayData }
  | { ok: false; error: string }

export function parseAndValidateGeographyJson(
  rawText: string,
  sourceLabel = 'geography.json',
): GeographyValidationResult {
  const trimmed = rawText.trim()
  if (!trimmed) {
    return {
      ok: false,
      error: 'Geography file is empty. Expected JSON with a 32×36 "land_sea_mask" array.',
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

  const rawMask =
    typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>).land_sea_mask
      : undefined

  if (!Array.isArray(rawMask) || rawMask.length !== GRID_ROWS) {
    return {
      ok: false,
      error: `Invalid "land_sea_mask": expected an object { "land_sea_mask": number[${GRID_ROWS}][${GRID_COLS}] }, got ${
        Array.isArray(rawMask) ? `${rawMask.length} rows` : typeof rawMask
      }.`,
    }
  }

  const mask: Grid2D = new Array(GRID_ROWS)
  for (let r = 0; r < GRID_ROWS; r++) {
    const row = rawMask[r]
    if (!Array.isArray(row) || row.length !== GRID_COLS) {
      return {
        ok: false,
        error: `Invalid land_sea_mask[${r}]: expected ${GRID_COLS} columns, got ${
          Array.isArray(row) ? row.length : typeof row
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
