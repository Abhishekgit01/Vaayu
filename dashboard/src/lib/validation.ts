import {
  GRID_COLS,
  GRID_ROWS,
  INIT_DAYS,
  NUM_CHANNELS,
  type DoySinCos,
  type ScenarioData,
  type StateSequence,
} from '../api/types'

export type ValidationResult =
  | { ok: true; data: ScenarioData }
  | { ok: false; error: string }

export function parseAndValidateScenario(
  rawText: string,
  sourceLabel?: string,
): ValidationResult {
  const trimmed = rawText.trim()
  if (!trimmed) {
    return { ok: false, error: 'Input is empty. Provide a JSON object with init_states and doy_sincos.' }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    return { ok: false, error: `Invalid JSON syntax: ${msg}` }
  }

  return validateScenarioObject(parsed, sourceLabel)
}

export function validateScenarioObject(
  input: unknown,
  sourceLabel?: string,
): ValidationResult {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return {
      ok: false,
      error: 'Root JSON value must be an object containing "init_states" and "doy_sincos".',
    }
  }

  const obj = input as Record<string, unknown>

  // Validate doy_sincos: [sin, cos]
  const doy = obj.doy_sincos
  if (!Array.isArray(doy) || doy.length !== 2) {
    return {
      ok: false,
      error: `Invalid "doy_sincos": expected a 2-element array [sin, cos], got ${
        Array.isArray(doy) ? `length ${doy.length}` : typeof doy
      }.`,
    }
  }
  if (
    typeof doy[0] !== 'number' ||
    !Number.isFinite(doy[0]) ||
    typeof doy[1] !== 'number' ||
    !Number.isFinite(doy[1])
  ) {
    return {
      ok: false,
      error: 'Invalid "doy_sincos": both sin and cos values must be finite numbers.',
    }
  }
  if (Math.abs(doy[0]) > 1.01 || Math.abs(doy[1]) > 1.01) {
    return {
      ok: false,
      error: `Invalid "doy_sincos": values [${doy[0]}, ${doy[1]}] are outside [-1, 1].`,
    }
  }

  // Validate init_states: [3][6][32][36]
  const states = obj.init_states
  if (!Array.isArray(states) || states.length !== INIT_DAYS) {
    return {
      ok: false,
      error: `Invalid "init_states": expected ${INIT_DAYS} consecutive days (shape [3][6][32][36]), got ${
        Array.isArray(states) ? `length ${states.length}` : typeof states
      }.`,
    }
  }

  for (let d = 0; d < INIT_DAYS; d++) {
    const dayState = states[d]
    if (!Array.isArray(dayState) || dayState.length !== NUM_CHANNELS) {
      return {
        ok: false,
        error: `Invalid init_states[${d}]: expected ${NUM_CHANNELS} channels, got ${
          Array.isArray(dayState) ? `length ${dayState.length}` : typeof dayState
        }.`,
      }
    }
    for (let c = 0; c < NUM_CHANNELS; c++) {
      const grid = dayState[c]
      if (!Array.isArray(grid) || grid.length !== GRID_ROWS) {
        return {
          ok: false,
          error: `Invalid init_states[${d}][${c}]: expected ${GRID_ROWS} rows, got ${
            Array.isArray(grid) ? `length ${grid.length}` : typeof grid
          }.`,
        }
      }
      for (let r = 0; r < GRID_ROWS; r++) {
        const row = grid[r]
        if (!Array.isArray(row) || row.length !== GRID_COLS) {
          return {
            ok: false,
            error: `Invalid init_states[${d}][${c}][${r}]: expected ${GRID_COLS} columns, got ${
              Array.isArray(row) ? `length ${row.length}` : typeof row
            }.`,
          }
        }
        for (let col = 0; col < GRID_COLS; col++) {
          const val = row[col]
          if (typeof val !== 'number' || !Number.isFinite(val)) {
            return {
              ok: false,
              error: `Non-finite or non-numeric value at init_states[${d}][${c}][${r}][${col}].`,
            }
          }
        }
      }
    }
  }

  return {
    ok: true,
    data: {
      init_states: states as StateSequence,
      doy_sincos: [doy[0], doy[1]] as DoySinCos,
      sourceLabel,
    },
  }
}
