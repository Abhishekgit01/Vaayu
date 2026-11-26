import {
  GRID_COLS,
  GRID_ROWS,
  NUM_CHANNELS,
  type HealthResponse,
  type SimulateRequest,
  type SimulateResponse,
  type StateSequence,
} from './types'

const DEFAULT_API_BASE = 'http://localhost:8000'
const HEALTH_TIMEOUT_MS = 8_000
const SIMULATE_TIMEOUT_MS = 120_000

export function getApiBase(): string {
  const raw = import.meta.env.VITE_API_BASE
  const base = typeof raw === 'string' && raw.trim().length > 0 ? raw.trim() : DEFAULT_API_BASE
  return base.replace(/\/+$/, '')
}

export class ApiError extends Error {
  readonly status: number | null
  readonly endpoint: string

  constructor(message: string, endpoint: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.endpoint = endpoint
  }
}

async function postJson<T>(
  endpoint: string,
  body: unknown | undefined,
  timeoutMs: number,
  externalSignal?: AbortSignal,
): Promise<T> {
  const url = `${getApiBase()}${endpoint}`
  const controller = new AbortController()
  let timedOut = false

  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const onExternalAbort = () => controller.abort()
  if (externalSignal) {
    if (externalSignal.aborted) {
      window.clearTimeout(timer)
      throw new ApiError('Request cancelled', endpoint, null)
    }
    externalSignal.addEventListener('abort', onExternalAbort, { once: true })
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })

    if (response.status !== 200) {
      let detail = ''
      try {
        const text = await response.text()
        detail = text ? `: ${text.slice(0, 240)}` : ''
      } catch {
        // ignore body read errors
      }
      throw new ApiError(
        `HTTP ${response.status} from ${endpoint}${detail}`,
        endpoint,
        response.status,
      )
    }

    let parsed: unknown
    try {
      parsed = await response.json()
    } catch {
      throw new ApiError(`Non-JSON response from ${endpoint}`, endpoint, response.status)
    }

    return parsed as T
  } catch (err) {
    if (err instanceof ApiError) {
      throw err
    }
    if (timedOut) {
      throw new ApiError(
        `Request to ${endpoint} timed out after ${Math.round(timeoutMs / 1000)}s`,
        endpoint,
        null,
      )
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('Request cancelled', endpoint, null)
    }
    const message = err instanceof Error ? err.message : String(err)
    throw new ApiError(`Failed to reach ${url}: ${message}`, endpoint, null)
  } finally {
    window.clearTimeout(timer)
    externalSignal?.removeEventListener('abort', onExternalAbort)
  }
}

function validateRolloutShape(rollout: unknown, expectedSteps: number): StateSequence {
  if (!Array.isArray(rollout) || rollout.length !== expectedSteps) {
    throw new ApiError(
      `Invalid rollout shape: expected ${expectedSteps} steps, got ${Array.isArray(rollout) ? rollout.length : typeof rollout}`,
      '/simulate',
      200,
    )
  }
  for (let t = 0; t < expectedSteps; t++) {
    const state = rollout[t]
    if (!Array.isArray(state) || state.length !== NUM_CHANNELS) {
      throw new ApiError(
        `Invalid rollout[${t}] channels: expected ${NUM_CHANNELS}`,
        '/simulate',
        200,
      )
    }
    for (let c = 0; c < NUM_CHANNELS; c++) {
      const grid = state[c]
      if (!Array.isArray(grid) || grid.length !== GRID_ROWS) {
        throw new ApiError(
          `Invalid rollout[${t}][${c}] rows: expected ${GRID_ROWS}`,
          '/simulate',
          200,
        )
      }
      for (let r = 0; r < GRID_ROWS; r++) {
        const row = grid[r]
        if (!Array.isArray(row) || row.length !== GRID_COLS) {
          throw new ApiError(
            `Invalid rollout[${t}][${c}][${r}] cols: expected ${GRID_COLS}`,
            '/simulate',
            200,
          )
        }
        for (let col = 0; col < GRID_COLS; col++) {
          if (typeof row[col] !== 'number' || !Number.isFinite(row[col])) {
            throw new ApiError(
              `Non-finite value at rollout[${t}][${c}][${r}][${col}]`,
              '/simulate',
              200,
            )
          }
        }
      }
    }
  }
  return rollout as StateSequence
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const data = await postJson<unknown>('/health', undefined, HEALTH_TIMEOUT_MS, signal)
  if (
    typeof data !== 'object' ||
    data === null ||
    typeof (data as Record<string, unknown>).status !== 'string' ||
    typeof (data as Record<string, unknown>).model_name !== 'string'
  ) {
    throw new ApiError(
      'Invalid /health response: expected { status: string, model_name: string }',
      '/health',
      200,
    )
  }
  const res = data as HealthResponse
  if (res.status !== 'ready') {
    throw new ApiError(`Server health status is "${res.status}" (expected "ready")`, '/health', 200)
  }
  return res
}

export async function runSimulation(
  request: SimulateRequest,
  signal?: AbortSignal,
): Promise<SimulateResponse> {
  const data = await postJson<unknown>('/simulate', request, SIMULATE_TIMEOUT_MS, signal)
  if (typeof data !== 'object' || data === null) {
    throw new ApiError('Invalid /simulate response: expected JSON object', '/simulate', 200)
  }
  const obj = data as Record<string, unknown>
  if (typeof obj.inference_ms !== 'number' || !Number.isFinite(obj.inference_ms)) {
    throw new ApiError(
      'Invalid /simulate response: missing finite inference_ms',
      '/simulate',
      200,
    )
  }
  const rollout = validateRolloutShape(obj.rollout, request.steps)
  return {
    rollout,
    inference_ms: obj.inference_ms,
  }
}
