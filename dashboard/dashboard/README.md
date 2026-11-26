# Monsoon Twin — Frontend Dashboard Handoff

Frontend-only React + TypeScript + Tailwind CSS dashboard for **Monsoon Twin**, an India climate digital twin operating on a 32×36 1-degree grid (`lat 6.5°N–37.5°N`, `lon 65.5°E–100.5°E`).

All fonts (`@fontsource/instrument-serif`, `@fontsource/geist-sans`, `@fontsource/geist-mono`), 3D rendering (`three`), and icons (`lucide-react`) are bundled locally via npm so the app works with no internet connection. No weather values, sample scenarios, or mock fallbacks exist anywhere in the codebase; imported scenario JSON is kept strictly in memory.

---

## Environment Variable

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE` | `http://localhost:8000` | Base URL for the live climate model HTTP API. Trailing slashes are trimmed automatically. |

---

## API Calls (`src/api/client.ts`)

`src/api/client.ts` is the sole HTTP client in the project. Every non-200 response, non-JSON response, invalid tensor shape, or timeout throws an `ApiError` and triggers the red error banner.

1. **`POST /health`** (`checkHealth`)
   - Called on initial page load and polled every 10 seconds, plus manual retry.
   - Expected response: `{ "status": "ready", "model_name": string }`.
2. **`POST /simulate`** (`runSimulation`)
   - Called on scenario import (for **Timeline Prime** with `perturbation: null`) and when running **Branch** (with a `[6][32][36]` physical delta array) or **Zero shock** (with an explicit all-zero `[6][32][36]` array).
   - Request body: `{ init_states: number[3][6][32][36], doy_sincos: [number, number], steps: number, perturbation: number[6][32][36] | null }`.
   - Expected response: `{ rollout: number[steps][6][32][36], inference_ms: number }`.
   - Wire units (axis 1): `0: t2m (K)`, `1: tp (mm/day)`, `2: u10 (m/s)`, `3: v10 (m/s)`, `4: msl (Pa)`, `5: tcwv (kg/m²)`. Display conversion happens strictly in the UI (`t2m` → `°C`, `msl` → `hPa`).

---

## File & Component Directory

### Entrypoints & Styles
- `src/main.tsx` — Sets the document title and mounts `<App />` into `#root`.
- `src/App.tsx` — Map-first (~70% map stage / ~30% right instrument rail) dashboard orchestrator: polls `/health` every 10s, manages in-memory scenario/rollout state, coordinates `/simulate` calls for Timeline Prime, Branch, and Zero shock, drives the shared film-editor timeline scrubber, playback speed (`0.5×`–`4×`), and keyboard shortcuts (`Space` = play/pause, `←`/`→` = step).
- `src/index.css` — Tailwind CSS v4 import, locally bundled `Instrument Serif`, `Geist Sans`, and `Geist Mono` `@font-face` rules, and "Petrichor & Lightning" theme tokens (`#0D1210` background, `#141B18` surface, `#26312B` hairlines, `#ECE7DA` bone text, `#F0B84B` lightning, `#7FB5B0` rain, `#D9593D` ember).
- `src/vite-env.d.ts` — Vite client type reference plus typed `ImportMetaEnv` for `VITE_API_BASE`.

### API Layer (`src/api/`)
- `src/api/types.ts` — Shared TypeScript interfaces and grid/channel constants (`HealthResponse`, `SimulateRequest`, `SimulateResponse`, `ScenarioData`, `State3D`, `StateSequence`, `DisplayChannelId`, etc.) used across the app.
- `src/api/client.ts` — Single HTTP client implementing `getApiBase`, `ApiError`, `checkHealth` (`POST /health`), and `runSimulation` (`POST /simulate`) with timeout handling and strict rollout shape validation.

### Computation Libraries (`src/lib/`)
- `src/lib/constants.ts` — Grid coordinate helpers (`rowToLat`, `colToLon`, `latToRow`, `lonToCol`), channel metadata, physical-to-display unit conversion (`t2m` K → °C, `msl` Pa → hPa), wind speed extraction (`√(u10² + v10²)`), and Branch − Prime difference grid helper.
- `src/lib/validation.ts` — Strict JSON parser and shape validator (`parseAndValidateScenario`, `validateScenarioObject`) verifying `init_states` (`[3][6][32][36]` finite numbers) and `doy_sincos` (`[sin, cos]`).
- `src/lib/colormaps.ts` — Smooth sequential palettes (temperature, rain, wind, pressure, water vapor, heatwave), zero-centered diverging palette for Difference maps, prebuilt 1025-entry RGB lookup tables (`writePaletteColor`), CSS gradient generation, and scale computation.
- `src/lib/heatmap.ts` — Rollout-only precomputation (`precomputeRolloutWorkspace` for all display channel grids, difference grids, rollout-consistent scales, wind vector arrows, and `tp > 10 mm/day` contours), offscreen canvas rasterizer supporting both crisp cells and scalar-space bilinear smooth interpolation (`renderHeatmapToBuffer`), and high-DPI canvas 5° lat/lon graticule + overlay renderer (`drawCanvasMapOverlays`).
- `src/lib/shock.ts` — Shock configuration model, named presets (`Warm Bay of Bengal`, `Weaker Arabian Sea wind`), box and Gaussian spatial weight masks, physical `[6][32][36]` delta builder (`buildPerturbation`, computing `%` relative to `init_states[2]` per cell), `buildZeroPerturbation`, active shock mask extraction, and map click/drag geometry helpers.
- `src/lib/spread.ts` — Euclidean distance-in-cells calculation from the shock footprint, response magnitude (`|Branch − Prime|`) binned by distance across rollout steps, day-by-day propagation front calculation, and reference constants (`3 cells` day-1 locality, `8 cells/day` front speed).
- `src/lib/impact.ts` — Exported `ImpactSettings` / `DEFAULT_IMPACT_SETTINGS` object (`{ heatwaveThresholdC: 35 }`), per-cell heatwave day counting (`t2m > threshold`), and Prime vs Branch headline comparison metrics.

### UI Components (`src/components/`)
- `src/components/StatusBar.tsx` — Thin 56px hairline top bar with serif wordmark left and plain-text telemetry right (`Server ● ready · model · ms · step k/N`).
- `src/components/ErrorBanner.tsx` — Flat full-width `#D9593D` ember error strip with plain text and `Retry` / `Dismiss` text actions.
- `src/components/TopographicLatticeBg.tsx` — Purely decorative Three.js 32×36 mathematical sine-wave lattice in `#26312B` hairline and `#7FB5B0` rain points for the empty state (respects `prefers-reduced-motion`, pauses when hidden, caps `devicePixelRatio` at 2).
- `src/components/EmptyState.tsx` — Full-bleed editorial landing view ("Load a starting atmosphere.") with file import, expandable mono JSON paste box, shape validation feedback, and rollout horizon input.
- `src/components/MapControls.tsx` — Thin film-editor timeline scrubber with per-step tick marks, instant pointer-drag playhead, play/pause, and speed selector (`0.5×`–`4×`), plus the slim right-rail `ChannelSelector`.
- `src/components/GridMap.tsx` — Crisp north-up (`row 0 = south`) `<canvas>` heatmap renderer with `Instrument Serif` italic labels, rain/lightning top hairline rules, 1px 5° lat/lon graticules, mono edge coordinates, and mono hover tooltip.
- `src/components/RolloutSurface3D.tsx` — Optional 3D `THREE.InstancedMesh` extruded 32×36 rollout renderer tilted ~35° North-up with bounded orbit, matching 2D color scales, and raycasted mono hover tooltips.
- `src/components/SplitCompareMap.tsx` — Draggable split-curtain (`Timeline Prime` | `Branch`) canvas comparison view.
- `src/components/ComparisonView.tsx` — Map stage orchestrating the Multiverse layout (*Timeline Prime* and *Branch* side by side with full-width bone-centered *Difference* beneath), split curtain, difference focus, and `2D / 3D` projection toggle.
- `src/components/ShockPanel.tsx` — Right-rail instrument panel for presets, box/Gaussian footprint placement with live 32×36 preview, channel amplitudes, physical delta readout, and **Run** / **Zero shock** controls.
- `src/components/SpreadChart.tsx` — Day-1 60%-marked Locality gauge, thin 1.5px directly-labelled front-distance chart with mono `8 cells/day` dashed reference, and the Difference-map front ripple ring.
- `src/components/ImpactPanel.tsx` — Editorial heatwave risk section with a 96–136px `Instrument Serif` cells-at-risk headline, Prime vs Branch area/cell delta comparison, visible `settings` object, and side-by-side risk maps.

### Dev-Only Preview (`src/dev/`)
- `src/dev/syntheticScenario.ts` — Runtime procedural generator (gated behind `?dev=1`) producing a clearly labelled `SYNTHETIC - DEV ONLY` `[3][6][32][36]` scenario and fake rollout so the workspace can be previewed without a server; activates a red `SYNTHETIC - DEV ONLY` badge while in use.
