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
- `src/App.tsx` — Map-first dashboard orchestrator with a full-height ~70% single map stage + sticky bottom timeline and ~30% right instrument rail above the fold, and the Spread Chart + Impact Panel below the fold. Polls `/health` every 10s, coordinates `/simulate` calls, and drives instant scrubbing and keyboard shortcuts (`Space` = play/pause, `←`/`→` = step).
- `src/index.css` — Tailwind CSS v4 import, locally bundled `Instrument Serif`, `Geist Sans`, and `Geist Mono` `@font-face` rules, and "Petrichor & Lightning" theme tokens (`#0D1210` background, `#141B18` surface, `#26312B` hairlines, `#ECE7DA` bone text, `#F0B84B` lightning, `#7FB5B0` rain, `#D9593D` ember).
- `src/vite-env.d.ts` — Vite client type reference plus typed `ImportMetaEnv` for `VITE_API_BASE`.

### API Layer (`src/api/`)
- `src/api/types.ts` — Shared TypeScript interfaces and grid/channel constants (`HealthResponse`, `SimulateRequest`, `SimulateResponse`, `ScenarioData`, `State3D`, `StateSequence`, `DisplayChannelId`, etc.) used across the app.
- `src/api/client.ts` — Single HTTP client implementing `getApiBase`, `ApiError`, `checkHealth` (`POST /health`), and `runSimulation` (`POST /simulate`) with timeout handling and strict rollout shape validation.

### Computation Libraries (`src/lib/`)
- `src/lib/constants.ts` — Grid coordinate helpers (`rowToLat`, `colToLon`, `latToRow`, `lonToCol`), wire channel metadata, physical-to-display unit conversion (`t2m` K → °C, `msl` Pa → hPa), static on-the-fly **Wind Speed** (`computeWindSpeedGrid`, `Math.sqrt(u10 ** 2 + v10 ** 2)`), and Branch − Prime difference grid helper.
- `src/lib/validation.ts` — Strict JSON parser and shape validator (`parseAndValidateScenario`, `validateScenarioObject`) verifying `init_states` (`[3][6][32][36]` finite numbers) and `doy_sincos` (`[sin, cos]`).
- `src/lib/colormaps.ts` — Smooth sequential palettes (temperature, rain, Deep Navy-to-Cyan/White `wind` speed, pressure, water vapor, heatwave), zero-centered bone (`#ECE7DA`) diverging palette for Difference maps, prebuilt 1025-entry RGB lookup tables (`writePaletteColor`), CSS gradient generation, and scale computation.
- `src/lib/heatmap.ts` — Rollout-only precomputation (`precomputeRolloutWorkspace` for all display channel grids, difference grids, rollout-consistent scales, per-step 5%-of-peak `|Branch − Prime|` response front contours, dashed 3-cell locality ring, dashed `8 cells/day` reference rings, wind vector arrows, and `tp > 10 mm/day` contours), offscreen canvas rasterizer (`renderHeatmapToBuffer`), and high-DPI canvas graticule/shock/ripple renderer (`drawCanvasMapOverlays`).
- `src/lib/geography.ts` — In-memory validator and marching-squares coastline extractor (`parseAndValidateGeographyJson`) for user-imported `32×36` `land_sea_mask` JSON files (zero bundled coastline or GeoJSON data).
- `src/lib/vorticity.ts` — Browser-computed spherical relative vorticity (`computeRelativeVorticityGrid`, $\zeta = \nabla \times (u_{10}, v_{10})$ in $10^{-5}\text{ s}^{-1}$) derived strictly from the rollout's `u10` and `v10` fields.
- `src/lib/windFlow.ts` — Bilinear `(u10, v10)` wind sampler, capped 2D particle-advection helpers, `prefers-reduced-motion` static streamline renderer, 3D streamline tracer, and `(Branch − Prime)` difference wind vector builder.
- `src/lib/shock.ts` — Shock configuration model, named presets (`Warm Bay of Bengal`, `Weaker Arabian Sea wind`), box and Gaussian spatial weight masks, physical `[6][32][36]` delta builder (`buildPerturbation`, computing `%` relative to `init_states[2]` per cell), `buildZeroPerturbation`, active shock mask extraction, and map click/drag geometry helpers.
- `src/lib/spread.ts` — Euclidean distance-in-cells calculation from the shock footprint, response magnitude (`|Branch − Prime|`) binned by distance across rollout steps, day-by-day propagation front calculation, and reference constants (`3 cells` day-1 locality, `8 cells/day` front speed).
- `src/lib/impact.ts` — Agronomic & Human Risk Decision Support System computations: **Kharif Crop Dry-Spell Index** (`computeKharifDrySpellImpact`, `compareKharifDrySpellImpact` with `tp < 2.5 mm/day` for `≥ 5` consecutive days) and scrubber-synced **Lethal Wet-Bulb Index** (`computeWetBulbStepImpact`, `compareWetBulbStepImpact` computing `Wet_Bulb_Eq = t2m + (tcwv * 0.15)` and flagging `Wet_Bulb_Eq > 310.0 K` as **Severe Human Danger**, with `1 cell ≈ 10,000 sq km` area summaries).

### UI Components (`src/components/`)
- `src/components/StatusBar.tsx` — Thin 56px hairline top bar with serif wordmark left and plain-text telemetry right (`Server ● ready · model · ms · step k/N`).
- `src/components/ErrorBanner.tsx` — Flat full-width `#D9593D` ember error strip with plain text and `Retry` / `Dismiss` text actions.
- `src/components/AsciiRainscapeBg.tsx` — Landing-page retro ASCII weather scene on a single HTML5 canvas 2D context over `#020617`: a static cumulonimbus canopy across the top 30% (`@ # % & W`) and static uneven terrain across the bottom 15% (`_ - / \ ^`), both in dim slate `#475569` and pre-rendered once per resize to an offscreen layer, plus rain columns (`| ' : .`) in low-opacity cyan `#22d3ee` falling from the cloud underside to the ground. The `requestAnimationFrame` loop is hard-capped at 15 FPS and pauses while the tab is hidden; `prefers-reduced-motion` or `navigator.webdriver` paints one still frame with no loop. The wrapper uses `z-index: -1` and `pointer-events: none`, and the component returns `null` (cancelling the loop) when import is clicked. Decorative only, not weather data.
- `src/components/EmptyState.tsx` — Full-bleed editorial landing view ("Load a starting atmosphere.") that unmounts `AsciiRainscapeBg` immediately when "Import scenario file" is clicked or a scenario loads.
- `src/components/MapControls.tsx` — Thin film-editor timeline scrubber with per-step tick marks, instant pointer-drag playhead, play/pause, and speed selector (`0.5×`–`4×`), plus the slim right-rail `ChannelSelector` for base wire channels and static derived channels (**Wind Speed** and **Rotation / relative vorticity**).
- `src/components/GridMap.tsx` — Static north-up (`row 0 = south`) `<canvas>` heatmap renderer with 10× bilinear (`320×360`) scalar upscaling prior to LUT coloring, `Instrument Serif` italic labels, rain/lightning top hairline rules, 1px 5° lat/lon graticules, mono edge coordinates, and exact 1°-cell-snapped mono hover tooltip.
- `src/components/WindFlowCanvas.tsx` — Capped 2D `<canvas>` particle-advection overlay with fading trails colored by wind speed from bilinearly interpolated `(u10, v10)` (falls back to static streamlines under `prefers-reduced-motion`).
- `src/components/RolloutSurface3D.tsx` — Optional 3D smooth `THREE.PlaneGeometry` (32×36) heightfield mesh with normalized channel vertex heights, matching 2D vertex colors, sea-cell darkening and draped 1px bone coastlines when geography is loaded, speed-colored 3D wind flow lines (or `Branch − Prime` wind arrows on Difference), flat `y = 0` sea-level reference plane, ~50° North-up camera tilt, bounded orbit, height exaggeration slider, and raycasted mono hover tooltips.
- `src/components/SplitCompareMap.tsx` — Draggable split-curtain (`Timeline Prime` | `Branch`) canvas comparison view.
- `src/components/ComparisonView.tsx` — Single full-height map stage switching between **Prime**, **Branch**, **Difference** (bone-centered `#ECE7DA` at exact zero), and **Split curtain** (plus `2D / 3D` toggle), showing a calm prompt with inline preset buttons on Branch/Difference before a shock has run.
- `src/components/ShockPanel.tsx` — Right-rail instrument panel for presets, box/Gaussian footprint placement with live 32×36 preview, channel amplitudes, physical delta readout, and **Run** / **Zero shock** controls.
- `src/components/SpreadChart.tsx` — Below-the-fold Day-1 60%-marked Locality gauge, auto-fitting Y-axis front-distance chart with direct labels and mono `8 cells/day` dashed reference, and Difference-map front ripple ring (dims with a one-line hint until a shock runs).
- `src/components/ImpactPanel.tsx` — Below-the-fold Agronomic & Human Risk Decision Support System featuring the primary **Kharif Crop Dry-Spell Index** (`sq km` Severe Crop Stress headline, Prime vs Branch comparison, and static dry-spell heatmaps) and the secondary scrubber-synced **Lethal Wet-Bulb Index** (`Wet_Bulb_Eq = t2m + (tcwv * 0.15)`, flagging `> 310.0 K` as **Severe Human Danger** with total area and cell counts plus static per-step heatmaps).

### Dev-Only Preview (`src/dev/`)
- `src/dev/syntheticScenario.ts` — Runtime procedural generator (gated behind `?dev=1`) producing a clearly labelled `SYNTHETIC - DEV ONLY` `[3][6][32][36]` scenario and fake rollout so the workspace can be previewed without a server; activates a red `SYNTHETIC - DEV ONLY` badge while in use.
