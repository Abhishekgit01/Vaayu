import { useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import type { ScenarioData } from '../api/types'
import { parseAndValidateScenario } from '../lib/validation'
import { TopographicLatticeBg } from './TopographicLatticeBg'

interface EmptyStateProps {
  steps: number
  onStepsChange: (steps: number) => void
  onImport: (scenario: ScenarioData) => void
  showDevPreview?: boolean
  onLoadSyntheticPreview?: () => void
}

export function EmptyState({
  steps,
  onStepsChange,
  onImport,
  showDevPreview = false,
  onLoadSyntheticPreview,
}: EmptyStateProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [showPasteBox, setShowPasteBox] = useState(false)
  const [pastedJson, setPastedJson] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isReadingFile, setIsReadingFile] = useState(false)

  const processRawText = (text: string, sourceLabel: string) => {
    const result = parseAndValidateScenario(text, sourceLabel)
    if (!result.ok) {
      setValidationError(result.error)
      return
    }
    setValidationError(null)
    onImport(result.data)
  }

  const handleFile = async (file: File) => {
    setIsReadingFile(true)
    setValidationError(null)
    try {
      const text = await file.text()
      processRawText(text, file.name)
    } catch (err) {
      setValidationError(
        `Could not read file "${file.name}": ${err instanceof Error ? err.message : String(err)}`,
      )
    } finally {
      setIsReadingFile(false)
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      void handleFile(file)
    }
    e.target.value = ''
  }

  const handleDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      void handleFile(file)
    }
  }

  const handlePasteSubmit = () => {
    processRawText(pastedJson, 'Pasted JSON')
  }

  return (
    <main
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`relative flex min-h-[calc(100vh-56px)] w-full flex-1 items-center overflow-hidden bg-[#0D1210] ${
        isDragging ? 'outline-1 -outline-offset-8 outline-dashed outline-[#7FB5B0]' : ''
      }`}
    >
      {/* Full-bleed decorative Three.js 32x36 topographic lattice */}
      <TopographicLatticeBg />

      {/* Subtle vignette keeping the left editorial column crystal clear */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'linear-gradient(90deg, rgba(13, 18, 16, 0.92) 0%, rgba(13, 18, 16, 0.72) 48%, rgba(13, 18, 16, 0.18) 100%)',
        }}
      />

      <div className="panel-enter relative z-10 mx-auto w-full max-w-[1680px] px-6 py-16 sm:px-12">
        <div className="max-w-xl">
          <h2 className="font-serif-display text-5xl leading-[1.02] tracking-tight text-[#ECE7DA] sm:text-7xl">
            Load a starting atmosphere.
          </h2>

          <p className="mt-4 max-w-md text-[15px] leading-relaxed text-[#8C958D]">
            Provide three consecutive days of physical fields (<code className="font-mono text-xs text-[#ECE7DA]">[3][6][32][36]</code>) and{' '}
            <code className="font-mono text-xs text-[#ECE7DA]">doy_sincos</code> over 6.5°–37.5°N, 65.5°–100.5°E.
          </p>

          {/* Actions */}
          <div className="mt-8 flex flex-wrap items-center gap-6">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              type="button"
              disabled={isReadingFile}
              onClick={() => fileInputRef.current?.click()}
              className="ui-transition rounded-[3px] border border-[#F0B84B] bg-[#F0B84B] px-5 py-2.5 text-sm font-medium text-[#0D1210] hover:bg-[#f4c668] disabled:opacity-50 cursor-pointer"
            >
              {isReadingFile ? 'Reading scenario…' : 'Import scenario file'}
            </button>

            <button
              type="button"
              onClick={() => setShowPasteBox((v) => !v)}
              aria-expanded={showPasteBox}
              className="ui-transition text-sm text-[#7FB5B0] underline-offset-4 hover:underline cursor-pointer"
            >
              {showPasteBox ? 'Hide JSON input' : 'Paste JSON'}
            </button>

            <label className="inline-flex items-center gap-2 border-l border-[#26312B] pl-5 text-xs text-[#8C958D]">
              <span>Horizon</span>
              <input
                type="number"
                min={1}
                max={60}
                value={steps}
                onChange={(e) => {
                  const val = Number.parseInt(e.target.value, 10)
                  if (!Number.isNaN(val)) onStepsChange(Math.max(1, Math.min(60, val)))
                }}
                className="tabular-nums w-12 rounded-[2px] border border-[#26312B] bg-[#141B18] px-1.5 py-1 text-center font-mono text-xs text-[#ECE7DA] focus:border-[#7FB5B0] focus:outline-none"
              />
              <span className="font-mono">days</span>
            </label>

            {showDevPreview && onLoadSyntheticPreview && (
              <button
                type="button"
                onClick={onLoadSyntheticPreview}
                className="ui-transition rounded-[2px] border border-[#D9593D] bg-[#141B18] px-2.5 py-1 font-mono text-xs text-[#D9593D] hover:bg-[#D9593D] hover:text-[#0D1210] cursor-pointer"
              >
                Load synthetic preview
              </button>
            )}
          </div>

          {/* Expandable mono JSON paste area */}
          {showPasteBox && (
            <div className="panel-enter mt-6 border-t border-[#26312B] pt-5">
              <label htmlFor="paste-scenario-json" className="block text-xs text-[#8C958D]">
                Scenario JSON (<span className="font-mono">init_states</span> and{' '}
                <span className="font-mono">doy_sincos</span>)
              </label>
              <textarea
                id="paste-scenario-json"
                value={pastedJson}
                onChange={(e) => setPastedJson(e.target.value)}
                spellCheck={false}
                rows={6}
                placeholder='{"init_states": [[[[...]]]], "doy_sincos": [0.0, 1.0]}'
                className="mt-2 w-full rounded-[3px] border border-[#26312B] bg-[#141B18] p-3 font-mono text-xs text-[#ECE7DA] placeholder:text-[#5A645D] focus:border-[#7FB5B0] focus:outline-none"
              />
              <div className="mt-3 flex items-center justify-between gap-4">
                <span className="font-mono text-xs text-[#5A645D] tabular-nums">
                  {pastedJson.length > 0
                    ? `${(pastedJson.length / 1024).toFixed(1)} KB`
                    : 't2m (K), tp (mm/day), u10, v10 (m/s), msl (Pa), tcwv (kg/m²)'}
                </span>
                <button
                  type="button"
                  onClick={handlePasteSubmit}
                  disabled={!pastedJson.trim()}
                  className="ui-transition rounded-[3px] border border-[#26312B] bg-[#1A231F] px-4 py-1.5 text-xs font-medium text-[#ECE7DA] hover:border-[#F0B84B] hover:text-[#F0B84B] disabled:opacity-40 cursor-pointer"
                >
                  Validate &amp; load
                </button>
              </div>
            </div>
          )}

          {/* Validation error */}
          {validationError && (
            <div
              role="alert"
              className="mt-6 border-l-2 border-[#D9593D] bg-[#141B18] px-4 py-3 text-xs text-[#ECE7DA]"
            >
              <p className="font-medium text-[#D9593D]">Invalid scenario payload</p>
              <p className="mt-1 font-mono text-[#ECE7DA]">{validationError}</p>
            </div>
          )}

          {/* Single small muted note on in-memory behavior */}
          <p className="mt-8 text-xs text-[#5A645D]">
            Imported scenarios remain in browser memory for this session and are never saved to disk
            or localStorage. You can also drop a JSON file anywhere on this screen.
          </p>
        </div>
      </div>
    </main>
  )
}
