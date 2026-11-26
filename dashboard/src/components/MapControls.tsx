import { useRef, type PointerEvent as ReactPointerEvent } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import type { DisplayChannelId } from '../api/types'
import { DISPLAY_CHANNELS } from '../lib/constants'

export type CompareMode = 'side-by-side' | 'split' | 'difference'
export type PlaybackSpeed = 0.5 | 1 | 2 | 4

export const PLAYBACK_SPEEDS: readonly PlaybackSpeed[] = [0.5, 1, 2, 4] as const

interface TimelineScrubberProps {
  stepIndex: number
  totalSteps: number
  onStepIndexChange: (index: number) => void
  isPlaying: boolean
  onTogglePlay: () => void
  playbackSpeed: PlaybackSpeed
  onPlaybackSpeedChange: (speed: PlaybackSpeed) => void
}

/**
 * Thin film-editor timeline scrubber directly beneath the map stage.
 * Instant scrubbing with tick marks per step, play/pause, and speed selector.
 */
export function MapControls({
  stepIndex,
  totalSteps,
  onStepIndexChange,
  isPlaying,
  onTogglePlay,
  playbackSpeed,
  onPlaybackSpeedChange,
}: TimelineScrubberProps) {
  const rulerRef = useRef<HTMLDivElement | null>(null)
  const draggingRef = useRef(false)

  const maxIdx = Math.max(0, totalSteps - 1)
  const canScrub = totalSteps > 0

  const scrubFromClientX = (clientX: number) => {
    const el = rulerRef.current
    if (!el || totalSteps <= 0) return
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0) return
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const nextIdx = Math.round(ratio * maxIdx)
    onStepIndexChange(nextIdx)
  }

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!canScrub) return
    e.currentTarget.setPointerCapture(e.pointerId)
    draggingRef.current = true
    scrubFromClientX(e.clientX)
  }

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return
    scrubFromClientX(e.clientX)
  }

  const handlePointerUp = () => {
    draggingRef.current = false
  }

  const playheadPct = maxIdx === 0 ? 0 : (Math.min(stepIndex, maxIdx) / maxIdx) * 100

  return (
    <section
      aria-label="Forecast timeline"
      className="flex flex-wrap items-center gap-4 border-t border-b border-[#26312B] py-2.5"
    >
      {/* Transport buttons */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          disabled={!canScrub || stepIndex <= 0}
          onClick={() => onStepIndexChange(Math.max(0, stepIndex - 1))}
          title="Step back (←)"
          aria-label="Previous day"
          className="ui-transition flex h-7 w-7 items-center justify-center rounded-[2px] border border-[#26312B] bg-[#141B18] text-[#ECE7DA] hover:border-[#8C958D] disabled:opacity-30 cursor-pointer"
        >
          <ChevronLeft className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>

        <button
          type="button"
          disabled={!canScrub || totalSteps <= 1}
          onClick={onTogglePlay}
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          className="ui-transition inline-flex h-7 items-center gap-1.5 rounded-[2px] border border-[#F0B84B] bg-[#F0B84B] px-3 text-xs font-medium text-[#0D1210] hover:bg-[#f4c668] disabled:opacity-35 cursor-pointer"
        >
          {isPlaying ? (
            <Pause className="h-3 w-3" strokeWidth={1.75} />
          ) : (
            <Play className="h-3 w-3" strokeWidth={1.75} />
          )}
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>

        <button
          type="button"
          disabled={!canScrub || stepIndex >= maxIdx}
          onClick={() => onStepIndexChange(Math.min(maxIdx, stepIndex + 1))}
          title="Step forward (→)"
          aria-label="Next day"
          className="ui-transition flex h-7 w-7 items-center justify-center rounded-[2px] border border-[#26312B] bg-[#141B18] text-[#ECE7DA] hover:border-[#8C958D] disabled:opacity-30 cursor-pointer"
        >
          <ChevronRight className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
      </div>

      {/* Film-editor tick timeline */}
      <div
        ref={rulerRef}
        data-day-scrubber="true"
        role="slider"
        tabIndex={canScrub ? 0 : -1}
        aria-label="Forecast day scrubber"
        aria-valuemin={1}
        aria-valuemax={Math.max(1, totalSteps)}
        aria-valuenow={stepIndex + 1}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        className={`relative flex h-8 min-w-[180px] flex-1 select-none items-center touch-none ${
          canScrub ? 'cursor-ew-resize' : 'opacity-40'
        }`}
      >
        {/* Horizontal baseline */}
        <div className="absolute right-0 left-0 h-[1px] bg-[#26312B]" />

        {/* Active elapsed rule */}
        {canScrub && (
          <div
            style={{ width: `${playheadPct}%` }}
            className="pointer-events-none absolute left-0 h-[1px] bg-[#F0B84B]"
          />
        )}

        {/* Step tick marks */}
        {canScrub &&
          Array.from({ length: totalSteps }, (_, i) => {
            const pct = maxIdx === 0 ? 0 : (i / maxIdx) * 100
            const isMajor = i === 0 || i === maxIdx || (i + 1) % 5 === 0
            const isPassed = i <= stepIndex
            return (
              <div
                key={i}
                style={{ left: `${pct}%` }}
                className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
              >
                <div
                  className={`w-[1px] ${isMajor ? 'h-3.5' : 'h-2'} ${
                    isPassed ? 'bg-[#8C958D]' : 'bg-[#26312B]'
                  }`}
                />
              </div>
            )
          })}

        {/* Playhead needle */}
        {canScrub && (
          <div
            style={{ left: `${playheadPct}%` }}
            className="pointer-events-none absolute top-0.5 bottom-0.5 -translate-x-1/2"
          >
            <div className="h-full w-[2px] bg-[#F0B84B]" />
          </div>
        )}
      </div>

      {/* Step counter */}
      <div className="font-mono text-xs text-[#ECE7DA] tabular-nums">
        {canScrub ? (
          <>
            Day <span className="text-[#F0B84B]">{stepIndex + 1}</span>
            <span className="text-[#5A645D]"> / {totalSteps}</span>
          </>
        ) : (
          <span className="text-[#5A645D]">Day —</span>
        )}
      </div>

      {/* Speed selector */}
      <div
        role="group"
        aria-label="Playback speed"
        className="flex items-center gap-2 border-l border-[#26312B] pl-4 font-mono text-xs tabular-nums"
      >
        {PLAYBACK_SPEEDS.map((spd) => (
          <button
            key={spd}
            type="button"
            onClick={() => onPlaybackSpeedChange(spd)}
            className={`ui-transition cursor-pointer ${
              playbackSpeed === spd
                ? 'font-medium text-[#F0B84B]'
                : 'text-[#5A645D] hover:text-[#ECE7DA]'
            }`}
          >
            {spd}×
          </button>
        ))}
      </div>
    </section>
  )
}

interface ChannelSelectorProps {
  channel: DisplayChannelId
  onChannelChange: (ch: DisplayChannelId) => void
}

/**
 * Slim right-rail channel selector (tabular instrument rows, 1px hairline rules).
 */
export function ChannelSelector({ channel, onChannelChange }: ChannelSelectorProps) {
  return (
    <div className="border-b border-[#26312B] pb-5">
      <p className="mb-2.5 text-xs text-[#8C958D]">Displayed channel</p>
      <div className="grid grid-cols-1 border-t border-[#26312B]">
        {DISPLAY_CHANNELS.map((ch) => {
          const active = ch.id === channel
          return (
            <button
              key={ch.id}
              type="button"
              onClick={() => onChannelChange(ch.id)}
              className={`ui-transition flex items-baseline justify-between border-b border-[#26312B] px-2.5 py-1.5 text-left cursor-pointer ${
                active
                  ? 'border-l-2 border-l-[#F0B84B] bg-[#141B18] text-[#ECE7DA]'
                  : 'text-[#8C958D] hover:bg-[#141B18]/60 hover:text-[#ECE7DA]'
              }`}
            >
              <span className="text-xs">
                <span className="font-mono text-[#ECE7DA]">{ch.shortLabel}</span>
                <span className="ml-2 text-[#8C958D]">{ch.label}</span>
              </span>
              <span className="font-mono text-[11px] text-[#5A645D]">{ch.displayUnit}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
