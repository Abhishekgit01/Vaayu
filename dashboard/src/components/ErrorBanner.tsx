interface ErrorBannerProps {
  title: string
  message: string
  onRetry?: () => void
  onDismiss?: () => void
}

export function ErrorBanner({ title, message, onRetry, onDismiss }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className="w-full border-b border-[#26312B] bg-[#D9593D] px-6 py-2 text-[#0D1210]"
    >
      <div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-4">
        <p className="min-w-0 text-xs leading-relaxed">
          <span className="font-semibold">{title}:</span>{' '}
          <span className="break-words font-mono">{message}</span>
        </p>
        <div className="flex shrink-0 items-center gap-4 text-xs font-medium">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="underline decoration-[#0D1210]/60 underline-offset-4 hover:decoration-[#0D1210] cursor-pointer"
            >
              Retry
            </button>
          )}
          {onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              className="underline decoration-[#0D1210]/60 underline-offset-4 hover:decoration-[#0D1210] cursor-pointer"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
