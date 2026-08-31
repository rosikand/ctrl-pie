export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden="true" className={`animate-pulse rounded-md bg-line-subtle ${className}`} />;
}

export function TableSkeleton({
  rows = 5,
  columns = 4,
  label = "Loading",
}: {
  rows?: number;
  columns?: number;
  label?: string;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface" aria-busy="true">
      <p className="sr-only" role="status">
        {label}
      </p>
      <div className="h-10 border-b border-line bg-canvas/70" />
      <div className="divide-y divide-line-subtle">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div
            key={rowIndex}
            className="grid gap-4 px-5 py-4"
            style={{ gridTemplateColumns: `1.6fr ${"1fr ".repeat(Math.max(0, columns - 1))}` }}
          >
            {Array.from({ length: columns }).map((__, columnIndex) => (
              <Skeleton key={columnIndex} className="h-4" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
