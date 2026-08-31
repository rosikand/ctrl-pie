import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

export type ChartPoint = { step: number; value: number };

const LINE = "#2563eb"; // accent-600: the single series hue.
const AREA = "rgba(37, 99, 235, 0.10)"; // ~10% wash, never a saturated block.
const GRID = "#e7e7ea"; // one step off surface, hairline, solid.
const SURFACE = "#ffffff";

const PAD = { top: 16, right: 18, bottom: 26, left: 56 };

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const rawStep = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step =
    (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 0.001; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

function useElementWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next && next > 0) setWidth(next);
    });
    observer.observe(element);
    setWidth(element.clientWidth || fallback);
    return () => observer.disconnect();
  }, [fallback]);
  return [ref, width] as const;
}

/**
 * One metric, one chart. Small multiples rather than a multi-series overlay, so
 * there is never a second y-axis and never a color legend to decode.
 */
export function LineChart({
  points,
  height = 220,
  formatValue,
  formatStep,
  label,
  className = "",
}: {
  points: ChartPoint[];
  height?: number;
  formatValue: (value: number) => string;
  formatStep: (step: number) => string;
  label: string;
  className?: string;
}) {
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const plotted = useMemo(
    () =>
      points
        .filter((point) => Number.isFinite(point.step) && Number.isFinite(point.value))
        .sort((left, right) => left.step - right.step),
    [points],
  );

  const geometry = useMemo(() => {
    if (plotted.length === 0) return null;
    const innerWidth = Math.max(80, width - PAD.left - PAD.right);
    const innerHeight = Math.max(60, height - PAD.top - PAD.bottom);
    const values = plotted.map((point) => point.value);
    const steps = plotted.map((point) => point.step);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const spread = rawMax - rawMin;
    const padding = spread === 0 ? Math.max(Math.abs(rawMax) * 0.1, 0.5) : spread * 0.08;
    const minValue = rawMin - padding;
    const maxValue = rawMax + padding;
    const minStep = Math.min(...steps);
    const maxStep = Math.max(...steps);
    const stepRange = maxStep - minStep;
    const x = (step: number) =>
      PAD.left + (stepRange === 0 ? innerWidth / 2 : ((step - minStep) / stepRange) * innerWidth);
    const y = (value: number) =>
      PAD.top + ((maxValue - value) / (maxValue - minValue)) * innerHeight;
    const coordinates = plotted.map((point) => ({ x: x(point.step), y: y(point.value) }));
    const line = coordinates
      .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
      .join(" ");
    const baseline = PAD.top + innerHeight;
    const area = `${line} L${coordinates.at(-1)!.x.toFixed(2)},${baseline} L${coordinates[0].x.toFixed(2)},${baseline} Z`;
    return {
      innerWidth,
      innerHeight,
      baseline,
      coordinates,
      line,
      area,
      ticks: niceTicks(rawMin, rawMax).map((value) => ({ value, y: y(value) })),
      minStep,
      maxStep,
      rawMin,
      rawMax,
      x,
    };
  }, [height, plotted, width]);

  const onPointerMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (!geometry) return;
      const bounds = event.currentTarget.getBoundingClientRect();
      const pointerX = event.clientX - bounds.left;
      let nearest = 0;
      let best = Number.POSITIVE_INFINITY;
      geometry.coordinates.forEach((coordinate, index) => {
        const distance = Math.abs(coordinate.x - pointerX);
        if (distance < best) {
          best = distance;
          nearest = index;
        }
      });
      setHoverIndex(nearest);
    },
    [geometry],
  );

  if (plotted.length === 0 || !geometry) {
    return (
      <div
        ref={containerRef}
        className={`grid place-items-center rounded-lg border border-dashed border-line text-xs text-ink-muted ${className}`}
        style={{ height }}
      >
        No valid points reported
      </div>
    );
  }

  const active = hoverIndex === null ? null : plotted[hoverIndex];
  const activeCoordinate = hoverIndex === null ? null : geometry.coordinates[hoverIndex];
  const lastCoordinate = geometry.coordinates.at(-1)!;
  const tooltipLeft = activeCoordinate
    ? Math.min(Math.max(activeCoordinate.x, 60), Math.max(width - 60, 60))
    : 0;

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${label} from step ${geometry.minStep} to ${geometry.maxStep}`}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHoverIndex(null)}
        className="touch-none"
      >
        {geometry.ticks.map((tick) => (
          <g key={tick.value}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={tick.y}
              y2={tick.y}
              stroke={GRID}
              strokeWidth={1}
            />
            <text
              x={PAD.left - 10}
              y={tick.y}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-ink-faint text-[10px] tabular-nums"
            >
              {formatValue(tick.value)}
            </text>
          </g>
        ))}

        <path d={geometry.area} fill={AREA} />
        <path
          d={geometry.line}
          fill="none"
          stroke={LINE}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {activeCoordinate && (
          <line
            x1={activeCoordinate.x}
            x2={activeCoordinate.x}
            y1={PAD.top}
            y2={geometry.baseline}
            stroke={GRID}
            strokeWidth={1}
          />
        )}

        <circle
          cx={lastCoordinate.x}
          cy={lastCoordinate.y}
          r={4}
          fill={LINE}
          stroke={SURFACE}
          strokeWidth={2}
        />
        {activeCoordinate && (
          <circle
            cx={activeCoordinate.x}
            cy={activeCoordinate.y}
            r={4.5}
            fill={LINE}
            stroke={SURFACE}
            strokeWidth={2}
          />
        )}

        <text
          x={PAD.left}
          y={height - 8}
          className="fill-ink-faint text-[10px] tabular-nums"
        >
          {formatStep(geometry.minStep)}
        </text>
        <text
          x={width - PAD.right}
          y={height - 8}
          textAnchor="end"
          className="fill-ink-faint text-[10px] tabular-nums"
        >
          {formatStep(geometry.maxStep)}
        </text>
      </svg>

      {active && activeCoordinate && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 rounded-lg border border-line bg-surface px-2.5 py-1.5 shadow-xs"
          style={{ left: tooltipLeft, top: 4 }}
        >
          <p className="font-mono text-xs font-medium tabular-nums text-ink">
            {formatValue(active.value)}
          </p>
          <p className="mt-0.5 text-2xs tabular-nums text-ink-muted">
            step {formatStep(active.step)}
          </p>
        </div>
      )}
    </div>
  );
}

/** Axis-free 12-point trend used inside dense summaries. */
export function Sparkline({
  points,
  width = 96,
  height = 28,
  className = "",
}: {
  points: number[];
  width?: number;
  height?: number;
  className?: string;
}) {
  const finite = points.filter((value) => Number.isFinite(value));
  if (finite.length < 2) return null;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const range = max - min || 1;
  const path = finite
    .map((value, index) => {
      const x = (index / (finite.length - 1)) * (width - 2) + 1;
      const y = height - 3 - ((value - min) / range) * (height - 6);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} aria-hidden="true" className={className}>
      <path d={path} fill="none" stroke={LINE} strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
