import { Bot, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono, StatusDot } from "../components/ui/Badge";
import { buttonClass, IconButton } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { TableSkeleton } from "../components/ui/Skeleton";
import {
  RowLink,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { useArms } from "../hooks/useArms";
import { formatTime, humanize, sentenceCase } from "../lib/format";
import { canStateTone, controlStateTone, isBusyStatus } from "../lib/status";
import type { ArmTelemetry, TelemetryConnectionState } from "../types/arms";

export function TelemetryBadge({ state }: { state: TelemetryConnectionState }) {
  const live = state === "live";
  const labels: Record<TelemetryConnectionState, string> = {
    connecting: "Connecting",
    live: "Telemetry live",
    reconnecting: "Reconnecting",
    offline: "Telemetry offline",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-2xs font-medium ${
        live ? "bg-positive-50 text-positive-700" : "bg-caution-50 text-caution-700"
      }`}
    >
      {live ? (
        <Wifi className="h-3 w-3" aria-hidden="true" />
      ) : (
        <WifiOff className="h-3 w-3" aria-hidden="true" />
      )}
      {labels[state]}
    </span>
  );
}

export function robotSubtitle(arm: ArmTelemetry): string {
  return [arm.side, arm.pair_id ? `pair ${arm.pair_id}` : null, arm.group_id]
    .filter(Boolean)
    .join(" · ");
}

function sortRobots(arms: ArmTelemetry[]): ArmTelemetry[] {
  return [...arms].sort((left, right) => {
    const pair = (left.pair_id ?? "~").localeCompare(right.pair_id ?? "~");
    if (pair !== 0) return pair;
    if (left.role !== right.role) return left.role === "leader" ? -1 : 1;
    return left.name.localeCompare(right.name);
  });
}

export function RobotsPage() {
  const { arms, loading, error, refresh, connectionState } = useArms();
  const robots = useMemo(() => sortRobots(arms), [arms]);
  const connectedCount = robots.filter((arm) => arm.connected).length;
  const warningCount = robots.reduce((total, arm) => total + arm.warnings.length, 0);

  return (
    <Page>
      <PageHeader
        title="Robots"
        description="Every logical YAM arm the configured driver exposes, with its durable identity, pairing, and live control state."
        meta={
          robots.length > 0 ? (
            <>
              <TelemetryBadge state={connectionState} />
              <span className="text-xs text-ink-muted">
                {connectedCount} of {robots.length} connected
              </span>
              {warningCount > 0 && (
                <span className="text-xs text-caution-700">
                  {warningCount} safety warning{warningCount === 1 ? "" : "s"}
                </span>
              )}
            </>
          ) : undefined
        }
        actions={
          <>
            <IconButton
              icon={RefreshCw}
              label="Refresh robots"
              spinning={loading}
              disabled={loading}
              onClick={() => void refresh()}
            />
            <Link to="/settings#yam-setup" className={buttonClass("primary", "md")}>
              Configure cell
            </Link>
          </>
        }
      />

      <PageSection>
        {loading && robots.length === 0 ? (
          <TableSkeleton rows={4} columns={5} label="Loading robots" />
        ) : robots.length === 0 ? (
          <EmptyState
            icon={error ? WifiOff : Bot}
            title={error ? "Robot telemetry is unavailable" : "No robots configured"}
            description={
              error ??
              "The configured driver exposes no arms yet. Assign physical transports to logical arms in the YAM cell setup."
            }
            action={
              <Link to="/settings#yam-setup" className={buttonClass("primary", "md")}>
                Open cell setup
              </Link>
            }
          />
        ) : (
          <>
            {warningCount > 0 && (
              <Alert tone="warning" title="Safety warnings reported" className="mb-5">
                {robots
                  .filter((arm) => arm.warnings.length > 0)
                  .map((arm) => `${arm.name}: ${arm.warnings.join("; ")}`)
                  .join(" · ")}
              </Alert>
            )}
            <Table label="Robots" minWidth="52rem">
              <TableHead>
                <TableHeaderCell>Robot</TableHeaderCell>
                <TableHeaderCell>Role</TableHeaderCell>
                <TableHeaderCell>Connection</TableHeaderCell>
                <TableHeaderCell>Bus</TableHeaderCell>
                <TableHeaderCell>End effector</TableHeaderCell>
                <TableHeaderCell align="right">Updated</TableHeaderCell>
              </TableHead>
              <TableBody>
                {robots.map((arm) => (
                  <TableRow key={arm.id} interactive>
                    <TableCell>
                      <RowLink to={`/robots/${encodeURIComponent(arm.id)}`}>{arm.name}</RowLink>
                      <p className="mt-0.5 truncate font-mono text-2xs text-ink-faint">{arm.id}</p>
                    </TableCell>
                    <TableCell>
                      <span className="text-ink">{sentenceCase(arm.role)}</span>
                      {robotSubtitle(arm) && (
                        <p className="mt-0.5 text-2xs text-ink-muted">{robotSubtitle(arm)}</p>
                      )}
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5">
                        <StatusDot
                          tone={controlStateTone[arm.control_state]}
                          pulse={isBusyStatus(arm.control_state)}
                        />
                        <span className="text-ink">{sentenceCase(humanize(arm.control_state))}</span>
                      </span>
                      <p className="mt-0.5 text-2xs text-ink-muted">
                        {arm.energized
                          ? arm.holding
                            ? "Energized · holding"
                            : "Energized"
                          : "Not energized"}
                      </p>
                    </TableCell>
                    <TableCell>
                      {arm.can ? (
                        <>
                          <Badge tone={canStateTone[arm.can.state]}>{humanize(arm.can.state)}</Badge>
                          <p className="mt-1 font-mono text-2xs text-ink-faint">
                            {arm.can.interface}
                          </p>
                        </>
                      ) : (
                        <span className="capitalize text-ink-muted">{arm.transport_kind}</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <Mono>{arm.end_effector_kind}</Mono>
                      {arm.warnings.length > 0 && (
                        <p className="mt-1 text-2xs font-medium text-caution-700">
                          {arm.warnings.length} warning{arm.warnings.length === 1 ? "" : "s"}
                        </p>
                      )}
                    </TableCell>
                    <TableCell align="right" muted>
                      {formatTime(arm.timestamp)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        )}
      </PageSection>
    </Page>
  );
}
