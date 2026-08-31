import {
  Bot,
  BrainCircuit,
  Database,
  RadioTower,
  Video,
} from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { Badge, StatusDot } from "../components/ui/Badge";
import { buttonClass } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { SectionHeading } from "../components/ui/Panel";
import { Stat, StatGrid } from "../components/ui/Stat";
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
import { useDeploymentList } from "../hooks/useInference";
import { useRecordingList } from "../hooks/useRecordings";
import { useTrainingRunList } from "../hooks/useTrainingRuns";
import { formatCount, formatRelative, humanize, sentenceCase } from "../lib/format";
import {
  controlStateTone,
  deploymentTone,
  isBusyStatus,
  recordingTone,
  trainingTone,
} from "../lib/status";
import { useSystemStatus } from "../state/systemStatus";
import type { Tone } from "../components/ui/Badge";

type Activity = {
  id: string;
  kind: "recording" | "training" | "deployment";
  title: string;
  detail: string;
  status: string;
  tone: Tone;
  timestamp: string;
  to: string;
};

const ACTIVITY_ICONS = {
  recording: Video,
  training: BrainCircuit,
  deployment: RadioTower,
};

export function OverviewPage() {
  const { status } = useSystemStatus();
  const { arms, connectionState } = useArms();
  const { recordings } = useRecordingList();
  const { runs } = useTrainingRunList();
  const { deployments } = useDeploymentList();

  const connectedRobots = arms.filter((arm) => arm.connected).length;
  const warnings = arms.flatMap((arm) => arm.warnings.map((warning) => ({ arm, warning })));
  const episodes = recordings.reduce((total, recording) => total + recording.episode_count, 0);
  const activeRuns = runs.filter(
    (run) => run.status === "running" || run.status === "created",
  ).length;
  const activeDeployments = deployments.filter(
    (deployment) => !["stopped", "failed"].includes(deployment.status),
  ).length;
  const runningSession = deployments.find((deployment) => deployment.status === "running");

  const activity = useMemo<Activity[]>(() => {
    const items: Activity[] = [
      ...recordings.map((recording) => ({
        id: `recording-${recording.id}`,
        kind: "recording" as const,
        title: recording.name,
        detail: `${recording.episode_count} episode${recording.episode_count === 1 ? "" : "s"}`,
        status: recording.status,
        tone: recordingTone[recording.status],
        timestamp: recording.updated_at,
        to: "/record",
      })),
      ...runs.map((run) => ({
        id: `run-${run.id}`,
        kind: "training" as const,
        title: run.name,
        detail: run.managed_job ? "Managed training" : "External training",
        status: run.status,
        tone: trainingTone[run.status],
        timestamp: run.updated_at,
        to: `/training/${encodeURIComponent(run.id)}`,
      })),
      ...deployments.map((deployment) => ({
        id: `deployment-${deployment.id}`,
        kind: "deployment" as const,
        title: deployment.name,
        detail: deployment.model_repo,
        status: deployment.status,
        tone: deploymentTone[deployment.status],
        timestamp: deployment.updated_at,
        to: `/inference/${encodeURIComponent(deployment.id)}`,
      })),
    ];
    return items
      .sort((left, right) => new Date(right.timestamp).valueOf() - new Date(left.timestamp).valueOf())
      .slice(0, 8);
  }, [deployments, recordings, runs]);

  return (
    <Page>
      <PageHeader
        title="Overview"
        description="The state of this cell: what is connected, what is being collected, what is training, and what is deployed."
        meta={
          status ? (
            <Badge tone={status.mode === "mock" ? "info" : "neutral"} dot>
              {status.mode === "mock" ? "Mock mode" : "Hardware mode"}
            </Badge>
          ) : undefined
        }
        actions={
          <Link to="/record" className={buttonClass("primary", "md")}>
            Record demonstrations
          </Link>
        }
      />

      {warnings.length > 0 && (
        <PageSection className="mt-8">
          <Alert
            tone={warnings.some((item) => item.warning.includes("NO SASH GUARD")) ? "danger" : "warning"}
            title={`${warnings.length} robot safety warning${warnings.length === 1 ? "" : "s"}`}
            action={
              <Link to="/robots" className={buttonClass("secondary", "sm")}>
                Inspect robots
              </Link>
            }
          >
            {warnings.map((item) => `${item.arm.name}: ${item.warning}`).join(" · ")}
          </Alert>
        </PageSection>
      )}

      <PageSection className="mt-8">
        <StatGrid columns={4}>
          <Stat
            icon={Bot}
            label="Robots"
            value={`${connectedRobots}/${arms.length || 0}`}
            hint={
              connectionState === "live" ? "Telemetry live" : `Telemetry ${connectionState}`
            }
          />
          <Stat
            icon={Video}
            label="Recording"
            value={formatCount(recordings.length)}
            hint={`${formatCount(episodes)} episodes captured`}
          />
          <Stat
            icon={BrainCircuit}
            label="Training"
            value={formatCount(runs.length)}
            hint={activeRuns > 0 ? `${activeRuns} in progress` : "No active runs"}
          />
          <Stat
            icon={RadioTower}
            label="Deployment"
            value={formatCount(deployments.length)}
            hint={
              runningSession
                ? `${runningSession.name} running`
                : activeDeployments > 0
                  ? `${activeDeployments} active`
                  : "Nothing deployed"
            }
          />
        </StatGrid>
      </PageSection>

      <PageSection>
        <div className="grid gap-10 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div>
            <SectionHeading
              title="Recent activity"
              description="Sessions, runs, and deployments ordered by their last update."
              className="mb-4"
            />
            {activity.length === 0 ? (
              <EmptyState
                icon={Database}
                title="Nothing has happened yet"
                description="Record a demonstration session to start the collect → train → deploy loop."
                action={
                  <Link to="/record" className={buttonClass("primary", "md")}>
                    Record demonstrations
                  </Link>
                }
              />
            ) : (
              <ul className="divide-y divide-line-subtle overflow-hidden rounded-xl border border-line bg-surface">
                {activity.map((item) => {
                  const Icon = ACTIVITY_ICONS[item.kind];
                  return (
                    <li key={item.id} className="relative flex items-center gap-4 px-5 py-3.5">
                      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-line-subtle text-ink-muted">
                        <Icon className="h-4 w-4" strokeWidth={1.8} aria-hidden="true" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <Link
                          to={item.to}
                          className="text-[13px] font-medium text-ink after:absolute after:inset-0 after:content-[''] hover:text-accent-700"
                        >
                          {item.title}
                        </Link>
                        <p className="mt-0.5 truncate text-2xs text-ink-muted">{item.detail}</p>
                      </span>
                      <Badge tone={item.tone} dot pulse={isBusyStatus(item.status)}>
                        {item.status}
                      </Badge>
                      <span className="w-16 shrink-0 text-right text-2xs text-ink-faint">
                        {formatRelative(item.timestamp)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="space-y-10">
            <div>
              <SectionHeading
                title="Robots"
                className="mb-4"
                actions={
                  <Link to="/robots" className="text-xs font-medium text-accent-700 hover:text-accent-800">
                    View all
                  </Link>
                }
              />
              {arms.length === 0 ? (
                <EmptyState
                  icon={Bot}
                  title="No robots configured"
                  description="Assign physical transports to logical arms in the YAM cell setup."
                  action={
                    <Link to="/settings#yam-setup" className={buttonClass("primary", "md")}>
                      Open cell setup
                    </Link>
                  }
                />
              ) : (
                <Table label="Robot summary" minWidth="20rem">
                  <TableHead>
                    <TableHeaderCell>Robot</TableHeaderCell>
                    <TableHeaderCell>State</TableHeaderCell>
                  </TableHead>
                  <TableBody>
                    {arms.slice(0, 4).map((arm) => (
                      <TableRow key={arm.id} interactive>
                        <TableCell>
                          <RowLink to={`/robots/${encodeURIComponent(arm.id)}`}>{arm.name}</RowLink>
                          <p className="mt-0.5 text-2xs text-ink-faint">
                            {sentenceCase(arm.role)}
                            {arm.pair_id ? ` · pair ${arm.pair_id}` : ""}
                          </p>
                        </TableCell>
                        <TableCell>
                          <span className="inline-flex items-center gap-1.5">
                            <StatusDot
                              tone={controlStateTone[arm.control_state]}
                              pulse={isBusyStatus(arm.control_state)}
                            />
                            {sentenceCase(humanize(arm.control_state))}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </div>

            {status && (
              <div>
                <SectionHeading
                  title="Services"
                  className="mb-4"
                  actions={
                    <Link
                      to="/settings"
                      className="text-xs font-medium text-accent-700 hover:text-accent-800"
                    >
                      Settings
                    </Link>
                  }
                />
                <ul className="divide-y divide-line-subtle overflow-hidden rounded-xl border border-line bg-surface">
                  {status.services.map((service) => {
                    const ready = ["connected", "configured"].includes(service.status);
                    return (
                      <li
                        key={service.id}
                        className="flex items-center justify-between gap-3 px-5 py-3"
                      >
                        <span className="min-w-0">
                          <span className="block text-[13px] font-medium text-ink">
                            {service.label}
                          </span>
                          <span className="mt-0.5 block truncate text-2xs text-ink-muted">
                            {service.detail}
                          </span>
                        </span>
                        <Badge tone={ready ? "success" : service.required ? "warning" : "neutral"}>
                          {service.status}
                        </Badge>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
      </PageSection>
    </Page>
  );
}
