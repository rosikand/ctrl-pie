import { Bot, Check, Cpu, Package, RadioTower } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { InferenceReadinessNotice, useInferenceReadiness } from "../components/InferenceReadiness";
import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { ChoiceCard, Field, Select, TextInput } from "../components/ui/Form";
import { useArms } from "../hooks/useArms";
import { useDeployPolicy } from "../hooks/useInference";
import { usePublicSettings } from "../hooks/usePublicSettings";
import { useTrainerModels } from "../hooks/useTrainerModels";
import { shortRevision } from "../lib/format";
import type { InferenceComputeSize, InferenceRuntime } from "../types/inference";
import type { TrainerModelSummary } from "../types/training";

const GPU_OPTIONS: Exclude<InferenceComputeSize, "CPU">[] = [
  "Modal: A10G",
  "Modal: A100",
  "Modal: H100",
];

const GPU_DESCRIPTIONS: Record<string, string> = {
  "Modal: A10G": "Smallest GPU. Good default for small LeRobot policies.",
  "Modal: A100": "More memory and throughput for larger checkpoints.",
  "Modal: H100": "Highest throughput and highest cost per minute.",
};

const MOCK_POLICY: TrainerModelSummary = {
  repo_id: "ctrl-pi/mock-policy",
  name: "mock-policy",
  revision: "0000000000000000000000000000000000000000",
  hub_url: "",
  private: true,
  gated: false,
  last_modified: null,
  pipeline_tag: "robotics",
  library_name: "ctrl-pi-stub",
  tags: ["mock", "offline"],
  card: {
    description: "Deterministic no-network policy for the complete mock inference loop.",
    base_model: [],
    datasets: [],
  },
  checkpoints: [],
};

type StepId = "model" | "robot" | "compute" | "deploy";

const STEPS: { id: StepId; label: string; icon: typeof Package }[] = [
  { id: "model", label: "Model", icon: Package },
  { id: "robot", label: "Robot", icon: Bot },
  { id: "compute", label: "Compute", icon: Cpu },
  { id: "deploy", label: "Deploy", icon: RadioTower },
];

function StepFrame({
  index,
  step,
  active,
  complete,
  summary,
  onEdit,
  children,
}: {
  index: number;
  step: { id: StepId; label: string };
  active: boolean;
  complete: boolean;
  summary?: string;
  onEdit: () => void;
  children: ReactNode;
}) {
  return (
    <section
      className={`rounded-xl border bg-surface transition ${
        active ? "border-line-strong" : "border-line"
      }`}
    >
      <div className="flex items-center justify-between gap-4 px-5 py-4">
        <div className="flex min-w-0 items-center gap-3">
          <span
            className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-2xs font-medium ${
              complete
                ? "bg-positive-50 text-positive-700"
                : active
                  ? "bg-ink text-white"
                  : "bg-line-subtle text-ink-faint"
            }`}
          >
            {complete ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : index + 1}
          </span>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold tracking-tight text-ink">{step.label}</h2>
            {!active && complete && summary && (
              <p className="mt-0.5 truncate text-xs text-ink-muted">{summary}</p>
            )}
          </div>
        </div>
        {!active && complete && (
          <Button size="sm" variant="ghost" onClick={onEdit}>
            Edit
          </Button>
        )}
      </div>
      {active && <div className="border-t border-line px-5 py-5">{children}</div>}
    </section>
  );
}

export function InferenceNewPage() {
  const navigate = useNavigate();
  const arms = useArms();
  const models = useTrainerModels();
  const { settings } = usePublicSettings();
  const { readiness, ready: environmentReady } = useInferenceReadiness();
  const { deploy, busy, error } = useDeployPolicy();

  const [step, setStep] = useState<StepId>("model");
  const [furthest, setFurthest] = useState(0);
  const [modelRepo, setModelRepo] = useState("");
  const [runtime, setRuntime] = useState<Exclude<InferenceRuntime, "stub">>("lerobot");
  const [armId, setArmId] = useState("");
  const [computeSize, setComputeSize] = useState<Exclude<InferenceComputeSize, "CPU">>(
    "Modal: A10G",
  );
  const [name, setName] = useState("");
  const defaultsApplied = useRef(false);

  const availableModels = useMemo(() => {
    const discovered = models.data?.models ?? [];
    if (!readiness?.mock_mode) return discovered;
    return [MOCK_POLICY, ...discovered.filter((model) => model.repo_id !== MOCK_POLICY.repo_id)];
  }, [models.data?.models, readiness?.mock_mode]);

  const followers = useMemo(
    () => arms.arms.filter((arm) => arm.role === "follower" && arm.connected),
    [arms.arms],
  );

  const selectedModel = availableModels.find((model) => model.repo_id === modelRepo) ?? null;
  const selectedFollower = followers.find((arm) => arm.id === armId) ?? null;

  useEffect(() => {
    if (!settings || defaultsApplied.current) return;
    defaultsApplied.current = true;
    setRuntime(settings.default_runtime);
    setComputeSize(settings.default_compute);
  }, [settings]);

  useEffect(() => {
    if (availableModels.length === 0) return;
    if (!availableModels.some((model) => model.repo_id === modelRepo)) {
      setModelRepo(availableModels[0].repo_id);
    }
  }, [availableModels, modelRepo]);

  useEffect(() => {
    if (!followers.some((arm) => arm.id === armId)) setArmId(followers[0]?.id ?? "");
  }, [armId, followers]);

  useEffect(() => {
    if (!name && selectedModel) setName(selectedModel.name);
  }, [name, selectedModel]);

  const stepIndex = STEPS.findIndex((item) => item.id === step);

  function advance(next: StepId) {
    const index = STEPS.findIndex((item) => item.id === next);
    setFurthest((current) => Math.max(current, index));
    setStep(next);
  }

  async function submit() {
    if (!selectedModel || !name.trim() || !environmentReady) return;
    const created = await deploy({
      name: name.trim(),
      model_repo: selectedModel.repo_id,
      checkpoint_revision: selectedModel.revision,
      runtime,
      compute_size: computeSize,
    });
    if (created) {
      navigate(`/inference/${encodeURIComponent(created.id)}`, { state: { armId } });
    }
  }

  return (
    <Page>
      <PageHeader
        back={{ to: "/inference", label: "Inference" }}
        title="New deployment"
        description="Choose a policy, the robot that will execute it, and the compute that serves it. Deploying verifies endpoint identity only; it never moves the robot."
      />

      <PageSection className="mt-8 space-y-5">
        <InferenceReadinessNotice />
        {error && (
          <Alert tone="danger" title={error.message}>
            {error.status === 503
              ? "Complete the required backend configuration in Settings."
              : error.status === 502
                ? "The backend safely rejected a provider or runtime failure. Check the server configuration and retry."
                : "Review the selected model, runtime, and compute, then retry."}
          </Alert>
        )}

        {STEPS.map((item, index) => {
          const complete = index < stepIndex || (index <= furthest && index !== stepIndex);
          const summary =
            item.id === "model"
              ? selectedModel
                ? `${selectedModel.repo_id} · ${runtime === "lerobot" ? "LeRobot" : "OpenPI"}`
                : undefined
              : item.id === "robot"
                ? selectedFollower
                  ? `${selectedFollower.name} · ${selectedFollower.id}`
                  : "No connected follower"
                : item.id === "compute"
                  ? computeSize
                  : name || undefined;

          return (
            <StepFrame
              key={item.id}
              index={index}
              step={item}
              active={step === item.id}
              complete={complete}
              summary={summary}
              onEdit={() => setStep(item.id)}
            >
              {item.id === "model" && (
                <div className="space-y-5">
                  {models.error && (
                    <Alert tone="warning" title="Model catalog unavailable">
                      {models.error.message}
                    </Alert>
                  )}
                  {availableModels.length === 0 ? (
                    <p className="text-xs text-ink-muted">
                      {models.initialLoading
                        ? "Loading the namespace model catalog…"
                        : "No deployable models were found in the configured namespace."}
                    </p>
                  ) : (
                    <div className="grid gap-2">
                      {availableModels.slice(0, 12).map((model) => (
                        <ChoiceCard
                          key={model.repo_id}
                          name="model"
                          selected={model.repo_id === modelRepo}
                          onSelect={() => setModelRepo(model.repo_id)}
                          title={
                            model.repo_id === MOCK_POLICY.repo_id
                              ? "Offline mock policy"
                              : model.repo_id
                          }
                          meta={
                            model.revision ? `@${shortRevision(model.revision, 10)}` : "default branch"
                          }
                          description={
                            model.repo_id === MOCK_POLICY.repo_id
                              ? "Deterministic no-network policy for the complete mock inference loop."
                              : model.card?.description?.trim() ||
                                `${model.pipeline_tag ?? "policy"} · ${model.checkpoints.length} checkpoint files`
                          }
                        />
                      ))}
                      {availableModels.length > 12 && (
                        <p className="text-2xs text-ink-faint">
                          Showing the first 12 models. Browse the full catalog on the Models page.
                        </p>
                      )}
                    </div>
                  )}

                  <Field
                    label="Runtime"
                    hint="OpenPI is emulated in mock mode and intentionally unavailable on real Modal in V1."
                  >
                    <Select
                      value={runtime}
                      onChange={(event) =>
                        setRuntime(event.target.value as Exclude<InferenceRuntime, "stub">)
                      }
                    >
                      <option value="lerobot">LeRobot</option>
                      <option value="openpi">OpenPI</option>
                    </Select>
                  </Field>

                  <div className="flex justify-end">
                    <Button
                      variant="primary"
                      disabled={!selectedModel}
                      onClick={() => advance("robot")}
                    >
                      Continue
                    </Button>
                  </div>
                </div>
              )}

              {item.id === "robot" && (
                <div className="space-y-5">
                  <Field
                    label="Connected follower"
                    hint={`Only connected arms whose authoritative role is follower are eligible. Telemetry: ${arms.connectionState}`}
                  >
                    <Select
                      value={armId}
                      onChange={(event) => setArmId(event.target.value)}
                      disabled={followers.length === 0}
                    >
                      {followers.length === 0 && <option value="">No connected follower</option>}
                      {followers.map((arm) => (
                        <option key={arm.id} value={arm.id}>
                          {arm.name} · {arm.id}
                          {arm.side ? ` · ${arm.side}` : ""}
                          {arm.pair_id ? ` · pair ${arm.pair_id}` : ""}
                        </option>
                      ))}
                    </Select>
                  </Field>

                  {followers.length === 0 && (
                    <Alert tone="warning" title="No connected follower arm">
                      Connect a follower in the YAM cell setup before starting execution. You can
                      still deploy the endpoint now and choose the robot when you start.
                    </Alert>
                  )}

                  {selectedFollower?.warnings.some((warning) => warning.includes("NO SASH GUARD")) && (
                    <Alert tone="danger" title="NO SASH GUARD">
                      This follower has no soft limits configured. ctrl-π never invents limits.
                    </Alert>
                  )}

                  <div className="flex justify-end">
                    <Button variant="primary" onClick={() => advance("compute")}>
                      Continue
                    </Button>
                  </div>
                </div>
              )}

              {item.id === "compute" && (
                <div className="space-y-5">
                  <div className="grid gap-2">
                    {GPU_OPTIONS.map((option) => (
                      <ChoiceCard
                        key={option}
                        name="compute"
                        selected={option === computeSize}
                        onSelect={() => setComputeSize(option)}
                        title={option}
                        description={GPU_DESCRIPTIONS[option]}
                      />
                    ))}
                  </div>
                  {readiness?.mock_mode && (
                    <p className="text-2xs leading-5 text-ink-muted">
                      In mock mode the selection is recorded but no GPU is allocated.
                    </p>
                  )}
                  <div className="flex justify-end">
                    <Button variant="primary" onClick={() => advance("deploy")}>
                      Continue
                    </Button>
                  </div>
                </div>
              )}

              {item.id === "deploy" && (
                <div className="space-y-5">
                  <Field label="Deployment name" hint="Shown in the deployment history.">
                    <TextInput
                      value={name}
                      maxLength={160}
                      placeholder="Pick policy"
                      onChange={(event) => setName(event.target.value)}
                    />
                  </Field>

                  <dl className="grid gap-x-8 gap-y-4 rounded-lg border border-line bg-canvas px-4 py-4 sm:grid-cols-2">
                    <div>
                      <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                        Policy
                      </dt>
                      <dd className="mt-1 break-all font-mono text-xs text-ink">
                        {selectedModel?.repo_id ?? "—"}
                        <span className="ml-1 text-ink-faint">
                          @{shortRevision(selectedModel?.revision ?? null, 10) || "default"}
                        </span>
                      </dd>
                    </div>
                    <div>
                      <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                        Runtime and compute
                      </dt>
                      <dd className="mt-1 text-xs text-ink">
                        {runtime === "lerobot" ? "LeRobot" : "OpenPI"} · {computeSize}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                        Robot
                      </dt>
                      <dd className="mt-1 text-xs text-ink">
                        {selectedFollower ? (
                          <>
                            {selectedFollower.name}{" "}
                            <Mono title={selectedFollower.id}>{selectedFollower.id}</Mono>
                          </>
                        ) : (
                          <span className="text-ink-muted">Chosen when execution starts</span>
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                        Mode
                      </dt>
                      <dd className="mt-1 text-xs text-ink">
                        <Badge tone={readiness?.mock_mode ? "info" : "neutral"}>
                          {readiness?.mock_mode ? "Mock runtime" : "Modal runtime"}
                        </Badge>
                      </dd>
                    </div>
                  </dl>

                  <Alert tone="neutral" title="Deploy does not move the robot">
                    Deploying verifies the exact runtime, model, and revision identity of the
                    endpoint. Execution starts only when you explicitly start a session on the
                    deployment page.
                  </Alert>

                  <div className="flex justify-end">
                    <Button
                      variant="primary"
                      size="lg"
                      icon={RadioTower}
                      loading={busy}
                      disabled={!selectedModel || !name.trim() || !environmentReady}
                      onClick={() => void submit()}
                    >
                      {busy ? "Deploying and verifying…" : "Deploy"}
                    </Button>
                  </div>
                </div>
              )}
            </StepFrame>
          );
        })}
      </PageSection>
    </Page>
  );
}
