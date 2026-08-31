import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { buttonClass } from "../components/ui/Button";
import { CodeBlock, InlineCode } from "../components/ui/Code";
import { SectionHeading } from "../components/ui/Panel";
import { useSystemStatus } from "../state/systemStatus";

function Step({
  index,
  title,
  children,
}: {
  index: number;
  title: string;
  children: ReactNode;
}) {
  return (
    <li className="relative grid grid-cols-[2rem_minmax(0,1fr)] gap-x-4 pb-10 last:pb-0">
      <div className="flex flex-col items-center">
        <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full border border-line bg-surface text-xs font-medium text-ink">
          {index}
        </span>
        <span aria-hidden="true" className="mt-2 w-px flex-1 bg-line" />
      </div>
      <div className="min-w-0 pt-0.5">
        <h3 className="text-sm font-semibold tracking-tight text-ink">{title}</h3>
        <div className="mt-3 space-y-4 text-[13px] leading-6 text-ink-secondary">{children}</div>
      </div>
    </li>
  );
}

export function SetupGuidePage() {
  const { status } = useSystemStatus();

  return (
    <Page>
      <PageHeader
        title="Set up"
        description="Go from a fresh git clone to YAM arms moving under ctrl-π on an Ubuntu box. Follow the steps in order; every hardware step is explicit and reversible."
        actions={
          <Link to="/settings" className={buttonClass("primary", "md")}>
            Open settings
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        }
      />

      <PageSection className="mt-8 max-w-prose">
        <Alert
          tone={status?.mode === "mock" ? "info" : "neutral"}
          title={
            status?.mode === "mock"
              ? "This install is running in mock mode"
              : status
                ? "This install is running in hardware mode"
                : "Checking the current mode"
          }
        >
          {status?.mode === "mock"
            ? "Steps 1–5 apply to what you are running now. Steps 6–10 switch the same install onto real YAM arms."
            : "Hardware mode never falls back to mocks: missing configuration or devices fail closed."}
        </Alert>
      </PageSection>

      <PageSection className="max-w-prose">
        <SectionHeading
          title="Installation"
          description="Docker is the shortest path to a production-shaped, single-process service."
          className="mb-6"
        />
        <ol>
          <Step index={1} title="Check the Ubuntu host">
            <p>You need, on the machine physically attached to the arms:</p>
            <ul className="list-disc space-y-1 pl-5">
              <li>Ubuntu with Docker Engine plus current Compose and Buildx (Compose 2.24+).</li>
              <li>PostgreSQL 14 or newer, reachable from inside the container.</li>
              <li>
                For real arms: the YAM leader/follower pairs wired to USB-CAN adapters, with CAN
                links already brought up by the host&apos;s own approved procedure.
              </li>
            </ul>
            <p>
              ctrl-π never runs <InlineCode>ip link set</InlineCode>, never changes bitrate, and
              never pings a motor during discovery. The host owns CAN link state.
            </p>
          </Step>

          <Step index={2} title="Clone and configure">
            <CodeBlock
              code={`git clone https://github.com/rosikand/ctrl-pie.git
cd ctrl-pie
cp .env.example .env
\${EDITOR:-vi} .env`}
            />
            <p>
              Set <InlineCode>DATABASE_URL</InlineCode> and leave{" "}
              <InlineCode>CTRL_PI_MOCK_MODE=true</InlineCode> for the first run. Hugging Face and
              Modal credentials can stay blank until you use those workflows.
            </p>
            <Alert tone="warning" title="A container-local localhost is not the host">
              A database URL containing <InlineCode>localhost</InlineCode> points back into the
              container. Use a hosted URL or a host/LAN address the container can reach.
            </Alert>
          </Step>

          <Step index={3} title="Start in mock mode">
            <CodeBlock
              code={`docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health`}
            />
            <p>
              Migrations run before the service starts. Open{" "}
              <InlineCode>http://127.0.0.1:8000</InlineCode>. The four-arm mock cell, synthetic
              camera, and stub compute need no YAM device and no cloud account.
            </p>
            <Alert tone="danger" title="ctrl-π has no authentication">
              Bind it to localhost or a trusted, firewalled LAN. Never expose it directly to the
              public Internet.
            </Alert>
          </Step>

          <Step index={4} title="Walk the mock loop once">
            <p>
              Record a short episode, publish it, and deploy the stub policy, so you know the whole
              path works before hardware is involved:
            </p>
            <ul className="list-disc space-y-1 pl-5">
              <li>
                <Link to="/record" className="font-medium text-accent-700 hover:text-accent-800">
                  Record
                </Link>{" "}
                — create a session, start teleop, enable slow sync, capture one episode.
              </li>
              <li>
                <Link to="/datasets" className="font-medium text-accent-700 hover:text-accent-800">
                  Datasets
                </Link>{" "}
                — browse the uploaded LeRobot dataset and its episodes.
              </li>
              <li>
                <Link to="/training" className="font-medium text-accent-700 hover:text-accent-800">
                  Training
                </Link>{" "}
                — watch a managed or externally reported run.
              </li>
              <li>
                <Link to="/inference" className="font-medium text-accent-700 hover:text-accent-800">
                  Inference
                </Link>{" "}
                — deploy the offline mock policy and start a session.
              </li>
            </ul>
            <p>The same loop runs headless:</p>
            <CodeBlock code="docker compose exec app python -m ctrl_pi.smoke --fake-hub" />
          </Step>

          <Step index={5} title="Add cloud accounts (optional)">
            <p>
              Hugging Face stores datasets and models; Modal runs managed training and real policy
              serving. Both stay server-side in <InlineCode>.env</InlineCode> — the browser never
              receives a credential.
            </p>
            <CodeBlock
              caption=".env"
              code={`HF_TOKEN=
HF_NAMESPACE=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
MODAL_PROXY_TOKEN_ID=
MODAL_PROXY_TOKEN_SECRET=`}
            />
            <p>
              Restart the service, then confirm each pair reads as ready in{" "}
              <Link to="/settings" className="font-medium text-accent-700 hover:text-accent-800">
                Settings → Credentials
              </Link>
              .
            </p>
          </Step>
        </ol>
      </PageSection>

      <PageSection className="max-w-prose">
        <SectionHeading
          title="Connecting real YAM arms"
          description="Hardware mode is a deliberate switch. It never silently falls back to mocks."
          className="mb-6"
        />
        <ol>
          <Step index={6} title="Pin the operator-owned i2rt checkout">
            <p>
              All-CAN workers load only a read-only checkout you provide. ctrl-π never fetches
              i2rt, resolves a branch, or chooses latest.
            </p>
            <CodeBlock
              caption=".env"
              code={`CTRL_PI_I2RT_CHECKOUT=/opt/i2rt
CTRL_PI_I2RT_DEPENDENCY_COMMIT=<full 40-character lowercase commit>`}
            />
            <CodeBlock code="make docker-yam-cell" />
            <p>
              The build refuses to continue unless the checkout root&apos;s HEAD matches that exact
              commit. Rebuild whenever the configured or mounted commit changes.
            </p>
          </Step>

          <Step index={7} title="Verify host CAN state and exclusive ownership">
            <CodeBlock
              code={`if pgrep -af '[p]ython.*minimum_gello\\.py'; then
  echo "BLOCKED: stop the external Lux/i2rt controller before ctrl-pi hardware startup" >&2
  exit 1
fi
ip -details -statistics link show type can`}
            />
            <p>
              Any other service that can write the same CAN buses is a blocker. Stop it through its
              own reviewed procedure first — the file-bus lease only serializes ctrl-π processes.
            </p>
          </Step>

          <Step index={8} title="Start hardware mode">
            <CodeBlock
              code={`docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml \\
  up -d --wait`}
            />
            <p>
              The override uses host networking, mounts <InlineCode>/sys:ro</InlineCode> and the
              pinned checkout at <InlineCode>/opt/i2rt:ro</InlineCode>, and adds only{" "}
              <InlineCode>NET_RAW</InlineCode>. It listens on{" "}
              <InlineCode>CTRL_PI_LISTEN_PORT</InlineCode> (default 8010) instead of publishing a
              bridge port.
            </p>
          </Step>

          <Step index={9} title="Configure the cell in Settings">
            <p>
              Open{" "}
              <Link
                to="/settings#yam-setup"
                className="font-medium text-accent-700 hover:text-accent-800"
              >
                Settings → YAM cell
              </Link>{" "}
              and follow it in order:
            </p>
            <ol className="list-decimal space-y-1 pl-5">
              <li>
                <strong>Discover</strong> — read-only OS inspection maps each USB-CAN adapter serial
                to its current interface. Roles are never inferred.
              </li>
              <li>
                <strong>Assign arms</strong> — logical id, role, pair/group/side, durable identity,
                and end effector. Never store <InlineCode>canN</InlineCode> as identity.
              </li>
              <li>
                <strong>Preflight</strong> — validates topology, identity resolution, link state,
                the pinned checkout, frame maps, and soft limits without opening hardware.
              </li>
              <li>
                <strong>Save the cell</strong> — persists normalized rows only.
              </li>
              <li>
                <strong>Connect selected arms</strong> — a separate, acknowledged, motion-capable
                operation.
              </li>
            </ol>
            <Alert tone="danger" title="Connecting a real follower can move it">
              Connect can energize motors, start gravity compensation, and make a CAN arm resist
              manual motion. Calibrating a <InlineCode>linear_4310</InlineCode> or{" "}
              <InlineCode>crank_4310</InlineCode> follower moves the jaws. Secure the workspace and
              verify the emergency stop first.
            </Alert>
            <p>
              A follower with no soft-limit file is shown as <strong>NO SASH GUARD</strong>. ctrl-π
              never invents limits.
            </p>
          </Step>

          <Step index={10} title="Run the ordered field-test gates">
            <p>
              Container access, directions, offsets, limits, calibration, loop behavior, E-stop
              response, and disconnect recovery are only proven on a physical rig. Work through the
              ordered H0–H7 session in <InlineCode>V1_2_FIELD_TEST_HANDOFF.md</InlineCode> before
              trusting an unattended cell.
            </p>
            <p>
              Teleop starts with synchronization disabled and performs no follower write; slow
              synchronization is a second explicit motion boundary. Inference Stop clears writes,
              safe-idles to gravity compensation, and releases the motion lease — an explicit
              Settings <strong>Disconnect</strong> is still what returns an arm to the
              de-energized, limp state.
            </p>
          </Step>
        </ol>
      </PageSection>

      <PageSection className="max-w-prose">
        <SectionHeading title="If something goes wrong" className="mb-4" />
        <ul className="space-y-3 text-[13px] leading-6 text-ink-secondary">
          <li>
            <strong className="text-ink">The banner says a connection needs attention.</strong> Open{" "}
            <Link to="/settings" className="font-medium text-accent-700 hover:text-accent-800">
              Settings
            </Link>{" "}
            — each service row states exactly which environment variable is missing.
          </li>
          <li>
            <strong className="text-ink">Preflight blocks an arm.</strong> The reason is per-arm:
            unresolved identity, a down link, a missing frame map or soft-limit file, or a checkout
            that does not match the pinned commit.
          </li>
          <li>
            <strong className="text-ink">Managed training or inference will not clean up.</strong>{" "}
            Use <InlineCode>make modal-panic</InlineCode>. It can stop paid resources and requires
            fresh operator approval.
          </li>
          <li>
            <strong className="text-ink">Deeper reference.</strong> See{" "}
            <InlineCode>docs/installation.mdx</InlineCode>,{" "}
            <InlineCode>docs/docker-deployment.md</InlineCode>,{" "}
            <InlineCode>docs/yam-setup.md</InlineCode>, and{" "}
            <InlineCode>docs/troubleshooting.md</InlineCode> in the repository.
          </li>
        </ul>
      </PageSection>
    </Page>
  );
}
