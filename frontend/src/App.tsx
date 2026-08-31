import {
  Bot,
  BookOpen,
  BrainCircuit,
  Database,
  LayoutGrid,
  Menu,
  Package,
  RadioTower,
  Settings,
  Video,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { SetupBanner } from "./components/SetupBanner";
import { IconButton } from "./components/ui/Button";
import { fetchYamSetup } from "./lib/api";
import { DatasetEpisodePage } from "./pages/DatasetEpisodePage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { InferenceDeploymentPage } from "./pages/InferenceDeploymentPage";
import { InferenceNewPage } from "./pages/InferenceNewPage";
import { InferencePage } from "./pages/InferencePage";
import { ModelsPage } from "./pages/ModelsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { RecordPage } from "./pages/RecordPage";
import { RobotDetailPage } from "./pages/RobotDetailPage";
import { RobotsPage } from "./pages/RobotsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SetupGuidePage } from "./pages/SetupGuidePage";
import { TrainingPage } from "./pages/TrainingPage";
import { TrainingRunPage } from "./pages/TrainingRunPage";
import { SystemStatusProvider, useSystemStatus } from "./state/systemStatus";

type NavigationItem = {
  label: string;
  path: string;
  icon: LucideIcon;
};

type NavigationGroup = {
  label: string;
  items: NavigationItem[];
};

const navigationGroups: NavigationGroup[] = [
  {
    label: "Operate",
    items: [
      { label: "Overview", path: "/overview", icon: LayoutGrid },
      { label: "Robots", path: "/robots", icon: Bot },
      { label: "Record", path: "/record", icon: Video },
    ],
  },
  {
    label: "Build",
    items: [
      { label: "Datasets", path: "/datasets", icon: Database },
      { label: "Training", path: "/training", icon: BrainCircuit },
      { label: "Models", path: "/models", icon: Package },
    ],
  },
  {
    label: "Deploy",
    items: [{ label: "Inference", path: "/inference", icon: RadioTower }],
  },
  {
    label: "Admin",
    items: [
      { label: "Set up", path: "/setup", icon: BookOpen },
      { label: "Settings", path: "/settings", icon: Settings },
    ],
  },
];

function ShellNavLink({ item, onNavigate }: { item: NavigationItem; onNavigate?: () => void }) {
  const Icon = item.icon;
  return (
    <NavLink
      to={item.path}
      onClick={onNavigate}
      className={({ isActive }) =>
        [
          "flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition",
          isActive
            ? "bg-surface text-ink shadow-xs ring-1 ring-line"
            : "text-ink-muted hover:bg-surface/70 hover:text-ink",
        ].join(" ")
      }
    >
      <Icon className="h-4 w-4 shrink-0" strokeWidth={1.8} aria-hidden="true" />
      {item.label}
    </NavLink>
  );
}

function ModeIndicator() {
  const { status } = useSystemStatus();
  return (
    <div className="flex items-center gap-2 px-2.5 text-2xs text-ink-muted">
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full ${
          status?.mode === "mock"
            ? "bg-accent-500"
            : status
              ? "bg-positive-500"
              : "bg-ink-faint"
        }`}
      />
      {status ? `${status.mode === "mock" ? "Mock" : "Hardware"} mode` : "Checking mode"}
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="flex h-10 items-center gap-2.5 px-2">
        <div className="grid h-7 w-7 place-items-center rounded-lg bg-ink text-sm font-semibold text-white">
          π
        </div>
        <div className="text-[13px] font-semibold tracking-tight text-ink">ctrl-π</div>
      </div>

      <nav aria-label="Primary" className="mt-8 flex-1 space-y-6 overflow-y-auto">
        {navigationGroups.map((group) => (
          <div key={group.label}>
            <p className="px-2.5 pb-1.5 text-2xs font-medium uppercase tracking-[0.12em] text-ink-faint">
              {group.label}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => (
                <ShellNavLink key={item.path} item={item} onNavigate={onNavigate} />
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="mt-6 border-t border-line pt-4">
        <ModeIndicator />
      </div>
    </>
  );
}

function AppShell() {
  const location = useLocation();
  const { status: settingsStatus, loading: settingsLoading, refresh: refreshSettings } =
    useSystemStatus();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const armsServiceStatus = settingsStatus?.services.find(
    (service) => service.id === "arms",
  )?.status;
  const postgresServiceStatus = settingsStatus?.services.find(
    (service) => service.id === "postgres",
  )?.status;
  const refreshInFlight = useRef(false);

  const closeMobileNav = useCallback(() => setMobileNavOpen(false), []);

  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (
      location.pathname === "/settings"
      || settingsStatus?.mode !== "hardware"
      || postgresServiceStatus !== "connected"
      || armsServiceStatus === "connected"
    ) return;

    let stopped = false;
    let timer: number | undefined;
    let controller: AbortController | null = null;

    const schedule = (delay = 2_000) => {
      window.clearTimeout(timer);
      if (stopped || document.visibilityState !== "visible") return;
      timer = window.setTimeout(() => void poll(), delay);
    };
    const poll = async () => {
      if (stopped || controller || document.visibilityState !== "visible") return;
      controller = new AbortController();
      let continuePolling = false;
      try {
        const yam = await fetchYamSetup(controller.signal);
        if (yam.connected || yam.state === "error") {
          if (
            !refreshInFlight.current
            && (yam.connected || armsServiceStatus !== "error")
          ) {
            refreshInFlight.current = true;
            refreshSettings();
          }
          return;
        }
        continuePolling = yam.saved && yam.auto_restore;
      } catch {
        continuePolling = true;
      } finally {
        controller = null;
        if (continuePolling) schedule();
      }
    };
    const visibilityChanged = () => {
      if (document.visibilityState === "visible") schedule(0);
      else window.clearTimeout(timer);
    };

    document.addEventListener("visibilitychange", visibilityChanged);
    schedule();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [armsServiceStatus, location.pathname, postgresServiceStatus, refreshSettings, settingsStatus?.mode]);

  useEffect(() => {
    if (!settingsLoading) refreshInFlight.current = false;
  }, [settingsLoading]);

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-60 flex-col border-r border-line bg-canvas px-4 py-5 lg:flex">
        <SidebarContent />
      </aside>

      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-line bg-canvas/90 px-4 backdrop-blur lg:hidden">
        <div className="flex items-center gap-2 text-[13px] font-semibold">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-ink text-xs text-white">
            π
          </span>
          ctrl-π
        </div>
        <IconButton
          icon={Menu}
          label="Open navigation"
          variant="ghost"
          onClick={() => setMobileNavOpen(true)}
        />
      </header>

      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            aria-hidden="true"
            onClick={closeMobileNav}
            className="absolute inset-0 animate-fade-in bg-ink/25"
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Navigation"
            className="relative flex h-full w-64 flex-col border-r border-line bg-canvas px-4 py-5"
          >
            <div className="absolute right-3 top-4">
              <IconButton icon={X} label="Close navigation" variant="ghost" size="sm" onClick={closeMobileNav} />
            </div>
            <SidebarContent onNavigate={closeMobileNav} />
          </div>
        </div>
      )}

      <main className="lg:ml-60">
        <SetupBanner />
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/robots" element={<RobotsPage />} />
          <Route path="/robots/:robotId" element={<RobotDetailPage />} />
          <Route path="/arms" element={<Navigate to="/robots" replace />} />
          <Route path="/record" element={<RecordPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:repoName" element={<DatasetEpisodePage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/:runId" element={<TrainingRunPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/inference/new" element={<InferenceNewPage />} />
          <Route path="/inference/:deploymentId" element={<InferenceDeploymentPage />} />
          <Route path="/setup" element={<SetupGuidePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export function App() {
  return (
    <SystemStatusProvider>
      <AppShell />
    </SystemStatusProvider>
  );
}
