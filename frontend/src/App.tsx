import {
  Bot,
  BrainCircuit,
  Database,
  RadioTower,
  Settings,
  Video,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { SetupBanner } from "./components/SetupBanner";
import { fetchSettingsStatus, type SettingsStatus } from "./lib/api";
import { ArmsPage } from "./pages/ArmsPage";
import { DatasetEpisodePage } from "./pages/DatasetEpisodePage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { RecordPage } from "./pages/RecordPage";
import { SettingsPage } from "./pages/SettingsPage";

type NavigationItem = {
  label: string;
  mobileLabel?: string;
  path: string;
  icon: LucideIcon;
};

const primaryNavigation: NavigationItem[] = [
  { label: "Arms", path: "/arms", icon: Bot },
  { label: "Record / Teleop", mobileLabel: "Record", path: "/record", icon: Video },
  { label: "Datasets", path: "/datasets", icon: Database },
  { label: "Training", path: "/training", icon: BrainCircuit },
  { label: "Inference", path: "/inference", icon: RadioTower },
];

const pageContent: Record<
  string,
  { eyebrow: string; title: string; description: string; icon: LucideIcon }
> = {
  training: {
    eyebrow: "Experiments",
    title: "Training",
    description: "Track external training runs, metrics, checkpoints, and model revisions.",
    icon: BrainCircuit,
  },
  inference: {
    eyebrow: "Deployment",
    title: "Inference",
    description: "Deploy policies to Modal and execute action chunks on connected arms.",
    icon: RadioTower,
  },
};

function ShellNavLink({ item }: { item: NavigationItem }) {
  const Icon = item.icon;

  return (
    <NavLink
      to={item.path}
      className={({ isActive }) =>
        [
          "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition",
          isActive
            ? "bg-white text-ink shadow-sm ring-1 ring-slate-200/80"
            : "text-slate-500 hover:bg-white/70 hover:text-slate-900",
        ].join(" ")
      }
    >
      <Icon className="h-[18px] w-[18px]" strokeWidth={1.8} />
      <span>{item.label}</span>
    </NavLink>
  );
}

function AppShell() {
  const [settingsStatus, setSettingsStatus] = useState<SettingsStatus | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(true);

  const refreshSettings = useCallback(() => {
    setSettingsLoading(true);
    setSettingsError(null);
    void fetchSettingsStatus()
      .then(setSettingsStatus)
      .catch((reason: unknown) => {
        setSettingsError(reason instanceof Error ? reason.message : "Could not reach the backend.");
      })
      .finally(() => setSettingsLoading(false));
  }, []);

  useEffect(() => {
    refreshSettings();
  }, [refreshSettings]);

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-60 border-r border-slate-200/80 bg-slate-50/90 px-4 py-5 backdrop-blur lg:flex lg:flex-col">
        <div className="flex h-10 items-center gap-2.5 px-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-ink text-lg font-semibold text-white">π</div>
          <div>
            <div className="text-[15px] font-semibold tracking-tight">ctrl-π</div>
            <div className="text-[10px] font-medium uppercase tracking-[0.16em] text-slate-400">Robot learning</div>
          </div>
        </div>

        <nav aria-label="Primary" className="mt-8 space-y-1">
          {primaryNavigation.map((item) => (
            <ShellNavLink key={item.path} item={item} />
          ))}
        </nav>

        <div className="mt-auto border-t border-slate-200/80 pt-4">
          <NavLink
            to="/settings"
            aria-label="Settings"
            className={({ isActive }) =>
              [
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition",
                isActive
                  ? "bg-white text-ink shadow-sm ring-1 ring-slate-200/80"
                  : "text-slate-500 hover:bg-white hover:text-slate-900",
              ].join(" ")
            }
          >
            <Settings className="h-[18px] w-[18px]" strokeWidth={1.8} />
            Settings
          </NavLink>
          <div className="mt-3 flex items-center gap-2 px-3 text-xs text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            Mock mode
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-10 flex h-16 items-center justify-between border-b border-slate-200 bg-white/90 px-4 backdrop-blur lg:hidden">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-ink text-white">π</span>
          ctrl-π
        </div>
        <NavLink to="/settings" aria-label="Settings" className="rounded-md p-2 text-slate-500 hover:bg-slate-100">
          <Settings className="h-5 w-5" />
        </NavLink>
      </header>

      <nav aria-label="Mobile primary" className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-5 border-t border-slate-200 bg-white px-1 py-1.5 lg:hidden">
        {primaryNavigation.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex flex-col items-center gap-1 rounded-md py-1 text-[10px] font-medium ${isActive ? "text-brand-600" : "text-slate-400"}`
              }
            >
              <Icon className="h-5 w-5" strokeWidth={1.8} />
              {item.mobileLabel ?? item.label}
            </NavLink>
          );
        })}
      </nav>

      <main className="pb-24 lg:ml-60 lg:pb-0">
        <SetupBanner status={settingsStatus} error={settingsError} />
        <Routes>
          <Route path="/" element={<Navigate to="/arms" replace />} />
          <Route path="/arms" element={<ArmsPage />} />
          <Route path="/record" element={<RecordPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:repoName" element={<DatasetEpisodePage />} />
          {Object.entries(pageContent).map(([key, content]) => (
            <Route key={key} path={`/${key}`} element={<PlaceholderPage {...content} />} />
          ))}
          <Route
            path="/settings"
            element={
              <SettingsPage
                status={settingsStatus}
                loading={settingsLoading}
                error={settingsError}
                refresh={refreshSettings}
              />
            }
          />
          <Route path="*" element={<Navigate to="/arms" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function PlaceholderPage({
  eyebrow,
  title,
  description,
  icon: Icon,
}: {
  eyebrow: string;
  title: string;
  description: string;
  icon: LucideIcon;
}) {
  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex items-start justify-between gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">{eyebrow}</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">{title}</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">{description}</p>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500 shadow-sm sm:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Mock mode
        </div>
      </header>

      <section className="mt-8 min-h-[360px] rounded-xl border border-slate-200 bg-white shadow-panel">
        <div className="grid min-h-[360px] place-items-center px-6 text-center">
          <div>
            <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
              <Icon className="h-5 w-5" strokeWidth={1.7} />
            </div>
            <p className="mt-4 text-sm font-medium text-slate-800">Ready for setup</p>
            <p className="mx-auto mt-1 max-w-sm text-sm leading-6 text-slate-400">
              This workspace will appear here as its service is configured.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export function App() {
  return <AppShell />;
}
