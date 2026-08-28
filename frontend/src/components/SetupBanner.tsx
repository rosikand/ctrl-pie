import { AlertTriangle, ArrowRight } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

import type { SettingsStatus } from "../lib/api";

export function SetupBanner({
  status,
  error,
}: {
  status: SettingsStatus | null;
  error: string | null;
}) {
  const location = useLocation();
  if (location.pathname === "/settings" || (!error && (!status || status.setup_complete))) {
    return null;
  }

  const count = status?.services.filter(
    (service) => service.required && !["connected", "configured"].includes(service.status),
  ).length;

  return (
    <div className="border-b border-amber-200 bg-amber-50 px-5 py-3 sm:px-8 lg:px-10">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
        <div className="flex items-center gap-3 text-sm text-amber-950">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
          <span>
            {error
              ? "The backend is unavailable. Start it to finish setup."
              : `${count ?? "Some"} required connection${count === 1 ? "" : "s"} need attention.`}
          </span>
        </div>
        <Link
          to="/settings"
          className="flex shrink-0 items-center gap-1 text-xs font-semibold text-amber-800 hover:text-amber-950"
        >
          Open checklist
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
    </div>
  );
}

