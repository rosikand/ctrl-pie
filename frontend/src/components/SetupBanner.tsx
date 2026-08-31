import { ArrowRight, TriangleAlert } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

import { useSystemStatus } from "../state/systemStatus";

/**
 * One line above the page when a required connection is missing. It never grows
 * into a second dashboard: it states the problem and links to the fix.
 */
export function SetupBanner() {
  const location = useLocation();
  const { status, error } = useSystemStatus();

  if (location.pathname === "/settings" || (!error && (!status || status.setup_complete))) {
    return null;
  }

  const count = status?.services.filter(
    (service) => service.required && !["connected", "configured"].includes(service.status),
  ).length;
  const armsNeedAttention = status?.services.some(
    (service) =>
      service.id === "arms" &&
      service.required &&
      !["connected", "configured"].includes(service.status),
  ) ?? false;

  return (
    <div className="border-b border-caution-100 bg-caution-50 px-6 py-2.5 sm:px-8 lg:px-12">
      <div className="mx-auto flex max-w-page items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 text-xs text-caution-700">
          <TriangleAlert className="h-4 w-4 shrink-0 text-caution-600" aria-hidden="true" />
          <span>
            {error
              ? "The backend is unavailable. Start it to finish setup."
              : `${count ?? "Some"} required connection${count === 1 ? "" : "s"} need attention.`}
          </span>
        </div>
        <Link
          to={armsNeedAttention ? "/settings#yam-setup" : "/settings"}
          className="flex shrink-0 items-center gap-1 text-xs font-medium text-caution-700 hover:text-caution-600"
        >
          {armsNeedAttention ? "Set up YAMs" : "Open checklist"}
          <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
        </Link>
      </div>
    </div>
  );
}
