import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export function Disclaimer({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-card border border-warning/30 bg-warning/10 px-4 py-3 text-xs text-warning",
        className,
      )}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>AI second opinion only. Final diagnosis subject to clinical judgment.</span>
    </div>
  );
}
