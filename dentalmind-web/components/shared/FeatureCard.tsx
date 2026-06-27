import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  accent?: "cyan" | "purple" | "success" | "warning";
  className?: string;
}

const accentMap = {
  cyan: "text-primary bg-primary/10 border-primary/30",
  purple: "text-secondary bg-secondary/10 border-secondary/30",
  success: "text-success bg-success/10 border-success/30",
  warning: "text-warning bg-warning/10 border-warning/30",
};

export function FeatureCard({
  icon: Icon,
  title,
  description,
  accent = "cyan",
  className,
}: FeatureCardProps) {
  return (
    <div
      className={cn(
        "rounded-card-lg border border-border bg-surface p-6 transition-all duration-300 hover:-translate-y-1 hover:border-primary/40",
        className,
      )}
    >
      <div className={cn("mb-4 flex h-11 w-11 items-center justify-center rounded-card border", accentMap[accent])}>
        <Icon className="h-5 w-5" />
      </div>
      <h3 className="mb-2 text-lg font-bold text-text">{title}</h3>
      <p className="text-sm leading-relaxed text-text-muted">{description}</p>
    </div>
  );
}
