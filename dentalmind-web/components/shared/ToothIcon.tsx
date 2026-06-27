import { cn } from "@/lib/utils";

export function ToothIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-12 w-12", className)}
    >
      <path
        d="M32 6c-6 0-9 4-13 4-5 0-9 4-9 11 0 5 2 8 2 14 0 8 4 23 9 23 4 0 4-12 7-12s3 12 7 12c5 0 9-15 9-23 0-6 2-9 2-14 0-7-4-11-9-11-4 0-7-4-13-4z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        fill="rgba(0,212,255,0.06)"
      />
    </svg>
  );
}
