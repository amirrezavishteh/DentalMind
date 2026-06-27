"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileImage } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Modality } from "@/lib/demo-data";

interface UploadZoneProps {
  onAnalyze: (modality: Modality) => void;
  loading: boolean;
}

const MODALITY_OPTIONS: { value: Modality | "auto"; label: string }[] = [
  { value: "auto", label: "Auto-detect modality" },
  { value: "bitewing", label: "Bitewing" },
  { value: "panoramic", label: "Panoramic (OPG)" },
  { value: "periapical", label: "Periapical" },
];

export function UploadZone({ onAnalyze, loading }: UploadZoneProps) {
  const [fileName, setFileName] = useState<string | null>(null);
  const [modality, setModality] = useState<Modality | "auto">("auto");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;
    setFileName(files[0].name);
  }, []);

  const resolveModality = (): Modality => {
    if (modality !== "auto") return modality;
    return "bitewing";
  };

  return (
    <div className="rounded-card-lg border border-border bg-surface p-8">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-card border-2 border-dashed border-border px-6 py-16 text-center transition-colors",
          dragOver && "border-primary bg-primary/5",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.dcm"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {fileName ? (
          <>
            <FileImage className="h-10 w-10 text-primary" />
            <p className="text-sm font-medium text-text">{fileName}</p>
            <p className="text-xs text-text-muted">Click to choose a different file</p>
          </>
        ) : (
          <>
            <UploadCloud className="h-10 w-10 text-text-muted" />
            <p className="text-sm font-medium text-text">
              Drag &amp; drop a radiograph, or click to browse
            </p>
            <p className="text-xs text-text-muted">Supports JPG, PNG, DICOM (.dcm)</p>
          </>
        )}
      </div>

      <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <Select value={modality} onValueChange={(v) => setModality(v as Modality | "auto")}>
          <SelectTrigger className="sm:w-64">
            <SelectValue placeholder="Modality" />
          </SelectTrigger>
          <SelectContent>
            {MODALITY_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant="primary"
          size="lg"
          disabled={!fileName || loading}
          onClick={() => onAnalyze(resolveModality())}
          className="sm:w-48"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
      </div>
    </div>
  );
}
