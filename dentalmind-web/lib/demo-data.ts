export type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
export type Modality = "bitewing" | "panoramic" | "periapical";

export interface Finding {
  class_name: string;
  confidence: number;
  surface?: string;
  bone_loss_mm?: number;
}

export interface TreatmentOption {
  rank: number;
  text: string;
}

export interface Cluster {
  tooth_fdi: string;
  severity: Severity;
  urgency: string;
  pattern: string;
  findings: Finding[];
  prompts: {
    options: TreatmentOption[];
    tests: string[];
    red_flags: string[];
  };
}

export interface Overlay {
  tooth: string;
  x: number;
  y: number;
  type: string;
  color: string;
}

export interface DemoResult {
  modality: Modality;
  image: string;
  clusters: Cluster[];
  overlays: Overlay[];
}

export const demoResults: Record<Modality, DemoResult> = {
  bitewing: {
    modality: "bitewing",
    image: "/samples/bitewing.svg",
    clusters: [
      {
        tooth_fdi: "26",
        severity: "HIGH",
        urgency: "soon_2_weeks",
        pattern: "caries_with_bone",
        findings: [
          { class_name: "dentin_caries", confidence: 0.91, surface: "mesial" },
          { class_name: "bone_loss", confidence: 0.88, bone_loss_mm: 3.2 },
        ],
        prompts: {
          options: [
            { rank: 1, text: "Pulp vitality test before restoring" },
            { rank: 2, text: "Composite restoration if vital + asymptomatic" },
            { rank: 3, text: "Root canal treatment if non-vital" },
          ],
          tests: ["Cold vitality test", "Percussion test"],
          red_flags: ["Spontaneous pain → urgent RCT", "Swelling → immediate"],
        },
      },
      {
        tooth_fdi: "27",
        severity: "LOW",
        urgency: "routine",
        pattern: "calculus_only",
        findings: [{ class_name: "calculus", confidence: 0.76, surface: "distal" }],
        prompts: {
          options: [
            { rank: 1, text: "Scaling and polishing at next hygiene visit" },
            { rank: 2, text: "Reinforce interdental cleaning technique" },
          ],
          tests: ["Periodontal probing"],
          red_flags: ["Bleeding on probing → periodontal referral"],
        },
      },
      {
        tooth_fdi: "36",
        severity: "MEDIUM",
        urgency: "soon_4_weeks",
        pattern: "enamel_caries",
        findings: [{ class_name: "enamel_caries", confidence: 0.82, surface: "occlusal" }],
        prompts: {
          options: [
            { rank: 1, text: "Fluoride varnish + diet counseling" },
            { rank: 2, text: "Sealant or preventive resin restoration" },
          ],
          tests: ["Transillumination", "Bitewing follow-up in 6 months"],
          red_flags: ["Progression to dentin on follow-up → restore"],
        },
      },
    ],
    overlays: [
      { tooth: "26", x: 45, y: 32, type: "caries", color: "#EF4444" },
      { tooth: "26", x: 47, y: 38, type: "bone_loss", color: "#F59E0B" },
      { tooth: "27", x: 58, y: 36, type: "calculus", color: "#60A5FA" },
      { tooth: "36", x: 44, y: 64, type: "caries", color: "#EF4444" },
    ],
  },
  panoramic: {
    modality: "panoramic",
    image: "/samples/panoramic.svg",
    clusters: [
      {
        tooth_fdi: "18",
        severity: "CRITICAL",
        urgency: "immediate",
        pattern: "impaction_with_lesion",
        findings: [
          { class_name: "impacted_tooth", confidence: 0.95 },
          { class_name: "periapical_lesion", confidence: 0.84 },
        ],
        prompts: {
          options: [
            { rank: 1, text: "Oral surgery referral for extraction" },
            { rank: 2, text: "CBCT for 3D root + nerve proximity assessment" },
            { rank: 3, text: "Antibiotics if active infection present" },
          ],
          tests: ["CBCT", "Sensibility testing on adjacent tooth 17"],
          red_flags: ["Trismus or facial swelling → same-day referral"],
        },
      },
      {
        tooth_fdi: "46",
        severity: "HIGH",
        urgency: "soon_2_weeks",
        pattern: "caries_with_bone",
        findings: [
          { class_name: "dentin_caries", confidence: 0.89, surface: "occlusal" },
          { class_name: "bone_loss", confidence: 0.81, bone_loss_mm: 2.6 },
        ],
        prompts: {
          options: [
            { rank: 1, text: "Periodontal charting + scaling" },
            { rank: 2, text: "Restorative treatment of carious lesion" },
            { rank: 3, text: "RCT if pulp involvement confirmed" },
          ],
          tests: ["Pulp vitality test", "Periodontal probing"],
          red_flags: ["Mobility grade II+ → periodontal referral"],
        },
      },
      {
        tooth_fdi: "11",
        severity: "MEDIUM",
        urgency: "soon_4_weeks",
        pattern: "restoration_margin_defect",
        findings: [{ class_name: "restoration_defect", confidence: 0.79, surface: "mesial" }],
        prompts: {
          options: [
            { rank: 1, text: "Replace restoration margin" },
            { rank: 2, text: "Monitor with bitewing in 6 months if asymptomatic" },
          ],
          tests: ["Visual-tactile exam", "Transillumination"],
          red_flags: ["Recurrent decay under restoration → replace immediately"],
        },
      },
    ],
    overlays: [
      { tooth: "18", x: 14, y: 35, type: "impaction", color: "#EF4444" },
      { tooth: "18", x: 16, y: 42, type: "lesion", color: "#FACC15" },
      { tooth: "46", x: 70, y: 66, type: "caries", color: "#EF4444" },
      { tooth: "46", x: 72, y: 60, type: "bone_loss", color: "#F59E0B" },
      { tooth: "11", x: 47, y: 30, type: "restoration", color: "#60A5FA" },
    ],
  },
  periapical: {
    modality: "periapical",
    image: "/samples/periapical.svg",
    clusters: [
      {
        tooth_fdi: "21",
        severity: "CRITICAL",
        urgency: "immediate",
        pattern: "periapical_lesion",
        findings: [
          { class_name: "periapical_lesion", confidence: 0.93 },
          { class_name: "dentin_caries", confidence: 0.87, surface: "distal" },
        ],
        prompts: {
          options: [
            { rank: 1, text: "Root canal treatment" },
            { rank: 2, text: "Drainage if acute abscess" },
            { rank: 3, text: "Extraction if non-restorable" },
          ],
          tests: ["Cold vitality test", "Percussion test", "Palpation"],
          red_flags: ["Facial swelling or fever → urgent same-day care"],
        },
      },
    ],
    overlays: [
      { tooth: "21", x: 50, y: 38, type: "lesion", color: "#FACC15" },
      { tooth: "21", x: 48, y: 50, type: "caries", color: "#EF4444" },
    ],
  },
};

export const severityMeta: Record<Severity, { label: string; badge: "danger" | "warning" | "cyan" | "success" }> = {
  CRITICAL: { label: "CRITICAL", badge: "danger" },
  HIGH: { label: "HIGH PRIORITY", badge: "warning" },
  MEDIUM: { label: "MEDIUM", badge: "cyan" },
  LOW: { label: "LOW", badge: "success" },
};

export const urgencyLabel: Record<string, string> = {
  immediate: "Immediate",
  soon_2_weeks: "2 weeks",
  soon_4_weeks: "4 weeks",
  routine: "Routine",
};
