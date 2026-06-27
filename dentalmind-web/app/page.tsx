import { Hero } from "@/components/home/Hero";
import { ProblemSection } from "@/components/home/ProblemSection";
import { MadClipSection } from "@/components/home/MadClipSection";
import { ModalitiesSection } from "@/components/home/ModalitiesSection";
import { StatsSection } from "@/components/home/StatsSection";
import { ComparisonTable } from "@/components/home/ComparisonTable";
import { CTABanner } from "@/components/home/CTABanner";

export default function Home() {
  return (
    <main>
      <Hero />
      <ProblemSection />
      <MadClipSection />
      <ModalitiesSection />
      <StatsSection />
      <ComparisonTable />
      <CTABanner />
    </main>
  );
}
