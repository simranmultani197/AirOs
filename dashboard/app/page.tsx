"use client";

import { useState, useEffect } from "react";
import Pulse from "@/components/Pulse";
import Stats from "@/components/Stats";
import SavingsCard from "@/components/SavingsCard";
import MTTRCard from "@/components/MTTRCard";
import FailureRootCausesChart from "@/components/FailureRootCausesChart";
import ConfidenceWidget from "@/components/ConfidenceWidget";
import SettingsPopover, { BusinessSettings } from "@/components/SettingsPopover";
import { Badge } from "@/components/ui/Badge";
import { Sparkline } from "@/components/ui/Sparkline";

export default function Home() {
  // Default Business Logic Settings
  const [settings, setSettings] = useState<BusinessSettings>({
    manualLaborCost: 150, // $150/hr Sr Engineer
    infraCost: 50,      // $50/mo trivial
    downtimeImpact: 100 // $100/min business impact
  });

  useEffect(() => {
    fetch("http://localhost:8000/api/settings")
      .then(res => res.json())
      .then(data => {
        setSettings(prev => ({
          ...prev,
          manualLaborCost: data.manual_labor_cost,
          infraCost: data.infrastructure_rate
        }));
      })
      .catch(err => console.error("Failed to load settings", err));
  }, []);

  return (
    <main className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-teal-400 tracking-tight">
                air-os
              </h1>
              <SettingsPopover settings={settings} onUpdate={setSettings} />
            </div>
            <p className="text-gray-400 mt-1 font-medium">Reliability Control Center</p>
          </div>
          <div className="flex items-center gap-4">
            <Badge label="Sentinel Active" variant="sentinel" state="active" />
            <Badge label="Medic Processing" variant="medic" state="processing" />
            <Sparkline label="Health" />
          </div>
        </header>

        {/* Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

          {/* Main Column (2/3): Stats and Charts */}
          <div className="lg:col-span-2 space-y-6">
            <SavingsCard settings={settings} />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-3">
                <Stats />
              </div>
            </div>
            <FailureRootCausesChart />
          </div>

          {/* Side Column (1/3): Metrics and Widgets */}
          <div className="space-y-6">
            <ConfidenceWidget />
            <MTTRCard />
          </div>

        </div>

        {/* Bottom Row: Live Pulse */}
        <Pulse />

      </div>
    </main>
  );
}
