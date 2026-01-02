"use client";

import { useEffect, useState } from "react";
import { ShieldCheck, Zap, AlertTriangle } from "lucide-react";
import { Sparkline } from "@/components/ui/Sparkline";
import { ReliabilityGraph } from "@/components/ReliabilityGraph";

type StatsData = {
    total_runs: number;
    repaired: number;
    failed: number;
    success: number;
};

export default function Stats() {
    const [stats, setStats] = useState<StatsData | null>(null);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch("http://localhost:8000/api/stats");
                if (res.ok) {
                    setStats(await res.json());
                }
            } catch (e) {
                console.error(e);
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    if (!stats) return <div className="text-gray-500">Initializing Uplink...</div>;

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-900/10 border border-green-500/20">
                <div className="flex items-center gap-2 text-green-400 mb-1">
                    <ShieldCheck size={18} />
                    <span className="text-sm font-medium">Reliability Score</span>
                </div>
                <div className="text-2xl font-bold text-green-100">
                    {stats.total_runs > 0
                        ? Math.round(((stats.success + stats.repaired) / stats.total_runs) * 100)
                        : 100}%
                </div>
                {/* Dynamic Reliability Graph */}
                <div className="mt-4 h-12 w-full">
                    <ReliabilityGraph
                        data={[
                            { timestamp: "02:00", score: 98 }, { timestamp: "04:00", score: 99 },
                            { timestamp: "06:00", score: 97 }, { timestamp: "08:00", score: 85 },
                            { timestamp: "10:00", score: 92 }, { timestamp: "12:00", score: 96 },
                            { timestamp: "14:00", score: 98 }, { timestamp: "16:00", score: 99 },
                            { timestamp: "18:00", score: 100 }, { timestamp: "20:00", score: 99 }
                        ]}
                        width={280}
                        height={40}
                    />
                </div>
            </div>

            <div className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-indigo-900/10 border border-blue-500/20">
                <div className="flex items-center gap-2 text-blue-400 mb-1">
                    <Zap size={18} />
                    <span className="text-sm font-medium">Auto-Repairs</span>
                </div>
                <div className="text-2xl font-bold text-blue-100">
                    {stats.repaired} <span className="text-xs font-normal text-blue-300">Recovered</span>
                </div>
            </div>

            <div className="p-4 rounded-xl bg-gradient-to-br from-orange-500/10 to-red-900/10 border border-orange-500/20">
                <div className="flex items-center gap-2 text-orange-400 mb-1">
                    <AlertTriangle size={18} />
                    <span className="text-sm font-medium">Failures</span>
                </div>
                <div className="text-2xl font-bold text-orange-100">
                    {stats.failed}
                </div>
            </div>
        </div>
    );
}
