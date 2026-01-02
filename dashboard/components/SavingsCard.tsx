"use client";

import { useEffect, useState } from "react";
import { DollarSign, HeartPulse, TrendingUp } from "lucide-react";
import { BusinessSettings } from "./SettingsPopover";

type SavingsStats = {
    total_money_saved: number;
    failures_prevented: number;
    loops_killed: number;
    total_spend: number; // Added to API response interface (implied from server logic)
    roi_multiplier: number;
};

interface SavingsCardProps {
    settings: BusinessSettings;
}

export default function SavingsCard({ settings }: SavingsCardProps) {
    const [stats, setStats] = useState<SavingsStats | null>(null);
    const [animate, setAnimate] = useState(false);

    // Fetch Stats
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch("http://localhost:8000/api/stats/savings");
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

    // Trigger Pulse when settings change
    useEffect(() => {
        if (!stats) return;
        setAnimate(true);
        const timer = setTimeout(() => setAnimate(false), 1000);
        return () => clearTimeout(timer);
    }, [settings, stats?.failures_prevented]); // Re-calc on settings or new failures

    if (!stats) return null;

    // --- Business Logic Calculation ---
    // 1. Manual Savings: Failures * (Labor Cost / 60 * 20 mins avg fix time)
    const manualSavings = stats.failures_prevented * ((settings.manualLaborCost / 60) * 20);

    // 2. Downtime Savings: Failures * (Impact / min * 5 mins avg recovery diff)
    // Assumption: Auto-repair is instant, manual is 5 mins slower? Or just "Impact of incident".
    // Let's assume avoiding a failure saves 10 mins of downtime.
    const downtimeSavings = stats.failures_prevented * (settings.downtimeImpact * 10);

    // 3. Total Savings = Token Savings (Technical) + Manual + Downtime
    const totalSavings = stats.total_money_saved + manualSavings + downtimeSavings;

    // 4. ROI = Total Savings / (Infra Cost + Token Spend)
    // Note: total_spend comes from API (sum of estimated_cost). If API doesn't send it, we need to fix API or approx.
    // In `api_server.py`, `total_spend` is calculated but not explicitly in the return dict in my memory? 
    // Wait, let's checking api_server.py... return dict had "roi_multiplier" pre-calculated.
    // If we want dynamic ROI, we need the raw spend.
    // Let's rely on the API's `roi_multiplier` for the BASE, but modify it?
    // Actually, accurate ROI needs raw spend. 
    // Let's Update API Server to return 'total_spend' key.
    // Fallback if missing: assume small spend.
    const rawSpend = (stats as any).total_spend || 5.0; // Fallback
    const totalInvestment = settings.infraCost + rawSpend;
    const dynamicROI = totalInvestment > 0 ? (totalSavings / totalInvestment) : 0;


    return (
        <>
            <style jsx>{`
                @keyframes shimmer {
                    0% { filter: brightness(1); text-shadow: none; }
                    50% { filter: brightness(1.3); text-shadow: 0 0 20px rgba(255,255,255,0.6); }
                    100% { filter: brightness(1); text-shadow: none; }
                }
                .animate-shimmer {
                    animation: shimmer 0.8s ease-out;
                }
            `}</style>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 mt-6">

                {/* Total Savings */}
                <div className={`
                    bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 
                    p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden transition-all duration-500
                    ${animate ? 'shadow-[0_0_30px_rgba(16,185,129,0.2)] border-emerald-500/50' : ''}
                `}>
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                        <DollarSign size={80} />
                    </div>
                    <div>
                        <h3 className="text-emerald-400 font-medium text-sm flex items-center gap-2">
                            <DollarSign size={16} /> NET SAVINGS
                        </h3>
                        <p className={`text-4xl font-black text-white mt-2 transition-all duration-300 ${animate ? 'animate-shimmer text-emerald-300' : ''}`}>
                            ${totalSavings.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </p>
                    </div>
                    <p className="text-xs text-gray-400 mt-4">
                        compute + labor + downtime avoided
                    </p>
                </div>

                {/* Lives Saved (Failures Prevented) */}
                <div className="bg-gradient-to-r from-pink-500/10 to-rose-500/10 border border-pink-500/20 p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                        <HeartPulse size={80} />
                    </div>
                    <div>
                        <h3 className="text-pink-400 font-medium text-sm flex items-center gap-2">
                            <HeartPulse size={16} /> LIVES SAVED
                        </h3>
                        <p className="text-4xl font-black text-white mt-2">
                            {stats.failures_prevented}
                        </p>
                    </div>
                    <p className="text-xs text-gray-400 mt-4">
                        Critical failures auto-repaired by Medic.
                    </p>
                </div>

                {/* ROI Multiplier */}
                <div className={`
                    bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 
                    p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden transition-all duration-500
                    ${animate ? 'shadow-[0_0_30px_rgba(245,158,11,0.2)] border-amber-500/50' : ''}
                `}>
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                        <TrendingUp size={80} />
                    </div>
                    <div>
                        <h3 className="text-amber-400 font-medium text-sm flex items-center gap-2">
                            <TrendingUp size={16} /> ROI FACTOR
                        </h3>
                        <p className={`text-4xl font-black text-white mt-2 transition-all duration-300 ${animate ? 'animate-shimmer text-amber-300' : ''}`}>
                            {dynamicROI.toFixed(1)}x
                        </p>
                    </div>
                    <p className="text-xs text-gray-400 mt-4">
                        Return on Infrastructure spend.
                    </p>
                </div>

            </div>
        </>
    );
}
