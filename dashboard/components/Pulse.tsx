"use client";

import { useEffect, useState } from "react";
import { Activity, MoreHorizontal } from "lucide-react";
import { StatusTag } from "@/components/ui/StatusTag";
import MedicInsightDrawer from "@/components/MedicInsightDrawer";

type Trace = {
    id: number;
    run_id: string;
    node_id: string;
    status: string;
    timestamp: string;
    input_state: any;
    output_state: any;
    recovery_attempts: number;
};

export default function Pulse() {
    const [traces, setTraces] = useState<Trace[]>([]);
    const [selectedTrace, setSelectedTrace] = useState<Trace | null>(null);
    const [lastSync, setLastSync] = useState<string>("");

    const [isLoading, setIsLoading] = useState(true);

    const fetchTraces = async () => {
        try {
            // Keep existing traces while fetching update
            // On first load, isLoading is true.
            const res = await fetch("http://localhost:8000/runs?limit=20");
            if (res.ok) {
                setTraces(await res.json());
                setLastSync(new Date().toLocaleTimeString());
            }
        } catch (e) {
            console.error("Failed to fetch traces", e);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchTraces();
        const interval = setInterval(fetchTraces, 5000); // 5 seconds polling
        return () => clearInterval(interval);
    }, []);

    // Skeleton Row Component
    const SkeletonRow = () => (
        <tr className="border-b border-white/5 animate-pulse">
            <td className="px-4 py-3"><div className="h-4 w-24 bg-white/10 rounded"></div></td>
            <td className="px-4 py-3"><div className="h-4 w-32 bg-white/10 rounded"></div></td>
            <td className="px-4 py-3"><div className="h-4 w-20 bg-white/10 rounded"></div></td>
            <td className="px-4 py-3"><div className="h-4 w-16 bg-white/10 rounded"></div></td>
            <td className="px-4 py-3 text-right"><div className="h-4 w-8 bg-white/10 rounded ml-auto"></div></td>
        </tr>
    );

    const [isDrawerLoading, setIsDrawerLoading] = useState(false);

    // Fetch Details on Row Click
    const handleRowClick = async (trace: Trace) => {
        setIsDrawerLoading(true);
        setSelectedTrace(trace); // Open modal immediately with stale/loading state

        try {
            // "Using the run_id of the selected row"
            const res = await fetch(`http://localhost:8000/api/runs/${trace.run_id}`);
            if (res.ok) {
                const detailData = await res.json();
                // API returns an list of traces (history). The Drawer expects a single trace object.
                // We'll use the latest trace (last item) as the current state,
                // and we could potentially pass the full history if the Drawer supported it.
                if (Array.isArray(detailData) && detailData.length > 0) {
                    setSelectedTrace(detailData[detailData.length - 1]);
                } else {
                    setSelectedTrace(detailData);
                }
            }
        } catch (e) {
            console.error("Failed to fetch run details", e);
        } finally {
            setIsDrawerLoading(false);
        }
    };

    return (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm mt-8 relative">
            <style jsx>{`
                @keyframes slidePurple {
                    0% { transform: translateY(-15px); opacity: 0; background-color: rgba(168, 85, 247, 0.2); }
                    100% { transform: translateY(0); opacity: 1; background-color: transparent; }
                }
                .animate-slide-purple {
                    animation: slidePurple 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
                }
            `}</style>

            <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold flex items-center gap-2">
                    <Activity className="text-blue-400" /> Live Pulse
                </h2>
                <span className="text-xs text-green-400 animate-pulse">● Monitoring Active</span>
            </div>

            <div className="overflow-x-auto min-h-[200px]">
                <table className="w-full text-left text-sm text-gray-400">
                    <thead className="bg-white/5 uppercase text-xs font-medium">
                        <tr>
                            <th className="px-4 py-3 rounded-l-lg">Run ID</th>
                            <th className="px-4 py-3">Node</th>
                            <th className="px-4 py-3">Status</th>
                            <th className="px-4 py-3">Confidence</th>
                            <th className="px-4 py-3 rounded-r-lg text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {isLoading && traces.length === 0 ? (
                            <>
                                <SkeletonRow />
                                <SkeletonRow />
                                <SkeletonRow />
                                <SkeletonRow />
                                <SkeletonRow />
                            </>
                        ) : (
                            traces.map((trace) => (
                                <tr
                                    key={trace.id}
                                    className={`
                                        transition-all border-b border-white/5 animate-slide-purple
                                        ${trace.status === 'repaired'
                                            // Soft glow for repaired + purple hover
                                            ? 'bg-purple-500/5 hover:bg-purple-500/10 cursor-pointer border-l-2 border-l-purple-500/50 hover:border-l-purple-500 shadow-[0_0_15px_rgba(168,85,247,0.05)]'
                                            : trace.status.startsWith('failed')
                                                ? 'hover:bg-rose-500/10 cursor-pointer border-l-2 border-l-transparent hover:border-l-rose-500'
                                                : 'hover:bg-white/5 cursor-default'
                                        }
                                    `}
                                    onClick={() => (trace.status === 'repaired' || trace.status.startsWith('failed')) && handleRowClick(trace)}
                                >
                                    <td className="px-4 py-3 font-mono text-white/70">
                                        <div className="flex items-center gap-2">
                                            {trace.run_id.slice(-8)}
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 font-medium text-white">{trace.node_id}</td>
                                    <td className="px-4 py-3">
                                        <StatusTag status={trace.status} />
                                    </td>
                                    <td className="px-4 py-3">
                                        {trace.status === 'repaired' && (
                                            <div className="flex items-center gap-2 w-24">
                                                <div className="h-1.5 flex-1 bg-gray-700 rounded-full overflow-hidden">
                                                    <div className="h-full bg-teal-400 w-[95%]"></div>
                                                </div>
                                                <span className="text-xs text-teal-400">95%</span>
                                            </div>
                                        )}
                                        {(trace.status === 'success' || trace.status.startsWith('failed')) && <span className="text-white/20">-</span>}
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                        <button className="p-1 hover:bg-white/10 rounded text-gray-500 hover:text-white">
                                            <MoreHorizontal size={16} />
                                        </button>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>

                {!isLoading && traces.length === 0 && (
                    <div className="text-center py-10 text-gray-600">
                        No active traces found. Run a script!
                    </div>
                )}
            </div>

            {/* Sync Status Footer */}
            <div className="absolute bottom-6 right-6 flex items-center gap-2 pointer-events-none">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-[pulse_5s_infinite]" />
                <span className="text-[10px] uppercase tracking-widest text-gray-600 font-mono">
                    Last Data Sync: {lastSync || "..."}
                </span>
            </div>

            <MedicInsightDrawer
                isOpen={!!selectedTrace}
                isLoading={isDrawerLoading}
                onClose={() => setSelectedTrace(null)}
                trace={selectedTrace}
            />
        </div>
    );
}
