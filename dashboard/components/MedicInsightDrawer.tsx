"use client";

import { Drawer } from "@/components/ui/Drawer";
import { Sparkles, Terminal, Activity, Download, ShieldAlert, Cpu, AlertTriangle, XCircle } from "lucide-react";

interface MedicInsightDrawerProps {
    isOpen: boolean;
    onClose: () => void;
    trace: any;
    isLoading?: boolean;
}

export default function MedicInsightDrawer({ isOpen, onClose, trace, isLoading }: MedicInsightDrawerProps) {
    if (!isOpen) return null; // Standardize open check

    // Loading State
    if (isLoading) {
        return (
            <Drawer isOpen={isOpen} onClose={onClose} title="">
                <div className="h-full flex flex-col items-center justify-center space-y-8 relative overflow-hidden">
                    {/* Tactical Pulse Animation */}
                    <div className="relative flex items-center justify-center">
                        <div className="absolute w-32 h-32 rounded-full border border-purple-500/20 animate-[ping_3s_linear_infinite]"></div>
                        <div className="absolute w-24 h-24 rounded-full border border-purple-500/40 animate-[ping_2s_linear_infinite_0.5s]"></div>
                        <div className="absolute w-16 h-16 rounded-full border border-purple-500/60 animate-[ping_1s_linear_infinite_0.8s]"></div>
                        <div className="relative w-4 h-4 rounded-full bg-purple-500 shadow-[0_0_20px_rgba(168,85,247,1)] z-10"></div>

                        {/* Radar Scan Effect */}
                        <div className="absolute inset-0 w-48 h-48 border border-white/5 rounded-full animate-[spin_4s_linear_infinite] border-t-purple-500/50 bg-gradient-to-t from-transparent via-transparent to-purple-500/10" style={{ clipPath: 'polygon(50% 50%, 0 0, 100% 0)' }}></div>
                    </div>

                    <div className="text-center space-y-2 z-10">
                        <p className="text-purple-300 font-mono text-sm tracking-[0.2em] uppercase animate-pulse">Establishing Uplink</p>
                        <p className="text-xs text-gray-500 font-mono">Retrieving Run Telemetry...</p>
                    </div>
                </div>
            </Drawer>
        );
    }

    if (!trace) return null;

    const isFailure = trace.status && trace.status.startsWith('failed');

    return (
        <Drawer
            isOpen={isOpen}
            onClose={onClose}
            title=""
            className={isFailure ? "border-l-4 border-rose-500 shadow-[0_0_50px_rgba(244,63,94,0.3)]" : "border-white/10"}
        >

            {/* Header with Run ID and Badge */}
            <div className="mb-6 flex items-center justify-between border-b border-white/5 pb-4">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <span className="text-gray-500 font-mono text-lg">RUN</span> {trace.run_id.slice(-8)}
                    </h2>
                    <p className="text-xs text-gray-400 font-mono mt-1">Node: {trace.node_id}</p>
                </div>

                {!isFailure ? (
                    <div className="bg-purple-500/20 text-purple-300 border border-purple-500/30 px-3 py-1.5 rounded-full flex items-center gap-2 text-xs font-bold uppercase tracking-wider shadow-[0_0_15px_rgba(168,85,247,0.3)]">
                        <Sparkles size={14} className="animate-pulse" />
                        Autonomous Repair
                    </div>
                ) : (
                    <div className="bg-rose-500/20 text-rose-300 border border-rose-500/30 px-3 py-1.5 rounded-full flex items-center gap-2 text-xs font-bold uppercase tracking-wider shadow-[0_0_15px_rgba(244,63,94,0.5)] animate-pulse">
                        <AlertTriangle size={14} />
                        Critical Failure
                    </div>
                )}
            </div>

            {/* Section 1: Hypothesis vs Breakdown Trace */}
            <div className="mb-8">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 block flex items-center gap-2">
                    <Activity size={14} /> {isFailure ? "Breakdown Trace" : "Medic Hypothesis"}
                </label>

                {!isFailure ? (
                    <div className="bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border border-indigo-500/20 rounded-xl p-5 relative">
                        <p className="text-gray-200 text-sm leading-relaxed font-light">
                            {trace.diagnosis ? (
                                <>
                                    <span className="text-purple-300 font-mono text-xs block mb-2">DIAGNOSIS ENGINE:</span>
                                    {trace.diagnosis}
                                </>
                            ) : (
                                <>
                                    Detected a <span className="text-rose-400 font-mono">SchemaValidationError</span> in the output payload.
                                    The field <code className="bg-black/30 px-1 py-0.5 rounded text-indigo-300">value</code> was expected to be an <code className="text-emerald-300">integer</code> but received a <code className="text-amber-300">string</code>.
                                    Initiated autonomous repair protocol to cast the type while preserving data integrity.
                                </>
                            )}
                        </p>
                    </div>
                ) : (
                    <div className="bg-gradient-to-br from-rose-900/10 to-red-900/5 border border-rose-500/20 rounded-xl p-5 relative">
                        <p className="text-gray-200 text-sm leading-relaxed font-light">
                            Critical Exception in <span className="text-rose-400 font-mono">Execution Protocol</span>.
                            <br />
                            <span className="text-rose-300 font-mono text-xs mt-2 block">
                                Error: {trace.diagnosis || (trace.output_state ? String(trace.output_state).replace(/^"|"$/g, '') : "Unknown Error")}
                            </span>
                        </p>
                    </div>
                )}
            </div>

            {/* Section 2: Action Log (Terminal Style) */}
            <div className="mb-8">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 block flex items-center gap-2">
                    <Terminal size={14} /> {isFailure ? "Action Log" : "Action Log"}
                </label>
                <div className={`${isFailure ? 'bg-rose-950/20 border-rose-500/20' : 'bg-black/80 border-white/10'} rounded-lg border p-4 font-mono text-xs shadow-inner transition-colors duration-500`}>
                    <div className="flex flex-col gap-2">
                        {!isFailure ? (
                            <>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-emerald-500">➜</span>
                                    <span>analyzing stack_trace...</span>
                                </div>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-emerald-500">➜</span>
                                    <span>generating repair_patch --strategy=semantic_fix</span>
                                </div>
                                <div className="flex gap-2 text-gray-400 pl-4 border-l border-white/10 ml-1">
                                    <span>Applying patch: {`{"op": "replace", "path": "/value", "value": 100}`}</span>
                                </div>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-emerald-500">➜</span>
                                    <span>validating output schema... <span className="text-emerald-400">OK</span></span>
                                </div>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-emerald-500">➜</span>
                                    <span className="text-purple-400">resume_execution()</span>
                                </div>
                            </>
                        ) : (
                            <>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-rose-500">➜</span>
                                    <span>verifying state hash... <span className="text-amber-400">COLLISION</span></span>
                                </div>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-rose-500">➜</span>
                                    <span>checking fuse capacity... <span className="text-rose-500">BLOWN</span></span>
                                </div>
                                <div className="flex gap-2 text-rose-400 pl-4 border-l border-rose-500/30 ml-1">
                                    <span>CRITICAL: {trace.output_state ? String(trace.output_state).slice(0, 100).replace(/^"|"$/g, '') : "Loop Detected"}...</span>
                                </div>
                                <div className="flex gap-2 text-gray-500">
                                    <span className="text-rose-500">➜</span>
                                    <span className="text-rose-500">emergency_halt()</span>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Section 3: Confidence Breakdown (Only for Repairs) */}
            {!isFailure && (
                <div className="mb-8 p-5 bg-white/5 rounded-2xl border border-white/5">
                    <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 block flex items-center gap-2">
                        <Cpu size={14} /> Confidence Metrics
                    </label>

                    <div className="space-y-4">
                        {/* Logprob Certainty */}
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-300">Logprob Certainty</span>
                                <span className="text-emerald-400 font-mono">98%</span>
                            </div>
                            <div className="h-1.5 w-full bg-gray-700/50 rounded-full overflow-hidden">
                                <div className="h-full bg-emerald-500 w-[98%] shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                            </div>
                        </div>

                        {/* State Validation */}
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-300">State Validation</span>
                                <span className="text-teal-400 font-mono">100%</span>
                            </div>
                            <div className="h-1.5 w-full bg-gray-700/50 rounded-full overflow-hidden">
                                <div className="h-full bg-teal-500 w-[100%]"></div>
                            </div>
                        </div>

                        {/* Historical Success Rate */}
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-300">Historical Success</span>
                                <span className="text-blue-400 font-mono">85%</span>
                            </div>
                            <div className="h-1.5 w-full bg-gray-700/50 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500 w-[85%]"></div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Footer Buttons */}
            <div className="flex gap-3 mt-auto pt-6 border-t border-white/5">
                <button
                    className={`flex-1 py-3 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2
                        ${isFailure
                            ? "bg-rose-500 hover:bg-rose-600 text-white shadow-[0_0_20px_rgba(244,63,94,0.5)] border border-rose-400"
                            : "bg-rose-500/10 border border-rose-500/30 text-rose-300 hover:bg-rose-500/20"
                        }
                    `}
                >
                    <ShieldAlert size={16} />
                    {isFailure ? "Urgent Intervention" : "Manual Override"}
                </button>
                <button
                    className="flex-1 py-3 rounded-lg bg-white/5 border border-white/10 text-gray-300 text-sm font-medium hover:bg-white/10 transition-all flex items-center justify-center gap-2"
                >
                    <Download size={16} />
                    Export Logs
                </button>
            </div>

        </Drawer>
    );
}
