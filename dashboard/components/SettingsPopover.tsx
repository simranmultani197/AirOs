"use client";

import { useState } from "react";
import { Settings, X } from "lucide-react";

export type BusinessSettings = {
    manualLaborCost: number;
    infraCost: number;
    downtimeImpact: number;
};

interface SettingsPopoverProps {
    settings: BusinessSettings;
    onUpdate: (newSettings: BusinessSettings) => void;
}

export default function SettingsPopover({ settings, onUpdate }: SettingsPopoverProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [tempSettings, setTempSettings] = useState(settings);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        onUpdate(tempSettings);
        setIsOpen(false);

        try {
            await fetch("http://localhost:8000/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    manual_labor_cost: tempSettings.manualLaborCost,
                    infrastructure_rate: tempSettings.infraCost
                })
            });
        } catch (err) {
            console.error("Failed to save settings", err);
        }
    };

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="p-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors text-gray-400 hover:text-white"
            >
                <Settings size={20} />
            </button>

            {isOpen && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setIsOpen(false)}
                    />

                    {/* Popover */}
                    <div className="absolute right-0 top-full mt-2 w-80 bg-slate-900/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl p-6 z-50 animate-in fade-in zoom-in-95 duration-200">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-white font-bold flex items-center gap-2">
                                <Settings size={16} className="text-purple-400" />
                                Business Logic
                            </h3>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="text-gray-500 hover:text-white transition-colors"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide">
                                    Manual Labor Cost ($/hr)
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                                    <input
                                        type="number"
                                        value={tempSettings.manualLaborCost}
                                        onChange={(e) => setTempSettings({ ...tempSettings, manualLaborCost: parseFloat(e.target.value) })}
                                        className="w-full bg-black/40 border border-white/10 rounded-lg py-2 pl-7 pr-3 text-white text-sm focus:outline-none focus:border-purple-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide">
                                    Infrastructure Cost ($/mo)
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                                    <input
                                        type="number"
                                        value={tempSettings.infraCost}
                                        onChange={(e) => setTempSettings({ ...tempSettings, infraCost: parseFloat(e.target.value) })}
                                        className="w-full bg-black/40 border border-white/10 rounded-lg py-2 pl-7 pr-3 text-white text-sm focus:outline-none focus:border-purple-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide">
                                    Avg. Downtime Impact ($/min)
                                </label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">$</span>
                                    <input
                                        type="number"
                                        value={tempSettings.downtimeImpact}
                                        onChange={(e) => setTempSettings({ ...tempSettings, downtimeImpact: parseFloat(e.target.value) })}
                                        className="w-full bg-black/40 border border-white/10 rounded-lg py-2 pl-7 pr-3 text-white text-sm focus:outline-none focus:border-purple-500 transition-colors"
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium py-2 rounded-lg text-sm transition-colors mt-2"
                            >
                                Update Calculations
                            </button>
                        </form>
                    </div>
                </>
            )}
        </div>
    );
}
