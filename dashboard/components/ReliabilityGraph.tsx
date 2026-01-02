"use client";

import { useMemo, useState } from "react";

interface DataPoint {
    timestamp: string;
    score: number;
}

interface ReliabilityGraphProps {
    data: DataPoint[];
    width?: number;
    height?: number;
    color?: string;
}

export function ReliabilityGraph({
    data,
    width = 200,
    height = 50,
    color = "#4ade80" // Success Green
}: ReliabilityGraphProps) {
    const [isHovered, setIsHovered] = useState(false);

    // 1. Calculate Scales & Path
    const { path, min, max, peak, low } = useMemo(() => {
        if (!data.length) return { path: "", min: 0, max: 100, peak: null, low: null };

        const scores = data.map(d => d.score);
        const minVal = Math.min(...scores);
        const maxVal = Math.max(...scores);
        const range = maxVal - minVal || 1;

        // Peak & Low Calculation for Tooltip
        const peakPoint = data.reduce((prev, curr) => curr.score > prev.score ? curr : prev, data[0]);
        const lowPoint = data.reduce((prev, curr) => curr.score < prev.score ? curr : prev, data[0]);

        const points = data.map((d, i) => {
            const x = (i / (data.length - 1)) * width;
            // Invert Y axis, leave padding
            const y = height - ((d.score - minVal) / range) * (height - 4) - 2;
            return `${x},${y}`;
        });

        // Simple straight lines for "High Density" look (or we could use bezier)
        // L denotes line to
        const pathD = `M ${points.join(" L ")}`;

        return { path: pathD, min: minVal, max: maxVal, peak: peakPoint, low: lowPoint };
    }, [data, width, height]);

    // Animation Config
    // We set dasharray to a large enough number (e.g. width * 2)
    // In CSS we animate offset from full to 0.

    return (
        <div
            className="relative"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
                {/* Definition for Gradient */}
                <defs>
                    <linearGradient id="glowGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity="0.4" />
                        <stop offset="100%" stopColor={color} stopOpacity="0.0" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Path with Animation */}
                {path && (
                    <path
                        d={path}
                        fill="none"
                        stroke={color}
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{
                            filter: `drop-shadow(0 0 3px ${color})`,
                        }}
                        className="animate-draw-path"
                    />
                )}
            </svg>

            {/* Hover Tooltip - "Analysis Summary" */}
            {isHovered && peak && low && (
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full w-48 bg-slate-900/90 backdrop-blur-md border border-white/10 p-3 rounded-lg shadow-xl text-xs z-20 pointer-events-none animate-in fade-in zoom-in-95 duration-200">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-400">Peak Performance</span>
                        <span className="text-emerald-400 font-mono">{peak.score}%</span>
                    </div>
                    <div className="text-[10px] text-gray-500 font-mono mb-2 text-right">
                        {peak.timestamp}
                    </div>

                    <div className="h-px bg-white/10 my-1" />

                    <div className="flex justify-between items-center mt-2 mb-1">
                        <span className="text-gray-400">Lowest Point</span>
                        <span className="text-rose-400 font-mono">{low.score}%</span>
                    </div>
                    <div className="text-[10px] text-gray-500 font-mono text-right">
                        {low.timestamp}
                    </div>
                </div>
            )}

            {/* CSS for draw animation */}
            <style jsx>{`
                .animate-draw-path {
                    stroke-dasharray: ${width * 1.5};
                    stroke-dashoffset: ${width * 1.5};
                    animation: draw 2s ease-out forwards;
                }
                @keyframes draw {
                    to {
                        stroke-dashoffset: 0;
                    }
                }
            `}</style>
        </div>
    );
}
