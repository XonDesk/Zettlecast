'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import type { GraphNode, GraphData } from '@/lib/types';
import { api } from '@/lib/api';

// Dynamically import to avoid SSR issues with canvas
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
    ssr: false,
    loading: () => <div className="graph-loading">Loading graph...</div>,
});

interface GraphProps {
    onNodeClick?: (node: GraphNode) => void;
}

const SOURCE_TYPE_COLORS: Record<string, string> = {
    pdf: '#ef4444',     // red
    web: '#3b82f6',     // blue
    audio: '#22c55e',   // green
    markdown: '#a855f7', // purple
    rss: '#f97316',     // orange
};

export default function Graph({ onNodeClick }: GraphProps) {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const graphRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // Fetch graph data
    useEffect(() => {
        const fetchGraph = async () => {
            try {
                setLoading(true);
                const graphData = await api.getGraph();
                setData(graphData);
                setError(null);
            } catch (err) {
                setError('Failed to load graph data');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchGraph();
    }, []);

    // Handle resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight,
                });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    const handleNodeClick = useCallback(
        (node: any) => {
            if (onNodeClick) {
                onNodeClick(node as GraphNode);
            }
        },
        [onNodeClick]
    );

    const nodeColor = useCallback((node: any) => {
        return SOURCE_TYPE_COLORS[node.source_type] || '#6b7280';
    }, []);

    const nodeLabel = useCallback((node: any) => {
        return node.name;
    }, []);

    if (loading) {
        return <div className="graph-loading">Loading graph...</div>;
    }

    if (error) {
        return <div className="graph-error">{error}</div>;
    }

    if (data.nodes.length === 0) {
        return (
            <div className="graph-empty">
                <p>No notes to visualize yet.</p>
                <p>Add some content to see your knowledge graph!</p>
            </div>
        );
    }

    return (
        <div ref={containerRef} className="graph-container">
            <ForceGraph2D
                ref={graphRef}
                graphData={data}
                width={dimensions.width}
                height={dimensions.height}
                nodeLabel={nodeLabel}
                nodeColor={nodeColor}
                nodeVal="val"
                linkWidth={(link: any) => Math.sqrt(link.value || 1) * 2}
                linkColor={() => '#94a3b8'}
                onNodeClick={handleNodeClick}
                enableNodeDrag
                enableZoomInteraction
                enablePanInteraction
                cooldownTicks={100}
                nodeCanvasObject={(node: any, ctx, globalScale) => {
                    const label = node.name;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Inter, sans-serif`;

                    // Draw node
                    const nodeSize = 5;
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
                    ctx.fillStyle = nodeColor(node);
                    ctx.fill();

                    // Draw label if zoomed in enough
                    if (globalScale > 0.5) {
                        ctx.fillStyle = '#f8fafc';  // Light color for dark background
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'top';
                        ctx.fillText(label.substring(0, 20), node.x, node.y + nodeSize + 2);
                    }
                }}
            />
            <div className="graph-legend">
                {Object.entries(SOURCE_TYPE_COLORS).map(([type, color]) => (
                    <span key={type} className="legend-item">
                        <span className="legend-dot" style={{ backgroundColor: color }} />
                        {type}
                    </span>
                ))}
            </div>
        </div>
    );
}
