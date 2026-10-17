'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import Sigma from 'sigma';
import type { NodeHoverDrawingFunction } from 'sigma/rendering';
import { api } from '@/lib/api';
import type { GraphData } from '@/lib/types';

const SOURCE_TYPE_COLORS: Record<string, string> = {
    audio: '#22c55e',
    pdf: '#ef4444',
    web: '#3b82f6',
    markdown: '#a855f7',
    image: '#eab308',
};
const DEFAULT_COLOR = '#6b7280';
const DIMMED = '#334155';


function buildGraph(data: GraphData): Graph {
    const graph = new Graph({ type: 'undirected' });
    data.nodes.forEach((n, i) => {
        // start on a circle; ForceAtlas2 spreads them out from there
        const angle = (2 * Math.PI * i) / Math.max(data.nodes.length, 1);
        graph.addNode(n.id, {
            label: n.name,
            sourceType: n.source_type,
            x: Math.cos(angle),
            y: Math.sin(angle),
            color: SOURCE_TYPE_COLORS[n.source_type] || DEFAULT_COLOR,
        });
    });
    data.links.forEach((l) => {
        if (graph.hasNode(l.source) && graph.hasNode(l.target) && !graph.hasEdge(l.source, l.target)) {
            graph.addEdge(l.source, l.target, { weight: l.value, size: 0.5 + l.value * 4 });
        }
    });
    graph.forEachNode((node) => {
        graph.setNodeAttribute(node, 'size', 6 + Math.sqrt(graph.degree(node)) * 3);
    });
    if (graph.order > 1) {
        forceAtlas2.assign(graph, {
            iterations: 400,
            settings: { ...forceAtlas2.inferSettings(graph), edgeWeightInfluence: 1, scalingRatio: 10 },
        });
    }
    return graph;
}

// Hover label on a dark card (sigma's default is a white box, unreadable with light text)
const drawHover: NodeHoverDrawingFunction = (context, data, settings) => {
    const size = settings.labelSize;
    context.font = `600 ${size}px ${settings.labelFont}`;
    const label = data.label || '';
    const width = context.measureText(label).width + 12;
    const height = size + 10;
    context.fillStyle = '#0f172a';
    context.strokeStyle = '#94a3b8';
    context.lineWidth = 1;
    context.beginPath();
    context.roundRect(data.x + data.size + 4, data.y - height / 2, width, height, 4);
    context.fill();
    context.stroke();
    context.fillStyle = '#f8fafc';
    context.fillText(label, data.x + data.size + 10, data.y + size / 3);
};

export default function KnowledgeGraph() {
    const router = useRouter();
    const containerRef = useRef<HTMLDivElement>(null);
    const sigmaRef = useRef<Sigma | null>(null);
    const [data, setData] = useState<GraphData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [hovered, setHovered] = useState<string | null>(null);
    const [selected, setSelected] = useState<string | null>(null);
    const [query, setQuery] = useState('');

    const graph = useMemo(() => (data ? buildGraph(data) : null), [data]);

    useEffect(() => {
        api.getGraph()
            .then(setData)
            .catch((err) => {
                console.error(err);
                setError('Failed to load graph data');
            });
    }, []);

    // Create the renderer once the graph is ready
    useEffect(() => {
        if (!graph || !containerRef.current || graph.order === 0) return;
        const renderer = new Sigma(graph, containerRef.current, {
            labelColor: { color: '#e2e8f0' },
            labelFont: 'Inter, Segoe UI, sans-serif',
            labelSize: 13,
            labelWeight: '500',
            labelDensity: 0.6,
            labelRenderedSizeThreshold: 8,
            defaultEdgeColor: '#475569',
            zIndex: true,
            stagePadding: 160, // room for labels and the toolbar around the outermost notes
            defaultDrawNodeHover: drawHover,
        });
        renderer.on('enterNode', ({ node }) => setHovered(node));
        renderer.on('leaveNode', () => setHovered(null));
        renderer.on('clickNode', ({ node }) => setSelected(node));
        renderer.on('doubleClickNode', ({ node, event }) => {
            event.preventSigmaDefault();
            router.push(`/notes/${node}`);
        });
        renderer.on('clickStage', () => setSelected(null));
        sigmaRef.current = renderer;
        return () => {
            renderer.kill();
            sigmaRef.current = null;
        };
    }, [graph, router]);

    // Highlight the hovered/selected note and its neighbours, dim the rest
    useEffect(() => {
        const renderer = sigmaRef.current;
        if (!renderer || !graph) return;
        const focus = hovered || selected;
        const neighbours = new Set(focus ? graph.neighbors(focus) : []);
        renderer.setSetting('nodeReducer', (node, attrs) => {
            if (!focus) return attrs;
            if (node === focus) return { ...attrs, highlighted: true, zIndex: 2 };
            if (neighbours.has(node)) return { ...attrs, forceLabel: true, zIndex: 1 };
            return { ...attrs, color: DIMMED, label: '', zIndex: 0 };
        });
        renderer.setSetting('edgeReducer', (edge, attrs) => {
            if (!focus) return attrs;
            return graph.hasExtremity(edge, focus)
                ? { ...attrs, color: '#94a3b8', zIndex: 1 }
                : { ...attrs, hidden: true };
        });
    }, [graph, hovered, selected]);

    const matches = useMemo(() => {
        if (!graph || query.trim().length < 2) return [];
        const q = query.toLowerCase();
        return graph.filterNodes((_, attrs) => String(attrs.label).toLowerCase().includes(q)).slice(0, 8);
    }, [graph, query]);

    const focusNode = (node: string) => {
        setSelected(node);
        setQuery('');
        const renderer = sigmaRef.current;
        const display = renderer?.getNodeDisplayData(node);
        if (renderer && display) {
            renderer.getCamera().animate({ x: display.x, y: display.y, ratio: 0.5 }, { duration: 500 });
        }
    };

    const selectedAttrs = selected && graph?.hasNode(selected) ? graph.getNodeAttributes(selected) : null;
    const selectedNeighbours = selected && graph?.hasNode(selected)
        ? graph
              .mapEdges(selected, (edge, attrs, source, target) => ({
                  id: source === selected ? target : source,
                  weight: attrs.weight as number,
              }))
              .sort((a, b) => b.weight - a.weight)
        : [];

    return (
        <div className="graph-view">
            <div className="graph-toolbar">
                <Link href="/notes" className="btn btn-secondary">← Notes</Link>
                <h1 className="graph-title">Knowledge Graph</h1>
                {graph && (
                    <span className="text-muted">
                        {graph.order} notes · {graph.size} links
                    </span>
                )}
                <div className="graph-search">
                    <input
                        className="input"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Find a note…"
                    />
                    {matches.length > 0 && (
                        <ul className="graph-search-results">
                            {matches.map((node) => (
                                <li key={node}>
                                    <button onClick={() => focusNode(node)}>{graph?.getNodeAttribute(node, 'label')}</button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
                <button className="btn btn-secondary" onClick={() => sigmaRef.current?.getCamera().animatedReset()}>
                    Fit
                </button>
            </div>

            {error && <div className="graph-error">{error}</div>}
            {!error && data && data.nodes.length === 0 && (
                <div className="graph-empty">
                    <p>No notes to visualize yet.</p>
                    <p>Add some content to see your knowledge graph!</p>
                </div>
            )}
            {!error && !data && <div className="graph-loading">Loading graph...</div>}

            <div ref={containerRef} className="graph-canvas" />

            <div className="graph-legend">
                {Object.entries(SOURCE_TYPE_COLORS).map(([type, color]) => (
                    <span key={type} className="legend-item">
                        <span className="legend-dot" style={{ backgroundColor: color }} />
                        {type}
                    </span>
                ))}
                <span className="legend-hint">Click to select · double-click to open · scroll to zoom</span>
            </div>

            {selectedAttrs && graph && (
                <aside className="graph-panel">
                    <div className="graph-panel-type" style={{ color: selectedAttrs.color }}>{selectedAttrs.sourceType}</div>
                    <h2 className="graph-panel-title">{selectedAttrs.label}</h2>
                    <Link href={`/notes/${selected}`} className="btn btn-primary">Open note</Link>
                    {selectedNeighbours.length > 0 && (
                        <>
                            <h3 className="graph-panel-heading">Linked notes</h3>
                            <ul className="graph-panel-links">
                                {selectedNeighbours.map(({ id, weight }) => (
                                    <li key={id}>
                                        <button onClick={() => focusNode(id)}>
                                            {graph.getNodeAttribute(id, 'label')}
                                        </button>
                                        <span className="text-muted">{weight.toFixed(2)}</span>
                                    </li>
                                ))}
                            </ul>
                        </>
                    )}
                </aside>
            )}
        </div>
    );
}
