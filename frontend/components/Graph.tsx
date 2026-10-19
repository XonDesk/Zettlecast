'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import Sigma from 'sigma';
import type { NodeHoverDrawingFunction } from 'sigma/rendering';
import { createNodeBorderProgram } from '@sigma/node-border';
import { api } from '@/lib/api';
import type { GraphData, LinkSuggestion } from '@/lib/types';

// Community fills. Graph nodes can sit next to any other colour, so the palette must hold up
// on every pair: these four are the largest set that passes the dataviz validator all-pairs on
// the #0f172a surface (CVD dE 6.9, so the communities are also named on the map). Colour
// follows the community id, never its rank; ids past the palette use the neutral fill.
const COMMUNITY_COLORS = ['#3987e5', '#c98500', '#d55181', '#008300'];
const OTHER_COMMUNITY = '#64748b';
const UNCLUSTERED = '#475569';
const DIMMED = '#1e293b';

// Source type as a neutral ring, so it can't be mistaken for a community colour
const RINGS: Record<string, { color: string; size: number; label: string }> = {
    web: { color: '#f8fafc', size: 0.22, label: 'web' },
    pdf: { color: '#94a3b8', size: 0.22, label: 'document' },
    markdown: { color: '#94a3b8', size: 0.22, label: 'document' },
    image: { color: '#f8fafc', size: 0.42, label: 'image' },
};
const RING_LEGEND = [{ label: 'podcast', color: 'transparent', size: 0 }, RINGS.web, RINGS.pdf, RINGS.image];
const TYPE_LABELS: Record<string, string> = { audio: 'podcast', pdf: 'PDF', markdown: 'note', web: 'web page', image: 'image' };
const MAP_LABELS = 12; // community names drawn on the map
const INSIDE_PULL = 4; // layout attraction of a link inside a community vs one across

const NodeRingProgram = createNodeBorderProgram({
    borders: [
        { size: { attribute: 'ringSize', defaultValue: 0 }, color: { attribute: 'ringColor', defaultValue: 'transparent' } },
        { size: { fill: true }, color: { attribute: 'color' } },
    ],
});

function communityColor(id: number | null | undefined): string {
    if (id === null || id === undefined) return UNCLUSTERED;
    return id >= 0 && id < COMMUNITY_COLORS.length ? COMMUNITY_COLORS[id] : OTHER_COMMUNITY;
}

function formatTime(seconds: number | null | undefined): string {
    if (seconds === null || seconds === undefined) return '';
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const pad = (n: number) => String(n).padStart(2, '0');
    return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

/** Passage text without its "[12.3s - 45.6s] Speaker:" prefix */
function passageText(text: string): string {
    return text.replace(/^\s*\[[\d.]+s(?:\s*-\s*[\d.]+s)?\]\s*/, '');
}

// Small seeded PRNG so the starting layout (and so the final one) is the same on every load
function mulberry32(seed: number) {
    return () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function buildGraph(data: GraphData): Graph {
    const graph = new Graph({ type: 'undirected' });
    // start each community around its own point on a circle: ForceAtlas2 then only has to
    // settle the details, and unrelated groups don't begin tangled together
    const order = [...new Set(data.nodes.map((n) => n.community ?? -1))];
    const random = mulberry32(42);
    data.nodes.forEach((n) => {
        const slot = order.indexOf(n.community ?? -1);
        const angle = (2 * Math.PI * slot) / Math.max(order.length, 1);
        const ring = RINGS[n.source_type];
        graph.addNode(n.id, {
            label: n.name,
            sourceType: n.source_type,
            community: n.community,
            x: Math.cos(angle) * 10 + (random() - 0.5) * 2,
            y: Math.sin(angle) * 10 + (random() - 0.5) * 2,
            color: communityColor(n.community),
            type: 'ring',
            ringColor: ring?.color ?? 'transparent',
            ringSize: ring?.size ?? 0,
        });
    });
    const community = new Map(data.nodes.map((n) => [n.id, n.community]));
    data.links.forEach((l) => {
        if (graph.hasNode(l.source) && graph.hasNode(l.target) && !graph.hasEdge(l.source, l.target)) {
            const c = community.get(l.source);
            const inside = c !== null && c !== undefined && c === community.get(l.target);
            graph.addEdge(l.source, l.target, {
                // layout pull: links inside a community attract harder than links across, so
                // communities form visible clusters while cross links still place them
                weight: l.value * (inside ? INSIDE_PULL : 1),
                value: l.value,
                kind: l.kind,
                size: l.kind === 'link' ? 2 : 0.5 + l.value * 3,
                color: l.kind === 'link' ? '#cbd5e1' : '#475569',
            });
        }
    });
    graph.forEachNode((node) => {
        graph.setNodeAttribute(node, 'size', 6 + Math.sqrt(graph.degree(node)) * 3);
    });
    if (graph.order > 1) {
        forceAtlas2.assign(graph, {
            iterations: graph.order > 1000 ? 400 : 800,
            settings: {
                ...forceAtlas2.inferSettings(graph),
                linLogMode: true, // tighter, better separated communities
                outboundAttractionDistribution: true, // hubs don't drag everything to the middle
                edgeWeightInfluence: 1,
                scalingRatio: 20, // more repulsion: room between communities
                gravity: 0.5,
                barnesHutOptimize: graph.order > 500,
            },
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

interface MapLabel {
    id: number;
    label: string;
    x: number;
    y: number;
}

export default function KnowledgeGraph() {
    const router = useRouter();
    const containerRef = useRef<HTMLDivElement>(null);
    const sigmaRef = useRef<Sigma | null>(null);
    const [data, setData] = useState<GraphData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [hovered, setHovered] = useState<string | null>(null);
    const [selected, setSelected] = useState<string | null>(null);
    const [focusCommunity, setFocusCommunity] = useState<number | null>(null);
    const [query, setQuery] = useState('');
    const [links, setLinks] = useState<{ uuid: string; items: LinkSuggestion[] } | null>(null);
    const [mapLabels, setMapLabels] = useState<MapLabel[]>([]);

    const graph = useMemo(() => (data ? buildGraph(data) : null), [data]);
    const communities = useMemo(() => data?.communities ?? [], [data]);

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
            nodeProgramClasses: { ring: NodeRingProgram },
            labelColor: { color: '#e2e8f0' },
            labelFont: 'Inter, Segoe UI, sans-serif',
            labelSize: 13,
            labelWeight: '500',
            // titles only for well-linked notes at this zoom; the rest on hover or zooming in
            labelDensity: 0.25,
            labelGridCellSize: 150,
            labelRenderedSizeThreshold: 14,
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
        renderer.on('clickStage', () => {
            setSelected(null);
            setFocusCommunity(null);
        });

        // community names at their centroids, kept in place as the camera moves
        const centroids = new Map<number, { x: number; y: number; n: number }>();
        graph.forEachNode((_, attrs) => {
            if (attrs.community === null || attrs.community === undefined) return;
            const c = centroids.get(attrs.community) ?? { x: 0, y: 0, n: 0 };
            centroids.set(attrs.community, { x: c.x + attrs.x, y: c.y + attrs.y, n: c.n + 1 });
        });
        const named = communities.slice(0, MAP_LABELS);
        const placeLabels = () => {
            setMapLabels(
                named.flatMap((c) => {
                    const sum = centroids.get(c.id);
                    if (!sum) return [];
                    const p = renderer.graphToViewport({ x: sum.x / sum.n, y: sum.y / sum.n });
                    return [{ id: c.id, label: c.label, x: p.x, y: p.y }];
                }),
            );
        };
        renderer.on('afterRender', placeLabels);
        sigmaRef.current = renderer;
        return () => {
            renderer.kill();
            sigmaRef.current = null;
        };
    }, [graph, communities, router]);

    // Highlight the hovered/selected note and its neighbours, or one community; dim the rest
    useEffect(() => {
        const renderer = sigmaRef.current;
        if (!renderer || !graph) return;
        const focus = hovered || selected;
        const neighbours = new Set(focus ? graph.neighbors(focus) : []);
        renderer.setSetting('nodeReducer', (node, attrs) => {
            if (focus) {
                if (node === focus) return { ...attrs, highlighted: true, zIndex: 2 };
                if (neighbours.has(node)) return { ...attrs, forceLabel: true, zIndex: 1 };
                return { ...attrs, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            if (focusCommunity !== null && attrs.community !== focusCommunity) {
                return { ...attrs, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            return attrs;
        });
        renderer.setSetting('edgeReducer', (edge, attrs) => {
            if (focus) {
                return graph.hasExtremity(edge, focus) ? { ...attrs, color: '#94a3b8', zIndex: 1 } : { ...attrs, hidden: true };
            }
            if (focusCommunity !== null) {
                const [a, b] = graph.extremities(edge);
                const inside = graph.getNodeAttribute(a, 'community') === focusCommunity
                    && graph.getNodeAttribute(b, 'community') === focusCommunity;
                return inside ? attrs : { ...attrs, hidden: true };
            }
            return attrs;
        });
    }, [graph, hovered, selected, focusCommunity]);

    // Why the selected note links to each neighbour (shared people/topics, matching passage)
    useEffect(() => {
        if (!selected) return;
        let cancelled = false;
        api.getNote(selected, true)
            .then((note) => {
                if (!cancelled) setLinks({ uuid: selected, items: note.suggestions ?? [] });
            })
            .catch((err) => console.error(err));
        return () => {
            cancelled = true;
        };
    }, [selected]);

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
    const selectedLinks = links && links.uuid === selected ? links.items : null;
    const selectedCommunity = selectedAttrs
        ? communities.find((c) => c.id === selectedAttrs.community)
        : undefined;

    return (
        <div className="graph-view">
            <div className="graph-toolbar">
                <Link href="/notes" className="btn btn-secondary">← Notes</Link>
                <h1 className="graph-title">Knowledge Graph</h1>
                {graph && (
                    <span className="text-muted">
                        {graph.order} notes · {graph.size} links · {communities.length} communities
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

            {!selected && !hovered && (
                <div className="graph-community-labels" aria-hidden="true">
                    {mapLabels
                        .filter((l) => focusCommunity === null || l.id === focusCommunity)
                        .map((l) => (
                            <span key={l.id} className="graph-community-label" style={{ left: l.x, top: l.y }}>
                                {l.label}
                            </span>
                        ))}
                </div>
            )}

            <div className="graph-legend">
                {communities.length > 0 && (
                    <div className="legend-group">
                        <span className="legend-group-title">Communities</span>
                        {communities.map((c) => (
                            <button
                                key={c.id}
                                className="legend-community"
                                aria-pressed={focusCommunity === c.id}
                                onClick={() => setFocusCommunity(focusCommunity === c.id ? null : c.id)}
                                title={`${c.size} notes: show only this community`}
                            >
                                <span className="legend-dot" style={{ backgroundColor: communityColor(c.id) }} />
                                {c.label || `Community ${c.id}`}
                                <span className="text-muted">{c.size}</span>
                            </button>
                        ))}
                    </div>
                )}
                <div className="legend-group">
                    <span className="legend-group-title">Ring</span>
                    {RING_LEGEND.map((r) => (
                        <span key={r.label} className="legend-item">
                            <span
                                className="legend-ring"
                                style={{ border: r.size ? `${Math.round(r.size * 10)}px solid ${r.color}` : '1px dashed #64748b' }}
                            />
                            {r.label}
                        </span>
                    ))}
                    <span className="legend-hint">Click a note to see why it links · double-click to open</span>
                </div>
            </div>

            {selectedAttrs && graph && (
                <aside className="graph-panel">
                    <div className="graph-panel-type" style={{ color: communityColor(selectedAttrs.community) }}>
                        {TYPE_LABELS[selectedAttrs.sourceType] ?? selectedAttrs.sourceType}
                        {selectedCommunity ? ` · ${selectedCommunity.label}` : ''}
                    </div>
                    <h2 className="graph-panel-title">{selectedAttrs.label}</h2>
                    <Link href={`/notes/${selected}`} className="btn btn-primary">Open note</Link>
                    <h3 className="graph-panel-heading">Linked notes</h3>
                    {!selectedLinks && <span className="text-muted">Loading…</span>}
                    {selectedLinks && selectedLinks.length === 0 && <span className="text-muted">No links yet.</span>}
                    {selectedLinks && selectedLinks.length > 0 && (
                        <ul className="graph-panel-links">
                            {selectedLinks.map((s) => (
                                <li key={s.uuid}>
                                    <div className="graph-panel-link-row">
                                        <button onClick={() => graph.hasNode(s.uuid) && focusNode(s.uuid)}>{s.title}</button>
                                        <span className="text-muted">{s.score.toFixed(2)}</span>
                                    </div>
                                    {s.reason && <div className="graph-panel-reason">{s.reason}</div>}
                                    {s.passage && (
                                        <blockquote className="graph-panel-passage">
                                            {s.passage.there.time !== null && <time>{formatTime(s.passage.there.time)}</time>}
                                            {passageText(s.passage.there.text).slice(0, 220)}…
                                        </blockquote>
                                    )}
                                </li>
                            ))}
                        </ul>
                    )}
                </aside>
            )}
        </div>
    );
}
