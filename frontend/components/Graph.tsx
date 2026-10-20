'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Graph from 'graphology';
import {
    forceCollide,
    forceLink,
    forceManyBody,
    forceSimulation,
    forceX,
    forceY,
    type Simulation,
    type SimulationLinkDatum,
    type SimulationNodeDatum,
} from 'd3-force';
import Sigma from 'sigma';
import type { NodeHoverDrawingFunction } from 'sigma/rendering';
import { createNodeBorderProgram } from '@sigma/node-border';
import { api } from '@/lib/api';
import { compileQuery } from '@/lib/graphQuery';
import type { GraphData, GraphNode, LinkSuggestion } from '@/lib/types';
import GraphPane, {
    type CommunityInsight,
    type Facets,
    type GraphSettings,
    mergeSettings,
} from './GraphPane';

// Community fills: one hue per community, as Obsidian's groups. Twelve is more than the eye
// can tell apart in every pair, so the communities are also named on the map and the colour of
// any of them can be changed in the pane. Colour follows the community id, never its rank;
// ids past the palette use the neutral fill.
const COMMUNITY_COLORS = [
    '#4e9af1', '#f0a33a', '#e0607e', '#3fb950', '#a78bfa', '#2dd4bf',
    '#e3c84b', '#ef6b5b', '#a3d65c', '#d46fd8', '#7dd3fc', '#c9a27e',
];
const OTHER_COMMUNITY = '#64748b';
const UNCLUSTERED = '#475569';
const DIMMED = '#1e293b';
// Colour by year: older notes dim, recent ones bright (one hue, so order reads at a glance)
const YEAR_OLD = [0x3b, 0x55, 0x7a];
const YEAR_NEW = [0x9f, 0xd0, 0xff];

// Source type as a neutral ring, so it can't be mistaken for a community colour
const RINGS: Record<string, { color: string; size: number; label: string }> = {
    web: { color: '#f8fafc', size: 0.22, label: 'web' },
    pdf: { color: '#94a3b8', size: 0.22, label: 'document' },
    markdown: { color: '#94a3b8', size: 0.22, label: 'document' },
    image: { color: '#f8fafc', size: 0.42, label: 'image' },
};
const RING_LEGEND = [{ label: 'podcast', color: 'transparent', size: 0 }, RINGS.web, RINGS.pdf, RINGS.image];
const TYPE_LABELS: Record<string, string> = { audio: 'podcast', pdf: 'PDF', markdown: 'note', web: 'web page', image: 'image' };
const SETTINGS_KEY = 'zettlecast.graph.settings.v2'; // v2: d3 forces
const MAX_INSIGHT_CHIPS = 6;

const NodeRingProgram = createNodeBorderProgram({
    borders: [
        { size: { attribute: 'ringSize', defaultValue: 0 }, color: { attribute: 'ringColor', defaultValue: 'transparent' } },
        { size: { fill: true }, color: { attribute: 'color' } },
    ],
});

function paletteColor(id: number | null | undefined, overrides: Record<number, string>): string {
    if (id === null || id === undefined) return UNCLUSTERED;
    if (overrides[id]) return overrides[id];
    return id >= 0 && id < COMMUNITY_COLORS.length ? COMMUNITY_COLORS[id] : OTHER_COMMUNITY;
}

function yearColor(year: number, from: number, to: number): string {
    const t = to > from ? (year - from) / (to - from) : 1;
    const c = YEAR_OLD.map((v, i) => Math.round(v + (YEAR_NEW[i] - v) * t));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
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

function loadSettings(): GraphSettings {
    try {
        const saved = window.localStorage.getItem(SETTINGS_KEY);
        return mergeSettings(saved ? JSON.parse(saved) : null);
    } catch {
        return mergeSettings(null);
    }
}

interface SimNode extends SimulationNodeDatum {
    id: string;
    community: number | null;
    size: number;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
    value: number;
    inside: boolean; // both ends in the same community
}

/** Pulls each note towards its community's centre (Obsidian has no such force; it's what groups them) */
function communityForce() {
    let nodes: SimNode[] = [];
    let strength = 0;
    const force = (alpha: number) => {
        const sums = new Map<number, { x: number; y: number; n: number }>();
        for (const n of nodes) {
            if (n.community === null) continue;
            const c = sums.get(n.community) ?? { x: 0, y: 0, n: 0 };
            sums.set(n.community, { x: c.x + n.x!, y: c.y + n.y!, n: c.n + 1 });
        }
        for (const n of nodes) {
            const c = n.community === null ? undefined : sums.get(n.community);
            if (!c || c.n < 2) continue;
            n.vx! += (c.x / c.n - n.x!) * strength * alpha;
            n.vy! += (c.y / c.n - n.y!) * strength * alpha;
        }
    };
    force.initialize = (ns: SimNode[]) => {
        nodes = ns;
    };
    force.strength = (k: number) => {
        strength = k;
        return force;
    };
    return force;
}

function createSimulation(): Simulation<SimNode, SimLink> {
    return forceSimulation<SimNode, SimLink>()
        .force('link', forceLink<SimNode, SimLink>().id((d) => d.id))
        .force('charge', forceManyBody<SimNode>().distanceMax(600))
        .force('x', forceX<SimNode>(0))
        .force('y', forceY<SimNode>(0))
        .force('collide', forceCollide<SimNode>())
        .force('community', communityForce())
        .stop();
}

/** Apply the pane's forces (and node sizes, for collisions) to the simulation */
function applyForces(sim: Simulation<SimNode, SimLink>, forces: GraphSettings['forces'], nodeSize: number): void {
    const links = (sim.force('link') as ReturnType<typeof forceLink<SimNode, SimLink>>);
    const degree = new Map<string, number>();
    links.links().forEach((l) => [l.source, l.target].forEach((end) => {
        const id = typeof end === 'object' ? end.id : String(end);
        degree.set(id, (degree.get(id) ?? 0) + 1);
    }));
    const deg = (end: SimNode | string | number) => degree.get(typeof end === 'object' ? end.id : String(end)) ?? 1;
    // links across communities weaken as the community pull grows, so groups can separate
    const across = 1 / (1 + 10 * forces.communityPull);
    links
        .distance(forces.distance)
        // d3's default strength (1 / smaller degree, here its root: hubs hold their notes closer)
        .strength((l) => (forces.link * (l.inside ? 1 : across) * (0.5 + l.value)) / Math.sqrt(Math.min(deg(l.source), deg(l.target))));
    (sim.force('charge') as ReturnType<typeof forceManyBody<SimNode>>).strength(-forces.repel);
    (sim.force('x') as ReturnType<typeof forceX<SimNode>>).strength(forces.center);
    (sim.force('y') as ReturnType<typeof forceY<SimNode>>).strength(forces.center);
    (sim.force('collide') as ReturnType<typeof forceCollide<SimNode>>).radius((n) => n.size * nodeSize + 2);
    (sim.force('community') as ReturnType<typeof communityForce>).strength(forces.communityPull);
}

/** Start each community around its own point on a circle */
function seedPositions(nodes: SimNode[]): void {
    const order = [...new Set(nodes.map((n) => n.community ?? -1))];
    const random = mulberry32(42);
    nodes.forEach((n) => {
        const angle = (2 * Math.PI * order.indexOf(n.community ?? -1)) / Math.max(order.length, 1);
        n.x = Math.cos(angle) * 300 + (random() - 0.5) * 60;
        n.y = Math.sin(angle) * 300 + (random() - 0.5) * 60;
        n.vx = 0;
        n.vy = 0;
    });
}

function nodeAttributes(n: GraphNode) {
    const ring = RINGS[n.source_type];
    return {
        label: n.name,
        sourceType: n.source_type,
        community: n.community,
        tags: n.tags ?? [],
        people: n.people ?? [],
        organisations: n.organisations ?? [],
        source: n.source ?? n.source_type,
        date: n.date,
        duration: n.duration,
        type: 'ring',
        ringColor: ring?.color ?? 'transparent',
        ringSize: ring?.size ?? 0,
        x: 0,
        y: 0,
        size: 6,
    };
}

/** Every note and link; the view shows a filtered copy of it */
function buildFullGraph(data: GraphData): Graph {
    const graph = new Graph({ type: 'undirected' });
    data.nodes.forEach((n) => graph.addNode(n.id, nodeAttributes(n)));
    data.links.forEach((l) => {
        if (graph.hasNode(l.source) && graph.hasNode(l.target) && !graph.hasEdge(l.source, l.target)) {
            graph.addEdge(l.source, l.target, {
                value: l.value,
                kind: l.kind,
                size: l.kind === 'link' ? 1.5 : 0.4 + l.value * 2,
                color: l.kind === 'link' ? '#cbd5e1' : '#475569',
            });
        }
    });
    // sizes are in layout units (sigma draws them relative to the positions, so they scale with zoom)
    graph.forEachNode((node) => graph.setNodeAttribute(node, 'size', 4 + Math.sqrt(graph.degree(node)) * 1.6));
    return graph;
}

/** What stands out in each community: values much more common there than overall */
function communityInsights(data: GraphData, colors: Record<number, string>): CommunityInsight[] {
    const total = data.nodes.length;
    const overall = { tags: new Map<string, number>(), people: new Map<string, number>(), organisations: new Map<string, number>() };
    const count = (map: Map<string, number>, values: string[]) =>
        new Set(values).forEach((v) => map.set(v, (map.get(v) ?? 0) + 1));
    data.nodes.forEach((n) => {
        count(overall.tags, n.tags ?? []);
        count(overall.people, n.people ?? []);
        count(overall.organisations, n.organisations ?? []);
    });
    return data.communities.map((c) => {
        const members = data.nodes.filter((n) => n.community === c.id);
        const top = (field: 'tags' | 'people' | 'organisations'): [string, number][] => {
            const inside = new Map<string, number>();
            members.forEach((n) => count(inside, n[field] ?? []));
            return [...inside.entries()]
                .filter(([v, k]) => {
                    const all = overall[field].get(v) ?? k;
                    // in at least 2 notes here, 1.5x as common as overall, and not in half of
                    // everything (the host)
                    return k >= 2 && (k / members.length) / (all / total) >= 1.5 && all < total / 2;
                })
                .sort((a, b) => b[1] - a[1])
                .slice(0, MAX_INSIGHT_CHIPS);
        };
        const dates = members.map((n) => n.date).filter((d): d is string => !!d).sort();
        return {
            id: c.id,
            label: c.label,
            size: members.length,
            color: paletteColor(c.id, colors),
            topics: top('tags'),
            people: top('people'),
            organisations: top('organisations'),
            from: dates[0] ?? null,
            to: dates[dates.length - 1] ?? null,
            hours: members.reduce((s, n) => s + (n.duration ?? 0), 0) / 3600,
        };
    });
}

function facetsOf(data: GraphData): Facets {
    const byCount = (values: string[]) => {
        const counts = new Map<string, number>();
        values.forEach((v) => counts.set(v, (counts.get(v) ?? 0) + 1));
        return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v);
    };
    return {
        tag: byCount(data.nodes.flatMap((n) => n.tags ?? [])),
        person: byCount(data.nodes.flatMap((n) => n.people ?? [])),
        org: byCount(data.nodes.flatMap((n) => n.organisations ?? [])),
        source: byCount(data.nodes.map((n) => n.source)),
    };
}

/** Node and link ids that pass the filters */
function visibleSet(full: Graph, s: GraphSettings['filter'], selected: string | null) {
    const match = compileQuery(s.query);
    const nodes = new Set<string>();
    full.forEachNode((node, a) => {
        if (s.hiddenTypes.includes(a.sourceType)) return;
        if (a.community !== null && s.hiddenCommunities.includes(a.community)) return;
        const year = Number(String(a.date ?? '').slice(0, 4));
        if (s.yearFrom !== null && !(year >= s.yearFrom)) return;
        if (s.yearTo !== null && !(year <= s.yearTo)) return;
        if (match && !match(a as never)) return;
        nodes.add(node);
    });
    const edges = new Set<string>();
    full.forEachEdge((edge, a, source, target) => {
        if (!nodes.has(source) || !nodes.has(target)) return;
        if (!s.kinds[a.kind as keyof typeof s.kinds]) return;
        if (a.kind !== 'link' && a.value < s.minWeight) return;
        edges.add(edge);
    });
    if (s.local && selected && full.hasNode(selected)) {
        // the selected note's neighbourhood over the visible links
        nodes.add(selected);
        const keep = new Set([selected]);
        let frontier = [selected];
        for (let d = 0; d < s.depth; d++) {
            const next: string[] = [];
            frontier.forEach((node) => full.forEachEdge(node, (edge, _a, source, target) => {
                const other = source === node ? target : source;
                if (edges.has(edge) && !keep.has(other)) {
                    keep.add(other);
                    next.push(other);
                }
            }));
            frontier = next;
        }
        [...nodes].forEach((n) => !keep.has(n) && nodes.delete(n));
        [...edges].forEach((e) => {
            const [a, b] = full.extremities(e);
            if (!keep.has(a) || !keep.has(b)) edges.delete(e);
        });
    }
    if (!s.orphans) {
        const linked = new Set<string>();
        edges.forEach((e) => full.extremities(e).forEach((n) => linked.add(n)));
        [...nodes].forEach((n) => !linked.has(n) && n !== selected && nodes.delete(n));
    }
    return { nodes, edges };
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
    const [settings, setSettings] = useState<GraphSettings>(loadSettings);
    const [paneOpen, setPaneOpen] = useState(true);
    const [visibleCount, setVisibleCount] = useState({ nodes: 0, links: 0 });

    const view = useMemo(() => new Graph({ type: 'undirected' }), []);
    const full = useMemo(() => (data ? buildFullGraph(data) : null), [data]);
    // layout state per note (position, velocity, pinned while dragged), kept for notes filtered
    // out so they come back where they were
    const sim = useMemo(() => createSimulation(), []);
    const simNodes = useRef(new Map<string, SimNode>());
    const drag = useRef({ node: null as string | null, moved: false, fitOnEnd: false });
    const communities = useMemo(() => data?.communities ?? [], [data]);
    const insights = useMemo(
        () => (data ? communityInsights(data, settings.groups.communityColors) : []),
        [data, settings.groups.communityColors],
    );
    const facets = useMemo<Facets>(() => (data ? facetsOf(data) : { tag: [], person: [], org: [], source: [] }), [data]);
    const years = useMemo(
        () => [...new Set((data?.nodes ?? []).map((n) => Number(n.date?.slice(0, 4))).filter(Number.isFinite))].sort(),
        [data],
    );
    const types = useMemo(() => {
        const counts = new Map<string, number>();
        (data?.nodes ?? []).forEach((n) => counts.set(n.source_type, (counts.get(n.source_type) ?? 0) + 1));
        return [...counts.entries()].map(([type, count]) => ({ type, label: TYPE_LABELS[type] ?? type, count }));
    }, [data]);

    useEffect(() => {
        try {
            window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch {
            // private window or blocked storage: settings last for this visit only
        }
    }, [settings]);

    useEffect(() => {
        api.getGraph()
            .then(setData)
            .catch((err) => {
                console.error(err);
                setError('Failed to load graph data');
            });
    }, []);

    const fit = useCallback(() => {
        const renderer = sigmaRef.current;
        if (!renderer) return;
        // the camera frames the graph as it is now and then stays put while the layout moves
        renderer.setCustomBBox(null);
        renderer.refresh();
        renderer.setCustomBBox(renderer.getBBox());
        renderer.getCamera().animatedReset({ duration: 400 });
    }, []);

    // --- Live layout (d3-force): runs while it has energy, then rests ---

    useEffect(() => {
        sim.on('tick', () => {
            const nodes = simNodes.current;
            view.updateEachNodeAttributes((node, a) => {
                const n = nodes.get(node);
                return n ? { ...a, x: n.x, y: n.y } : a;
            }, { attributes: ['x', 'y'] });
        });
        sim.on('end', () => {
            if (drag.current.fitOnEnd) {
                drag.current.fitOnEnd = false;
                fit();
            }
        });
        return () => {
            sim.stop();
            sim.on('tick', null).on('end', null);
        };
    }, [sim, view, fit]);

    // Lay every note out once, before the first frame, so the graph opens settled
    useEffect(() => {
        if (!full) return;
        const nodes = full.mapNodes((id, a) => ({ id, community: a.community ?? null, size: a.size } as SimNode));
        seedPositions(nodes);
        simNodes.current = new Map(nodes.map((n) => [n.id, n]));
        const prelayout = createSimulation().nodes(nodes);
        const byId = simNodes.current;
        (prelayout.force('link') as ReturnType<typeof forceLink<SimNode, SimLink>>).links(
            full.mapEdges((_, a, source, target) => ({
                source, target, value: a.value as number,
                inside: byId.get(source)!.community !== null && byId.get(source)!.community === byId.get(target)!.community,
            })),
        );
        applyForces(prelayout, settings.forces, settings.display.nodeSize);
        prelayout.tick(300);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [full]);

    // --- The view graph follows the filters ---

    // the selection changes the visible notes only in neighbourhood mode
    const localCentre = settings.filter.local ? selected : null;
    useEffect(() => {
        if (!full) return;
        const { nodes, edges } = visibleSet(full, settings.filter, localCentre);
        const layout = simNodes.current;
        const changed = view.order > 0;
        view.forEachNode((node) => !nodes.has(node) && view.dropNode(node));
        nodes.forEach((node) => {
            if (!view.hasNode(node)) {
                const n = layout.get(node);
                view.addNode(node, { ...full.getNodeAttributes(node), x: n?.x ?? 0, y: n?.y ?? 0 });
            }
        });
        view.forEachEdge((edge) => !edges.has(edge) && view.dropEdge(edge));
        edges.forEach((edge) => {
            if (!view.hasEdge(edge)) {
                const [a, b] = full.extremities(edge);
                view.addEdgeWithKey(edge, a, b, { ...full.getEdgeAttributes(edge) });
            }
        });
        setVisibleCount({ nodes: view.order, links: view.size });

        sim.nodes(view.mapNodes((node) => layout.get(node)!));
        (sim.force('link') as ReturnType<typeof forceLink<SimNode, SimLink>>).links(
            view.mapEdges((_, a, source, target) => ({
                source, target, value: a.value as number,
                inside: layout.get(source)!.community !== null && layout.get(source)!.community === layout.get(target)!.community,
            })),
        );
        applyForces(sim, settings.forces, settings.display.nodeSize);
        if (changed) {
            // the remaining notes regroup, then the camera frames them
            drag.current.fitOnEnd = true;
            sim.alpha(0.5).restart();
        } else {
            sim.alpha(0.05).restart();
        }
        // forces and sizes are applied by their own effect
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [full, view, sim, settings.filter, localCentre]);

    useEffect(() => {
        applyForces(sim, settings.forces, settings.display.nodeSize);
        if (sim.nodes().length) sim.alpha(Math.max(sim.alpha(), 0.3)).restart();
    }, [sim, settings.forces, settings.display.nodeSize]);

    // Create the renderer once the data is in
    useEffect(() => {
        if (!full || !containerRef.current) return;
        const renderer = new Sigma(view, containerRef.current, {
            nodeProgramClasses: { ring: NodeRingProgram },
            labelColor: { color: '#e2e8f0' },
            labelFont: 'Inter, Segoe UI, sans-serif',
            labelSize: 13,
            labelWeight: '500',
            // titles only for well-linked notes at this zoom; the rest on hover or zooming in
            labelDensity: 0.25,
            labelGridCellSize: 150,
            labelRenderedSizeThreshold: 14,
            // sizes are in layout units, like the collision radius, so nodes never overlap
            // on screen and grow as you zoom in (titles then appear, as in Obsidian)
            itemSizesReference: 'positions',
            zoomToSizeRatioFunction: (ratio) => ratio,
            defaultEdgeColor: '#475569',
            zIndex: true,
            stagePadding: 160, // room for labels and the toolbar around the outermost notes
            defaultDrawNodeHover: drawHover,
        });
        renderer.setCustomBBox(renderer.getBBox());
        const state = drag.current;

        renderer.on('enterNode', ({ node }) => setHovered(node));
        renderer.on('leaveNode', () => setHovered(null));
        renderer.on('clickNode', ({ node }) => {
            if (!state.moved) setSelected(node);
        });
        renderer.on('doubleClickNode', ({ node, event }) => {
            event.preventSigmaDefault();
            router.push(`/notes/${node}`);
        });
        renderer.on('clickStage', () => {
            setSelected(null);
            setFocusCommunity(null);
        });

        // drag a note: it stays under the pointer and the layout follows it
        renderer.on('downNode', ({ node }) => {
            const n = simNodes.current.get(node);
            if (!n) return;
            state.node = node;
            state.moved = false;
            n.fx = n.x;
            n.fy = n.y;
            sim.alphaTarget(0.3).restart();
        });
        renderer.on('moveBody', ({ event }) => {
            const n = state.node ? simNodes.current.get(state.node) : undefined;
            if (!n) return;
            const pos = renderer.viewportToGraph(event);
            n.fx = pos.x;
            n.fy = pos.y;
            state.moved = true;
            event.preventSigmaDefault();
            event.original.preventDefault();
            event.original.stopPropagation();
        });
        const release = () => {
            const n = state.node ? simNodes.current.get(state.node) : undefined;
            state.node = null;
            if (!n) return;
            n.fx = null;
            n.fy = null;
            sim.alphaTarget(0);
        };
        renderer.on('upNode', release);
        renderer.on('upStage', release);

        // community names just above their notes, kept in place as the camera and layout move
        // (from sigma's own display positions, so they line up with what is drawn)
        renderer.on('afterRender', () => {
            const spans = new Map<number, { x: number; top: number; n: number }>();
            view.forEachNode((node, a) => {
                if (a.community === null || a.community === undefined) return;
                const d = renderer.getNodeDisplayData(node);
                if (!d) return;
                const p = renderer.framedGraphToViewport({ x: d.x, y: d.y });
                const c = spans.get(a.community) ?? { x: 0, top: Infinity, n: 0 };
                spans.set(a.community, { x: c.x + p.x, top: Math.min(c.top, p.y), n: c.n + 1 });
            });
            setMapLabels(
                communities.flatMap((c) => {
                    const span = spans.get(c.id);
                    if (!span || span.n < 2) return [];
                    return [{ id: c.id, label: c.label, x: span.x / span.n, y: span.top - 18 }];
                }),
            );
        });
        sigmaRef.current = renderer;
        return () => {
            renderer.kill();
            sigmaRef.current = null;
        };
    }, [full, view, sim, communities, router]);

    // --- Appearance: colours, sizes, highlight ---

    const colorOf = useMemo(() => {
        const g = settings.groups;
        const custom = g.custom
            .map((group) => ({ match: compileQuery(group.query), color: group.color }))
            .filter((group) => group.match);
        const sources = facets.source;
        const typeOrder = types.map((t) => t.type);
        const [from, to] = [years[0] ?? 0, years[years.length - 1] ?? 0];
        return (a: Record<string, unknown>): string => {
            for (const group of custom) if (group.match!(a as never)) return group.color;
            switch (g.colorBy) {
                case 'year': {
                    const year = Number(String(a.date ?? '').slice(0, 4));
                    return Number.isFinite(year) && year ? yearColor(year, from, to) : UNCLUSTERED;
                }
                case 'source':
                    return COMMUNITY_COLORS[sources.indexOf(a.source as string)] ?? OTHER_COMMUNITY;
                case 'type':
                    return COMMUNITY_COLORS[typeOrder.indexOf(a.sourceType as string)] ?? OTHER_COMMUNITY;
                default:
                    return paletteColor(a.community as number | null, g.communityColors);
            }
        };
    }, [settings.groups, facets, types, years]);

    useEffect(() => {
        const renderer = sigmaRef.current;
        if (!renderer) return;
        const { display } = settings;
        const focus = hovered || selected;
        const neighbours = new Set(focus && view.hasNode(focus) ? view.neighbors(focus) : []);
        renderer.setSetting('labelRenderedSizeThreshold', display.textFade);
        renderer.setSetting('nodeReducer', (node, attrs) => {
            const shown = { ...attrs, color: colorOf(attrs), size: attrs.size * display.nodeSize };
            if (focus && view.hasNode(focus)) {
                if (node === focus) return { ...shown, highlighted: true, zIndex: 2 };
                if (neighbours.has(node)) return { ...shown, forceLabel: true, zIndex: 1 };
                return { ...shown, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            if (focusCommunity !== null && attrs.community !== focusCommunity) {
                return { ...shown, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            return shown;
        });
        renderer.setSetting('edgeReducer', (edge, attrs) => {
            const shown = { ...attrs, size: attrs.size * display.linkThickness, hidden: display.linkThickness === 0 };
            if (focus && view.hasNode(focus)) {
                return view.hasExtremity(edge, focus) ? { ...shown, color: '#94a3b8', zIndex: 1 } : { ...shown, hidden: true };
            }
            if (focusCommunity !== null) {
                const [a, b] = view.extremities(edge);
                const inside = view.getNodeAttribute(a, 'community') === focusCommunity
                    && view.getNodeAttribute(b, 'community') === focusCommunity;
                return inside ? shown : { ...shown, hidden: true };
            }
            return shown;
        });
    }, [view, full, hovered, selected, focusCommunity, settings, colorOf]);

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
        if (!full || query.trim().length < 2) return [];
        const q = query.toLowerCase();
        return full.filterNodes((_, attrs) => String(attrs.label).toLowerCase().includes(q)).slice(0, 8);
    }, [full, query]);

    const focusNode = (node: string) => {
        setSelected(node);
        setQuery('');
        const renderer = sigmaRef.current;
        const display = renderer?.getNodeDisplayData(node);
        if (renderer && display) {
            renderer.getCamera().animate({ x: display.x, y: display.y, ratio: 0.5 }, { duration: 500 });
        }
    };

    const focusOnCommunity = (id: number | null) => {
        setFocusCommunity(id);
        setSelected(null);
        const renderer = sigmaRef.current;
        if (!renderer || id === null) return;
        const points = view.filterNodes((_, a) => a.community === id)
            .map((n) => renderer.getNodeDisplayData(n))
            .filter((d): d is NonNullable<typeof d> => !!d);
        if (points.length === 0) return;
        const xs = points.map((p) => p.x);
        const ys = points.map((p) => p.y);
        const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
        renderer.getCamera().animate(
            { x: (Math.max(...xs) + Math.min(...xs)) / 2, y: (Math.max(...ys) + Math.min(...ys)) / 2,
              ratio: Math.min(1, Math.max(0.2, span * 1.4)) },
            { duration: 500 },
        );
    };

    // Obsidian's "Animate": scatter the visible notes and watch them find their places
    const animate = () => {
        seedPositions(sim.nodes());
        drag.current.fitOnEnd = true;
        fit();
        sim.alpha(1).restart();
    };

    const selectedAttrs = selected && full?.hasNode(selected) ? full.getNodeAttributes(selected) : null;
    const selectedLinks = links && links.uuid === selected ? links.items : null;
    const selectedCommunity = selectedAttrs ? communities.find((c) => c.id === selectedAttrs.community) : undefined;

    const legend = (
        <div className="gp-legend">
            <span className="gp-label">Ring</span>
            {RING_LEGEND.map((r) => (
                <span key={r.label} className="legend-item">
                    <span
                        className="legend-ring"
                        style={{ border: r.size ? `${Math.round(r.size * 10)}px solid ${r.color}` : '1px dashed #64748b' }}
                    />
                    {r.label}
                </span>
            ))}
        </div>
    );

    return (
        <div className="graph-view">
            <div className="graph-toolbar">
                <Link href="/notes" className="btn btn-secondary">← Notes</Link>
                <h1 className="graph-title">Knowledge Graph</h1>
                {full && (
                    <span className="text-muted graph-counts">
                        {full.order} notes · {full.size} links · {communities.length} communities
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
                                    <button onClick={() => focusNode(node)}>{full?.getNodeAttribute(node, 'label')}</button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
                <button className="btn btn-secondary" onClick={fit}>Fit</button>
                <button className="btn btn-secondary" aria-pressed={paneOpen} onClick={() => setPaneOpen(!paneOpen)}
                        title="Filters, groups, display and forces">
                    ⚙ Graph settings
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

            {!selected && !hovered && settings.display.communityLabels && settings.groups.colorBy === 'community' && (
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

            <div className="graph-hint">Drag notes · click to see why they link · double-click to open</div>

            {data && (
                <GraphPane
                    open={paneOpen}
                    onClose={() => setPaneOpen(false)}
                    settings={settings}
                    onChange={setSettings}
                    types={types}
                    years={years}
                    communities={insights}
                    facets={facets}
                    visible={visibleCount}
                    hasSelection={!!selected}
                    focusCommunity={focusCommunity}
                    onFocusCommunity={focusOnCommunity}
                    onAnimate={animate}
                    legend={legend}
                />
            )}

            {selectedAttrs && full && (
                <aside className="graph-panel">
                    <div className="graph-panel-type" style={{ color: colorOf(selectedAttrs) }}>
                        {TYPE_LABELS[selectedAttrs.sourceType] ?? selectedAttrs.sourceType}
                        {selectedCommunity ? ` · ${selectedCommunity.label}` : ''}
                    </div>
                    <h2 className="graph-panel-title">{selectedAttrs.label}</h2>
                    <div className="graph-panel-actions">
                        <Link href={`/notes/${selected}`} className="btn btn-primary">Open note</Link>
                        <button className="btn btn-secondary" onClick={() => setSelected(null)}>Close</button>
                    </div>
                    <h3 className="graph-panel-heading">Linked notes</h3>
                    {!selectedLinks && <span className="text-muted">Loading…</span>}
                    {selectedLinks && selectedLinks.length === 0 && <span className="text-muted">No links yet.</span>}
                    {selectedLinks && selectedLinks.length > 0 && (
                        <ul className="graph-panel-links">
                            {selectedLinks.map((s) => (
                                <li key={s.uuid}>
                                    <div className="graph-panel-link-row">
                                        <button onClick={() => full.hasNode(s.uuid) && focusNode(s.uuid)}>{s.title}</button>
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
