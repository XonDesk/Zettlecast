'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Graph from 'graphology';
import Sigma from 'sigma';
import type { NodeHoverDrawingFunction } from 'sigma/rendering';
import { createNodeBorderProgram } from '@sigma/node-border';
import { api } from '@/lib/api';
import { compileQuery } from '@/lib/graphQuery';
import type { GraphData, GraphNode, LinkSuggestion, SearchResult } from '@/lib/types';
import { formatTimestamp } from '@/lib/utils';
import { type ContourLayer, type ContourPoint, densityContours, drawContours } from './graphContours';
import { type LayoutMessage, type LayoutRequest, mulberry32, seedPositions, type SimNode } from './graphForces';
import GraphPane, {
    type CommunityInsight,
    type Facets,
    type GraphSettings,
    KIND_COLORS,
    type LinkKind,
    mergeSettings,
} from './GraphPane';

// Community fills: one hue per community, as Obsidian's groups. Twelve is more than the eye
// can tell apart in every pair, so the communities are also named on the map and the colour of
// any of them can be changed in the pane. Unless changed, colours are picked per graph so linked
// communities (neighbours on the map) look as different as possible.
// Colour follows the community id, never its rank; ids past the palette get generated colours.
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
    reddit: { color: '#ff7a45', size: 0.3, label: 'reddit' },
};
const KIND_RING = 0.38; // outline of a focused note's neighbours (thicker than the source-type rings)
const KIND_RANK: Record<LinkKind, number> = { similar: 0, duplicate: 1, link: 2 };
const RING_LEGEND = [{ label: 'podcast', color: 'transparent', size: 0 }, RINGS.web, RINGS.pdf, RINGS.image, RINGS.reddit];
const TYPE_LABELS: Record<string, string> = { audio: 'podcast', pdf: 'PDF', markdown: 'note', web: 'web page', image: 'image', reddit: 'Reddit save' };
const SETTINGS_KEY = 'zettlecast.graph.settings.v2'; // v2: d3 forces
const MAX_INSIGHT_CHIPS = 6;
const CONTOUR_INTERVAL = 1000; // ms between contour recomputes while the layout moves (again when it rests)
const LAYOUT_KEY = 'zettlecast.graph.layout.v1'; // last settled positions, so reopening skips the pre-layout
const PRELAYOUT_TICKS = 300;
const PRELAYOUT_TICKS_CACHED = 40; // most notes start where they were: settle only the new ones in
const SEARCH_MIN_CHARS = 3;
const SEARCH_DEBOUNCE_MS = 450; // passage search runs the embedder + reranker: wait for a pause in typing
const SEARCH_RESULTS = 12;

/** A search snippet without its leading "[12.3s - 45.6s]" (the result shows the time), shortened. */
function snippetText(snippet: string): string {
    const text = snippet.replace(/^\[[\d.]+s(?: - [\d.]+s)?\]\s*/, '');
    return text.length > 220 ? `${text.slice(0, 220)}…` : text;
}

const NodeRingProgram = createNodeBorderProgram({
    borders: [
        { size: { attribute: 'ringSize', defaultValue: 0 }, color: { attribute: 'ringColor', defaultValue: 'transparent' } },
        { size: { fill: true }, color: { attribute: 'color' } },
    ],
});

function hslHex(h: number, s: number, l: number): string {
    const a = s * Math.min(l, 1 - l);
    const f = (n: number) => {
        const k = (n + h / 30) % 12;
        return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
}

/** The colour for a community id: the palette, then golden-angle hues (stable per id, far from the last). */
function defaultColor(id: number): string {
    if (id < 0) return OTHER_COMMUNITY;
    if (id < COMMUNITY_COLORS.length) return COMMUNITY_COLORS[id];
    const k = id - COMMUNITY_COLORS.length;
    return hslHex((15 + k * 137.508) % 360, 0.62, k % 2 ? 0.7 : 0.56);
}

function paletteColor(id: number | null | undefined, overrides: Record<number, string>): string {
    if (id === null || id === undefined) return UNCLUSTERED;
    return overrides[id] ?? defaultColor(id);
}

function colorDistance(a: string, b: string): number {
    // "redmean" RGB distance: cheap and close enough to perceived difference
    const rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const [r1, g1, b1] = rgb(a);
    const [r2, g2, b2] = rgb(b);
    const r = (r1 + r2) / 2;
    return Math.sqrt((2 + r / 256) * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + (2 + (255 - r) / 256) * (b1 - b2) ** 2);
}

/**
 * A colour for every community, each one different, chosen so communities that share many links
 * (they sit next to each other on the map) look as different as possible. Largest first: they
 * take up the most of the map, so they get the clearest contrasts.
 */
function autoCommunityColors(data: GraphData): Record<number, string> {
    const ids = data.communities.map((c) => c.id);
    const size = new Map(ids.map((id) => [id, 0]));
    const communityOf = new Map<string, number>();
    data.nodes.forEach((n) => {
        if (n.community === null || n.community === undefined) return;
        communityOf.set(n.id, n.community);
        size.set(n.community, (size.get(n.community) ?? 0) + 1);
    });
    const shared = new Map<string, number>();
    const key = (a: number, b: number) => (a < b ? `${a}:${b}` : `${b}:${a}`);
    data.links.forEach((l) => {
        const a = communityOf.get(l.source);
        const b = communityOf.get(l.target);
        if (a === undefined || b === undefined || a === b) return;
        shared.set(key(a, b), (shared.get(key(a, b)) ?? 0) + (l.value ?? 1));
    });
    const candidates = Array.from({ length: Math.max(ids.length, COMMUNITY_COLORS.length) }, (_, i) => defaultColor(i));
    const out: Record<number, string> = {};
    const order = [...ids].sort((a, b) => (size.get(b) ?? 0) - (size.get(a) ?? 0));
    for (const id of order) {
        const free = candidates.filter((c) => !Object.values(out).includes(c));
        let best = free[0];
        let bestPenalty = Infinity;
        for (const colour of free) {
            let penalty = 0;
            for (const [other, assigned] of Object.entries(out)) {
                const w = shared.get(key(id, Number(other))) ?? 0;
                // close colours on linked communities cost the most; any two alike cost a little
                penalty += (w + 0.05) * Math.exp(-colorDistance(colour, assigned) / 120);
            }
            if (penalty < bestPenalty) {
                bestPenalty = penalty;
                best = colour;
            }
        }
        out[id] = best;
    }
    return out;
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

function loadSettings(): GraphSettings {
    try {
        const saved = window.localStorage.getItem(SETTINGS_KEY);
        return mergeSettings(saved ? JSON.parse(saved) : null);
    } catch {
        return mergeSettings(null);
    }
}

type LayoutCache = Record<string, [number, number]>;

function loadLayout(): LayoutCache {
    try {
        return JSON.parse(window.localStorage.getItem(LAYOUT_KEY) ?? '{}') as LayoutCache;
    } catch {
        return {};
    }
}

function saveLayout(positions: Map<string, [number, number]>): void {
    const out: LayoutCache = {};
    positions.forEach(([x, y], id) => {
        if (Number.isFinite(x) && Number.isFinite(y)) out[id] = [Math.round(x), Math.round(y)];
    });
    try {
        window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(out));
    } catch {
        /* storage full or blocked: the next open lays out from scratch */
    }
}

/**
 * Positions from the last visit; a note that wasn't there starts next to a linked note that was.
 * Returns how many notes had a saved position.
 */
function restorePositions(nodes: SimNode[], neighbours: (id: string) => string[]): number {
    const saved = loadLayout();
    let known = 0;
    for (const n of nodes) {
        const p = saved[n.id];
        if (p) {
            [n.x, n.y] = p;
            known++;
        }
    }
    const random = mulberry32(7);
    for (const n of nodes) {
        if (saved[n.id]) continue;
        const anchor = neighbours(n.id).map((id) => saved[id]).find(Boolean);
        if (anchor) {
            n.x = anchor[0] + (random() - 0.5) * 40;
            n.y = anchor[1] + (random() - 0.5) * 40;
        }
    }
    return known;
}

function nodeAttributes(n: GraphNode) {
    const ring = RINGS[n.source_type];
    return {
        label: n.name,
        brief: n.brief ?? '', // what the note is: the hover bar and the panel show it
        image: n.image ?? false,
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
function visibleSet(full: Graph, s: GraphSettings['filter']) {
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
    if (!s.orphans) {
        const linked = new Set<string>();
        edges.forEach((e) => full.extremities(e).forEach((n) => linked.add(n)));
        [...nodes].forEach((n) => !linked.has(n) && nodes.delete(n));
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
    // passage search (embeddings + reranker): the notes found stay highlighted until the box is cleared
    const [hits, setHits] = useState<{ query: string; results: SearchResult[] } | null>(null);
    const [searching, setSearching] = useState(false);
    const [resultsOpen, setResultsOpen] = useState(false);
    const [links, setLinks] = useState<{ uuid: string; items: LinkSuggestion[] } | null>(null);
    // community names on the map: placed straight in the DOM after each frame (React state per frame
    // re-rendered the whole page at every camera or layout move)
    const labelEls = useRef(new Map<number, HTMLSpanElement>());
    const [settings, setSettings] = useState<GraphSettings>(loadSettings);
    const [paneOpen, setPaneOpen] = useState(true);
    const [visibleCount, setVisibleCount] = useState({ nodes: 0, links: 0 });
    // kept screenshots' thumbnails, loaded once the graph is up so hovering shows them at once
    // (held here so the browser keeps them decoded)
    const thumbs = useRef(new Map<string, HTMLImageElement>());

    const view = useMemo(() => new Graph({ type: 'undirected' }), []);
    const full = useMemo(() => (data ? buildFullGraph(data) : null), [data]);
    // the force layout runs in a worker (graphLayout.worker.ts); the page keeps the latest position
    // of every note, filtered out or not, so a note comes back where it was
    const layout = useRef<Worker | null>(null);
    const layoutIds = useRef<string[]>([]); // order of the worker's positions
    const positions = useRef(new Map<string, [number, number]>());
    const pendingXY = useRef<{ xy: Float32Array | null; frame: number }>({ xy: null, frame: 0 });
    // the graph whose first layout is done: the view and the renderer wait for it
    const [laidOut, setLaidOut] = useState<Graph | null>(null);
    const layoutFor = useRef<Graph | null>(null);
    // frame the notes once the layout comes to rest (after a filter change)
    const fitOnEnd = useRef(false);
    // density contours: recomputed (at most every CONTOUR_INTERVAL ms) when the layout or what
    // they show changes, redrawn on every frame for the camera
    const contours = useRef({ layers: [] as ContourLayer[], dirty: true, last: 0, timer: 0 });
    const contourInput = useRef({
        enabled: false, bandwidth: 20, dim: false, focusCommunity: null as number | null,
        colorOf: (() => UNCLUSTERED) as (a: Record<string, unknown>) => string,
    });
    const communities = useMemo(() => data?.communities ?? [], [data]);
    const hitNodes = useMemo(
        () => new Set((hits?.results ?? []).map((r) => r.uuid).filter((id) => full?.hasNode(id))),
        [hits, full],
    );
    // picked for this graph (neighbours far apart); the colours chosen in the pane override them
    const autoColors = useMemo(() => (data ? autoCommunityColors(data) : {}), [data]);
    const communityColors = useMemo(
        () => ({ ...autoColors, ...settings.groups.communityColors }),
        [autoColors, settings.groups.communityColors],
    );
    const insights = useMemo(
        () => (data ? communityInsights(data, communityColors) : []),
        [data, communityColors],
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

    // on a phone the pane would cover the whole map: it starts closed there (Graph settings opens it)
    useEffect(() => {
        if (window.matchMedia('(max-width: 700px)').matches) setPaneOpen(false);
    }, []);

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

    // --- Force layout (graphLayout.worker.ts): runs while it has energy, then rests ---

    const postLayout = useCallback((request: LayoutRequest) => layout.current?.postMessage(request), []);

    useEffect(() => {
        const worker = new Worker(new URL('./graphLayout.worker.ts', import.meta.url));
        layout.current = worker;
        const pending = pendingXY.current;
        // positions as they come, applied at most once per frame (each apply makes sigma
        // re-upload every note and link)
        const take = () => {
            const xy = pending.xy;
            if (!xy) return;
            pending.xy = null;
            layoutIds.current.forEach((id, i) => positions.current.set(id, [xy[2 * i], xy[2 * i + 1]]));
        };
        const apply = () => {
            pending.frame = 0;
            take();
            view.updateEachNodeAttributes((node, a) => {
                const p = positions.current.get(node);
                return p ? { ...a, x: p[0], y: p[1] } : a;
            }, { attributes: ['x', 'y'] });
            contours.current.dirty = true;
        };
        worker.onmessage = (event: MessageEvent<LayoutMessage>) => {
            const m = event.data;
            if (m.type === 'positions') {
                pending.xy = m.xy;
                if (!pending.frame) pending.frame = requestAnimationFrame(apply);
            } else if (m.type === 'ready') {
                take();
                saveLayout(positions.current);
                setLaidOut(layoutFor.current);
            } else if (m.type === 'end') {
                if (pending.frame) {
                    cancelAnimationFrame(pending.frame);
                    apply();
                }
                saveLayout(positions.current);
                // contours for where the notes came to rest, without waiting out the interval
                contours.current.dirty = true;
                contours.current.last = 0;
                sigmaRef.current?.scheduleRender();
                if (fitOnEnd.current) {
                    fitOnEnd.current = false;
                    fit();
                }
            }
        };
        return () => {
            cancelAnimationFrame(pending.frame);
            pending.frame = 0;
            worker.terminate();
            layout.current = null;
        };
    }, [view, fit]);

    // Lay every note out once, before the first frame, so the graph opens settled
    useEffect(() => {
        if (!full) return;
        const nodes = full.mapNodes((id, a) => ({ id, community: a.community ?? null, size: a.size } as SimNode));
        seedPositions(nodes);
        // reopening: most notes have their last positions, so a short settle replaces the full layout
        const known = restorePositions(nodes, (id) => full.neighbors(id));
        const cached = known >= 0.8 * nodes.length;
        layoutIds.current = nodes.map((n) => n.id);
        positions.current = new Map(nodes.map((n) => [n.id, [n.x!, n.y!]]));
        layoutFor.current = full;
        postLayout({
            type: 'init',
            nodes,
            links: full.mapEdges((_, a, source, target) => ({ source, target, value: a.value as number })),
            forces: settings.forces,
            nodeSize: settings.display.nodeSize,
            alpha: cached ? 0.3 : 1,
            ticks: cached ? PRELAYOUT_TICKS_CACHED : PRELAYOUT_TICKS,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [full]);

    // --- The view graph follows the filters ---

    useEffect(() => {
        if (!full || laidOut !== full) return;
        const { nodes, edges } = visibleSet(full, settings.filter);
        const changed = view.order > 0;
        view.forEachNode((node) => !nodes.has(node) && view.dropNode(node));
        nodes.forEach((node) => {
            if (!view.hasNode(node)) {
                const [x, y] = positions.current.get(node) ?? [0, 0];
                view.addNode(node, { ...full.getNodeAttributes(node), x, y });
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

        if (changed) {
            // the remaining notes regroup, then the camera frames them
            fitOnEnd.current = true;
        }
        postLayout({
            type: 'view',
            ids: view.nodes(),
            links: view.mapEdges((_, a, source, target) => ({ source, target, value: a.value as number })),
            // first show: the pre-layout has placed every note, so nothing moves until asked (a live
            // settle here ran ~170 steps, each one re-uploading every note and link: seconds of stutter)
            alpha: changed ? 0.5 : null,
        });
    }, [full, laidOut, view, settings.filter, postLayout]);

    useEffect(() => {
        postLayout({ type: 'forces', forces: settings.forces, nodeSize: settings.display.nodeSize });
    }, [settings.forces, settings.display.nodeSize, postLayout]);

    // Create the renderer once the first layout is done
    useEffect(() => {
        if (!full || laidOut !== full || !containerRef.current) return;
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
            // room for labels and the toolbar around the outermost notes (on a phone 160 px a side
            // would leave the map a sliver)
            stagePadding: containerRef.current.clientWidth < 700 ? 24 : 160,
            defaultDrawNodeHover: drawHover,
        });
        renderer.setCustomBBox(renderer.getBBox());
        const c = contours.current;
        const contourCanvas = renderer.createCanvas('contours', {
            beforeLayer: 'edges',
            style: { width: '100%', height: '100%', pointerEvents: 'none' },
        });

        renderer.on('enterNode', ({ node }) => setHovered(node));
        renderer.on('leaveNode', () => setHovered(null));
        renderer.on('clickNode', ({ node }) => {
            setSelected(node);
        });
        renderer.on('doubleClickNode', ({ node, event }) => {
            event.preventSigmaDefault();
            router.push(`/notes/${node}`);
        });
        renderer.on('clickStage', () => {
            setSelected(null);
            setFocusCommunity(null);
        });

        // notes are not draggable: a drag reheats the whole layout, so even a click moved every note

        renderer.on('afterRender', () => {
            const input = contourInput.current;
            if (!input.enabled) {
                c.layers = [];
            } else if (c.dirty && performance.now() - c.last >= CONTOUR_INTERVAL) {
                const points: ContourPoint[] = [];
                view.forEachNode((_, a) => {
                    if (input.focusCommunity !== null && a.community !== input.focusCommunity) return;
                    points.push({ x: a.x, y: a.y, color: input.colorOf(a) });
                });
                c.layers = densityContours(points, input.bandwidth);
                c.dirty = false;
                c.last = performance.now();
            } else if (c.dirty && !c.timer) {
                // the layout moved since: catch up once the interval has passed
                c.timer = window.setTimeout(() => {
                    c.timer = 0;
                    renderer.scheduleRender();
                }, CONTOUR_INTERVAL);
            }
            drawContours(contourCanvas, c.layers, (p) => renderer.graphToViewport(p), input.dim ? 0.35 : 1);
        });

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
            labelEls.current.forEach((el, id) => {
                const span = spans.get(id);
                if (!span || span.n < 2) {
                    el.style.display = 'none';
                    return;
                }
                el.style.display = 'block';
                el.style.transform = `translate(${span.x / span.n}px, ${span.top - 18}px) translate(-50%, -50%)`;
            });
        });
        sigmaRef.current = renderer;
        return () => {
            window.clearTimeout(c.timer);
            c.timer = 0;
            renderer.kill();
            sigmaRef.current = null;
        };
    }, [full, laidOut, view, postLayout, communities, router]);

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
                    return defaultColor(sources.indexOf(a.source as string));
                case 'type':
                    return defaultColor(typeOrder.indexOf(a.sourceType as string));
                default:
                    return paletteColor(a.community as number | null, communityColors);
            }
        };
    }, [settings.groups, communityColors, facets, types, years]);

    useEffect(() => {
        contourInput.current = {
            ...contourInput.current,
            enabled: settings.display.contours,
            bandwidth: settings.display.contourBandwidth,
            focusCommunity,
            colorOf,
        };
        contours.current.dirty = true;
        sigmaRef.current?.scheduleRender();
    }, [settings.display.contours, settings.display.contourBandwidth, focusCommunity, colorOf, settings.filter, full]);

    // hovering or selecting a note only fades the contours: no recompute
    const contourDim = !!(hovered || selected);
    useEffect(() => {
        contourInput.current.dim = contourDim;
        sigmaRef.current?.scheduleRender();
    }, [contourDim]);

    useEffect(() => {
        const renderer = sigmaRef.current;
        if (!renderer) return;
        const { display } = settings;
        const focus = hovered || selected;
        // each neighbour by its strongest kind of link to the focused note
        const neighbours = new Map<string, LinkKind>();
        if (focus && view.hasNode(focus)) {
            view.forEachEdge(focus, (_edge, a, source, target) => {
                const other = source === focus ? target : source;
                const kind = a.kind as LinkKind;
                const had = neighbours.get(other);
                if (!had || KIND_RANK[kind] > KIND_RANK[had]) neighbours.set(other, kind);
            });
        }
        renderer.setSetting('labelRenderedSizeThreshold', display.textFade);
        renderer.setSetting('nodeReducer', (node, attrs) => {
            const shown = { ...attrs, color: colorOf(attrs), size: attrs.size * display.nodeSize };
            if (focus && view.hasNode(focus)) {
                if (node === focus) return { ...shown, highlighted: true, zIndex: 2 };
                const kind = neighbours.get(node);
                if (kind) return { ...shown, ringColor: KIND_COLORS[kind] ?? '#94a3b8', ringSize: KIND_RING, forceLabel: true, zIndex: 1 };
                return { ...shown, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            if (hitNodes.size) {
                return hitNodes.has(node) ? { ...shown, highlighted: true, forceLabel: true, zIndex: 2 }
                    : { ...shown, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            if (focusCommunity !== null && attrs.community !== focusCommunity) {
                return { ...shown, color: DIMMED, ringColor: 'transparent', label: '', zIndex: 0 };
            }
            return shown;
        });
        renderer.setSetting('edgeReducer', (edge, attrs) => {
            const shown = { ...attrs, size: attrs.size * display.linkThickness, hidden: display.linkThickness === 0 };
            if (focus && view.hasNode(focus)) {
                return view.hasExtremity(edge, focus)
                    ? { ...shown, color: KIND_COLORS[attrs.kind as LinkKind] ?? '#94a3b8', zIndex: 1 } : { ...shown, hidden: true };
            }
            if (hitNodes.size) {
                const [a, b] = view.extremities(edge);
                return hitNodes.has(a) && hitNodes.has(b) ? shown : { ...shown, hidden: true };
            }
            if (focusCommunity !== null) {
                const [a, b] = view.extremities(edge);
                const inside = view.getNodeAttribute(a, 'community') === focusCommunity
                    && view.getNodeAttribute(b, 'community') === focusCommunity;
                return inside ? shown : { ...shown, hidden: true };
            }
            return shown;
        });
    }, [view, full, laidOut, hovered, selected, focusCommunity, settings, colorOf, hitNodes]);

    useEffect(() => {
        if (!laidOut) return;
        laidOut.forEachNode((node, a) => {
            if (!a.image || thumbs.current.has(node)) return;
            const img = new Image();
            img.src = api.getNoteThumbUrl(node);
            thumbs.current.set(node, img);
        });
    }, [laidOut]);

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
        return full.filterNodes((_, attrs) => String(attrs.label).toLowerCase().includes(q)).slice(0, 5);
    }, [full, query]);

    useEffect(() => {
        const q = query.trim();
        if (q.length < SEARCH_MIN_CHARS) {
            setHits(null);
            return;
        }
        let cancelled = false;
        const timer = setTimeout(() => {
            setSearching(true);
            api.search(q, SEARCH_RESULTS, true)
                .then((data) => {
                    if (!cancelled) setHits({ query: q, results: data.results });
                })
                .catch((err) => console.error('Search failed:', err))
                .finally(() => {
                    if (!cancelled) setSearching(false);
                });
        }, SEARCH_DEBOUNCE_MS);
        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }, [query]);

    const clearSearch = () => {
        setQuery('');
        setHits(null);
        setResultsOpen(false);
    };

    const focusNode = (node: string) => {
        setSelected(node);
        setResultsOpen(false);
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
        postLayout({ type: 'animate' });
        fitOnEnd.current = true;
        fit();
    };

    const selectedAttrs = selected && full?.hasNode(selected) ? full.getNodeAttributes(selected) : null;

    // links by kind under the pane's checkboxes: the selected note's (filters aside, so hidden ones count), else all
    const kindCounts = useMemo(() => {
        const counts: Record<LinkKind, number> = { similar: 0, duplicate: 0, link: 0 };
        if (!full) return { counts, ofNote: false };
        const ofNote = !!selected && full.hasNode(selected);
        const count = (_edge: string, a: Record<string, unknown>) => {
            if ((a.kind as LinkKind) in counts) counts[a.kind as LinkKind] += 1;
        };
        if (ofNote) full.forEachEdge(selected!, count);
        else full.forEachEdge(count);
        return { counts, ofNote };
    }, [full, selected]);
    const selectedLinks = links && links.uuid === selected ? links.items : null;
    const selectedCommunity = selectedAttrs ? communities.find((c) => c.id === selectedAttrs.community) : undefined;
    const hoveredAttrs = hovered && hovered !== selected && full?.hasNode(hovered) ? full.getNodeAttributes(hovered) : null;

    /** "podcast · Lactate threshold · 2024-03-02 · 64 min" */
    const noteMeta = (a: Record<string, unknown>) => [
        TYPE_LABELS[a.sourceType as string] ?? a.sourceType,
        communities.find((c) => c.id === a.community)?.label,
        typeof a.date === 'string' ? a.date.slice(0, 10) : null,
        typeof a.duration === 'number' && a.duration > 0 ? `${Math.round(a.duration / 60)} min` : null,
    ].filter(Boolean).join(' · ');

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
                        onChange={(e) => {
                            setQuery(e.target.value);
                            setResultsOpen(true);
                        }}
                        onFocus={() => setResultsOpen(true)}
                        onKeyDown={(e) => {
                            if (e.key === 'Escape') clearSearch();
                        }}
                        placeholder="Search notes and what was said…"
                        aria-label="Search notes"
                    />
                    {query && (
                        <button className="graph-search-clear" onClick={clearSearch} aria-label="Clear search" title="Clear (Esc)">✕</button>
                    )}
                    {resultsOpen && (matches.length > 0 || hits || searching) && (
                        <div className="graph-search-results">
                            {matches.length > 0 && (
                                <>
                                    <div className="graph-search-heading">Titles</div>
                                    <ul>
                                        {matches.map((node) => (
                                            <li key={node}>
                                                <button onClick={() => focusNode(node)}>{full?.getNodeAttribute(node, 'label')}</button>
                                            </li>
                                        ))}
                                    </ul>
                                </>
                            )}
                            {query.trim().length >= SEARCH_MIN_CHARS && (
                                <div className="graph-search-heading">
                                    Passages{searching ? ' · searching…' : hits ? ` · ${hits.results.length} notes highlighted` : ''}
                                </div>
                            )}
                            {hits && hits.results.length === 0 && !searching && (
                                <div className="graph-search-empty">Nothing found.</div>
                            )}
                            {hits && hits.results.length > 0 && (
                                <ul>
                                    {hits.results.map((r) => (
                                        <li key={r.uuid} className="graph-search-hit">
                                            <button onClick={() => full?.hasNode(r.uuid) && focusNode(r.uuid)}
                                                    title={full?.hasNode(r.uuid) ? 'Show on the graph' : 'Not on the graph'}>
                                                <span className="graph-search-hit-title">{r.title}</span>
                                                <span className="graph-search-hit-snippet">
                                                    {snippetText(r.snippet)}
                                                </span>
                                            </button>
                                            <Link className="graph-search-open"
                                                  href={r.timestamp_seconds != null
                                                      ? `/notes/${r.uuid}?t=${Math.floor(r.timestamp_seconds)}` : `/notes/${r.uuid}`}>
                                                {r.timestamp_seconds != null ? `▶ ${formatTimestamp(r.timestamp_seconds)}` : 'Open'}
                                            </Link>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
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
            {!error && (!data || laidOut !== full) && <div className="graph-loading">Loading graph...</div>}

            <div ref={containerRef} className="graph-canvas" />

            {!selected && !hovered && settings.display.communityLabels && settings.groups.colorBy === 'community' && (
                <div className="graph-community-labels" aria-hidden="true">
                    {communities
                        .filter((c) => focusCommunity === null || c.id === focusCommunity)
                        .map((c) => (
                            <span key={c.id} className="graph-community-label"
                                  ref={(el) => {
                                      if (el) labelEls.current.set(c.id, el);
                                      else labelEls.current.delete(c.id);
                                  }}>
                                {c.label}
                            </span>
                        ))}
                </div>
            )}

            {hoveredAttrs ? (
                // the note under the pointer, without leaving the graph (click keeps it in the panel)
                <div className={`graph-brief${selectedAttrs ? ' with-panel' : ''}${paneOpen ? ' with-pane' : ''}`}>
                    {hoveredAttrs.image && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img className="graph-brief-thumb" src={api.getNoteThumbUrl(hovered!)} alt=""
                             onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                    )}
                    <div className="graph-brief-body">
                        <div className="graph-brief-meta" style={{ color: colorOf(hoveredAttrs) }}>{noteMeta(hoveredAttrs)}</div>
                        <div className="graph-brief-title">{hoveredAttrs.label}</div>
                        <p className="graph-brief-text">{hoveredAttrs.brief as string}</p>
                        <div className="graph-brief-chips">
                            {[...(hoveredAttrs.people as string[]).slice(0, 3), ...(hoveredAttrs.tags as string[]).slice(0, 5)]
                                .map((v, i) => <span key={`${i}-${v}`} className="graph-brief-chip">{v}</span>)}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="graph-hint">Click to see why notes link · double-click to open</div>
            )}

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
                    kindCounts={kindCounts}
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
                    {selectedAttrs.image && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img className="graph-panel-thumb" src={api.getNoteThumbUrl(selected!)} alt=""
                             onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                    )}
                    <p className="graph-panel-brief">{selectedAttrs.brief as string}</p>
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
