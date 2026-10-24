// The graph's force layout (d3-force), shared by the layout worker (which runs it) and the page
// (which seeds the start positions), plus the messages between the two.

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
import type { GraphSettings } from './GraphPane';

export type Forces = GraphSettings['forces'];

export interface SimNode extends SimulationNodeDatum {
    id: string;
    community: number | null;
    size: number;
}

export interface SimLink extends SimulationLinkDatum<SimNode> {
    value: number;
    inside: boolean; // both ends in the same community
}

/** A link as the page sends it: ids, not nodes */
export interface LayoutLink {
    source: string;
    target: string;
    value: number;
}

/** Page -> worker */
export type LayoutRequest =
    // every note (positions restored or seeded) and link; lay them out `ticks` steps, then rest
    | { type: 'init'; nodes: SimNode[]; links: LayoutLink[]; forces: Forces; nodeSize: number; alpha: number; ticks: number }
    // the notes and links the filters leave; alpha to restart the live layout at, or null to leave it resting
    | { type: 'view'; ids: string[]; links: LayoutLink[]; alpha: number | null }
    | { type: 'forces'; forces: Forces; nodeSize: number }
    // scatter the visible notes and lay them out again
    | { type: 'animate' };

/** Worker -> page */
export type LayoutMessage =
    // x, y of every note, in the order the init sent them
    | { type: 'positions'; xy: Float32Array }
    // the init's layout is done (its positions came just before)
    | { type: 'ready' }
    // the live layout came to rest
    | { type: 'end' };

// Small seeded PRNG so the starting layout (and so the final one) is the same on every load
export function mulberry32(seed: number) {
    return () => {
        seed |= 0;
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
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

export function createSimulation(): Simulation<SimNode, SimLink> {
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
export function applyForces(sim: Simulation<SimNode, SimLink>, forces: Forces, nodeSize: number): void {
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
export function seedPositions(nodes: SimNode[]): void {
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
