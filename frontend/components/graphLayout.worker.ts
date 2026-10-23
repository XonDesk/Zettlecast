/// <reference lib="webworker" />
// The graph's force layout, off the page's thread: the page only applies the positions it gets
// (on the page, each step's physics held up drawing, the mouse and zoom: ~20 fps on a slow CPU).

import { forceLink } from 'd3-force';
import {
    applyForces,
    createSimulation,
    type Forces,
    type LayoutLink,
    type LayoutMessage,
    type LayoutRequest,
    seedPositions,
    type SimLink,
    type SimNode,
} from './graphForces';

// the live layout stops a little earlier than d3's 0.001 (decay unchanged): the last steps
// barely move anything but each one makes the page re-upload every note and link
const LIVE_ALPHA_MIN = 0.005;
const FRAME_MS = 16; // one step per frame, as d3's own timer on the page did

const scope = self as unknown as DedicatedWorkerGlobalScope;
const sim = createSimulation().alphaMin(LIVE_ALPHA_MIN);
// layout state per note (position, velocity, pinned while dragged), kept for notes filtered
// out so they come back where they were
let all: SimNode[] = [];
const byId = new Map<string, SimNode>();
let forces: Forces | null = null;
let nodeSize = 1;
let timer: ReturnType<typeof setTimeout> | null = null;

function send(message: LayoutMessage, transfer: Transferable[] = []) {
    scope.postMessage(message, transfer);
}

function sendPositions() {
    const xy = new Float32Array(all.length * 2);
    all.forEach((n, i) => {
        xy[2 * i] = n.x!;
        xy[2 * i + 1] = n.y!;
    });
    send({ type: 'positions', xy }, [xy.buffer]);
}

function simLinks(links: LayoutLink[]): SimLink[] {
    return links.map((l) => {
        const community = byId.get(l.source)?.community ?? null;
        return { ...l, inside: community !== null && community === byId.get(l.target)?.community };
    });
}

function setLinks(target: typeof sim, links: LayoutLink[]) {
    (target.force('link') as ReturnType<typeof forceLink<SimNode, SimLink>>).links(simLinks(links));
}

function step() {
    const start = performance.now();
    sim.tick();
    sendPositions();
    if (sim.alpha() < sim.alphaMin()) {
        timer = null;
        send({ type: 'end' });
        return;
    }
    timer = setTimeout(step, Math.max(0, FRAME_MS - (performance.now() - start)));
}

function run(alpha?: number) {
    if (!sim.nodes().length) return;
    if (alpha !== undefined) sim.alpha(alpha);
    if (timer === null) timer = setTimeout(step, 0);
}

scope.onmessage = (event: MessageEvent<LayoutRequest>) => {
    const m = event.data;
    switch (m.type) {
        case 'init': {
            all = m.nodes;
            byId.clear();
            all.forEach((n) => byId.set(n.id, n));
            forces = m.forces;
            nodeSize = m.nodeSize;
            // every note and link, laid out before the page draws anything
            const prelayout = createSimulation().nodes(all);
            setLinks(prelayout, m.links);
            applyForces(prelayout, forces, nodeSize);
            prelayout.alpha(m.alpha).tick(m.ticks);
            sendPositions();
            send({ type: 'ready' });
            break;
        }
        case 'view':
            sim.nodes(m.ids.map((id) => byId.get(id)!).filter(Boolean));
            setLinks(sim, m.links);
            if (forces) applyForces(sim, forces, nodeSize);
            if (m.alpha !== null) run(m.alpha);
            break;
        case 'forces':
            forces = m.forces;
            nodeSize = m.nodeSize;
            applyForces(sim, forces, nodeSize);
            run(Math.max(sim.alpha(), 0.3));
            break;
        case 'drag': {
            const n = byId.get(m.id);
            if (!n) break;
            n.fx = m.x;
            n.fy = m.y;
            sim.alphaTarget(0.3);
            run();
            break;
        }
        case 'release': {
            const n = byId.get(m.id);
            if (n) {
                n.fx = null;
                n.fy = null;
            }
            sim.alphaTarget(0);
            break;
        }
        case 'animate':
            seedPositions(sim.nodes());
            run(1);
            break;
    }
};
