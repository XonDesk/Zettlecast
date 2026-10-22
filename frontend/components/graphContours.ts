import { contourDensity } from 'd3-contour';

export interface ContourPoint {
    x: number;
    y: number;
    color: string;
}

/** One colour group's density outlines, sparsest first, as paths in layout units */
export interface ContourLayer {
    color: string;
    paths: Path2D[];
}

// d3-contour 4's density.contours(data): the grid is computed once, then outlined at any
// threshold (@types/d3-contour 3 doesn't declare it yet)
type DensityContours = ((threshold: number) => MultiPolygon) & { max: number };
interface MultiPolygon {
    coordinates: [number, number][][][];
}

const LEVELS = 6;
const MIN_POINTS = 2;

/**
 * Density contours per colour group, as in Apple's Embedding Atlas: a kernel density estimate of
 * each group's notes, outlined at even fractions of that group's peak (so a small group shows as
 * clearly as a large one). Computed in layout units, so the camera only has to transform them.
 */
export function densityContours(points: ContourPoint[], bandwidth: number): ContourLayer[] {
    if (points.length === 0) return [];
    const pad = bandwidth * 3;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const [x0, y0] = [Math.min(...xs) - pad, Math.min(...ys) - pad];
    const width = Math.ceil(Math.max(...xs) - x0 + pad);
    const height = Math.ceil(Math.max(...ys) - y0 + pad);
    // about 300 cells across whatever the spread (d3 rounds the cell size to a power of two)
    const cellSize = Math.max(1, Math.round(Math.max(width, height) / 300));

    const groups = new Map<string, ContourPoint[]>();
    points.forEach((p) => groups.set(p.color, [...(groups.get(p.color) ?? []), p]));

    const layers: ContourLayer[] = [];
    groups.forEach((members, color) => {
        if (members.length < MIN_POINTS) return;
        const density = contourDensity<ContourPoint>()
            .x((p) => p.x - x0)
            .y((p) => p.y - y0)
            .size([width, height])
            .cellSize(cellSize)
            .bandwidth(bandwidth);
        const contours = (density as unknown as { contours: (data: ContourPoint[]) => DensityContours }).contours(members);
        const paths: Path2D[] = [];
        for (let i = 1; i <= LEVELS; i++) {
            const shape = contours((contours.max * i) / (LEVELS + 1));
            const path = new Path2D();
            shape.coordinates.forEach((polygon) => polygon.forEach((ring) => {
                ring.forEach(([x, y], j) => (j ? path.lineTo(x + x0, y + y0) : path.moveTo(x + x0, y + y0)));
                path.closePath();
            }));
            paths.push(path);
        }
        layers.push({ color, paths });
    });
    return layers;
}

/** Draw the layers under the graph; `toViewport` maps layout units to CSS pixels */
export function drawContours(
    canvas: HTMLCanvasElement,
    layers: ContourLayer[],
    toViewport: (p: { x: number; y: number }) => { x: number; y: number },
    opacity: number,
): void {
    const dpr = window.devicePixelRatio || 1;
    const [w, h] = [canvas.clientWidth, canvas.clientHeight];
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (layers.length === 0) return;

    // layout -> viewport is affine (normalisation + camera), so three points define it
    const unit = 1000;
    const o = toViewport({ x: 0, y: 0 });
    const ex = toViewport({ x: unit, y: 0 });
    const ey = toViewport({ x: 0, y: unit });
    const [a, b, c, d] = [(ex.x - o.x) / unit, (ex.y - o.y) / unit, (ey.x - o.x) / unit, (ey.y - o.y) / unit];
    ctx.setTransform(a * dpr, b * dpr, c * dpr, d * dpr, o.x * dpr, o.y * dpr);
    const scale = Math.sqrt(Math.abs(a * d - b * c));

    // translucent fills stack up towards each group's core, then the outlines over all of them
    ctx.globalAlpha = 0.07 * opacity;
    layers.forEach((layer) => {
        ctx.fillStyle = layer.color;
        layer.paths.forEach((path) => ctx.fill(path, 'evenodd'));
    });
    ctx.lineWidth = 1 / scale;
    layers.forEach((layer) => {
        ctx.strokeStyle = layer.color;
        layer.paths.forEach((path, i) => {
            ctx.globalAlpha = (0.3 + (0.4 * i) / LEVELS) * opacity;
            ctx.stroke(path);
        });
    });
    ctx.globalAlpha = 1;
}
