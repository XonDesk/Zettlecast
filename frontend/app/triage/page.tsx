'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { api, type TriageDecision, type TriageItem, type TriageRegion, type TriageState } from '@/lib/api';

// Screenshot triage: one screenshot at a time; swipe (phone) or arrow keys
//   ← / swipe left   discard: nothing is added
//   → / swipe right  keep: the information goes into the knowledge base, not the image
//   ↑ / swipe up     keep with the image (it matters visually)
//   Backspace / Z    undo the last decision
//   ✎ / M            circle what the screenshot is for, then keep it (Esc clears the circle)

const SWIPE_PX = 90; // how far a card must travel to count
const REFILL_AT = 5; // fetch more when this few are left
const LEAVE_MS = 180;

const ACTIONS: { decision: TriageDecision; key: string; label: string; hint: string }[] = [
    { decision: 'skip', key: '←', label: 'Discard', hint: 'Not added' },
    { decision: 'keep_image', key: '↑', label: 'Keep + image', hint: 'Information and the image' },
    { decision: 'keep', key: '→', label: 'Keep', hint: 'Information only' },
];

type Point = [number, number];

/** The circled region: bounding box and the stroke, as fractions of the image */
function regionOf(points: Point[]): TriageRegion | null {
    if (points.length < 5) return null;
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const [x, y] = [Math.min(...xs), Math.min(...ys)];
    const [w, h] = [Math.max(...xs) - x, Math.max(...ys) - y];
    return w > 0.02 && h > 0.01 ? { box: [x, y, w, h], points } : null;
}

function directionOf(dx: number, dy: number): TriageDecision | null {
    if (-dy > SWIPE_PX && Math.abs(dy) > Math.abs(dx)) return 'keep_image';
    if (dx > SWIPE_PX) return 'keep';
    if (dx < -SWIPE_PX) return 'skip';
    return null;
}

export default function TriagePage() {
    const [state, setState] = useState<TriageState | null>(null);
    const [items, setItems] = useState<TriageItem[]>([]);
    const [history, setHistory] = useState<{ item: TriageItem; decision: TriageDecision }[]>([]);
    const [drag, setDrag] = useState<{ dx: number; dy: number } | null>(null);
    const [leaving, setLeaving] = useState<TriageDecision | null>(null);
    const [zoomed, setZoomed] = useState(false);
    const [folder, setFolder] = useState('');
    const [showFolders, setShowFolders] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const start = useRef<{ x: number; y: number; id: number } | null>(null);
    // circling: drawing mode, the stroke (fractions of the image), the image's box in the card
    const [marking, setMarking] = useState(false);
    const [stroke, setStroke] = useState<Point[]>([]);
    const [frame, setFrame] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    // decisions sent but maybe not in the next fetch yet: never show those again
    const inFlight = useRef(new Set<string>());
    const busy = useRef(false);

    // bumped to fetch the next batch and fresh counts
    const [fetches, setFetches] = useState(0);
    const load = useCallback(() => setFetches((n) => n + 1), []);

    useEffect(() => {
        let cancelled = false;
        api.getTriage(30)
            .then((next) => {
                if (cancelled) return;
                setState(next);
                setItems((current) => {
                    const have = new Set(current.map((i) => i.path));
                    return [...current, ...next.items.filter((i) => !have.has(i.path) && !inFlight.current.has(i.path))];
                });
                setError(null);
            })
            .catch((e) => {
                console.error(e);
                if (!cancelled) setError('Could not reach the Zettlecast server');
            });
        return () => {
            cancelled = true;
        };
    }, [fetches]);

    // while kept screenshots are being read, keep the counts current
    useEffect(() => {
        if (!state?.processing) return;
        const timer = setInterval(load, 5000);
        return () => clearInterval(timer);
    }, [state?.processing, load]);

    const current = items[0];

    const startMarking = useCallback(() => {
        setStroke([]);
        setMarking(true);
        setZoomed(false);
    }, []);

    const decide = useCallback((decision: TriageDecision) => {
        const item = items[0];
        if (!item || busy.current) return;
        busy.current = true;
        inFlight.current.add(item.path);
        setLeaving(decision);
        if (items.length - 1 < REFILL_AT && state && state.remaining > items.length) load();
        const region = decision === 'skip' ? null : regionOf(stroke);
        setTimeout(() => {
            setItems((list) => list.slice(1));
            setLeaving(null);
            setDrag(null);
            setZoomed(false);
            setStroke([]);
            setMarking(false);
            busy.current = false;
        }, LEAVE_MS);
        setHistory((h) => [...h.slice(-49), { item, decision }]);
        setState((s) => s && {
            ...s,
            remaining: Math.max(0, s.remaining - 1),
            decided: { ...s.decided, [decision]: (s.decided[decision] ?? 0) + 1 },
            queue: decision === 'skip' ? s.queue : { ...s.queue, pending: (s.queue.pending ?? 0) + 1 },
        });
        api.decideTriage(item.path, decision, region).catch((e) => {
            console.error(e);
            setError(`Couldn't save the decision on ${item.name}; it will come up again`);
            inFlight.current.delete(item.path);
        });
    }, [items, state, load, stroke]);

    const undo = useCallback(async () => {
        const last = history[history.length - 1];
        if (!last || busy.current) return;
        setHistory((h) => h.slice(0, -1));
        try {
            const { undone } = await api.undoTriage(last.item.path);
            inFlight.current.delete(last.item.path);
            setItems((list) => [last.item, ...list.filter((i) => i.path !== last.item.path)]);
            setState((s) => s && {
                ...s,
                remaining: s.remaining + 1,
                decided: { ...s.decided, [last.decision]: Math.max(0, (s.decided[last.decision] ?? 1) - 1) },
                queue: last.decision === 'skip' || undone?.note_kept ? s.queue
                    : { ...s.queue, pending: Math.max(0, (s.queue.pending ?? 1) - 1) },
            });
            setMessage(undone?.note_kept
                ? 'Undone, but it was already read into a note: delete that note if you don\u2019t want it'
                : `Undone: ${last.item.name}`);
        } catch (e) {
            console.error(e);
            setError('Undo failed');
        }
    }, [history]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement) return;
            if (e.key === 'ArrowLeft') decide('skip');
            else if (e.key === 'ArrowRight') decide('keep');
            else if (e.key === 'ArrowUp') decide('keep_image');
            else if (e.key === 'Backspace' || e.key.toLowerCase() === 'z') undo();
            else if (e.key === ' ') setZoomed((z) => !z);
            else if (e.key.toLowerCase() === 'm') startMarking();
            else if (e.key === 'Escape') {
                setStroke([]);
                setMarking(false);
            }
            else return;
            e.preventDefault();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [decide, undo, startMarking]);

    useEffect(() => {
        if (!message) return;
        const timer = setTimeout(() => setMessage(null), 4000);
        return () => clearTimeout(timer);
    }, [message]);

    // where the image sits in its card, for the circle overlay (it's letterboxed to fit)
    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;
        const measure = () => setFrame({ left: img.offsetLeft, top: img.offsetTop, width: img.offsetWidth, height: img.offsetHeight });
        const observer = new ResizeObserver(measure);
        observer.observe(img);
        return () => observer.disconnect();
    }, [current?.path, zoomed]);

    // --- Swipe, or draw while circling ---

    const imagePoint = (e: React.PointerEvent): Point | null => {
        const rect = imgRef.current?.getBoundingClientRect();
        if (!rect || !rect.width || !rect.height) return null;
        const clamp = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 1e4) / 1e4;
        return [clamp((e.clientX - rect.left) / rect.width), clamp((e.clientY - rect.top) / rect.height)];
    };

    const onPointerDown = (e: React.PointerEvent) => {
        if (zoomed || busy.current) return;
        start.current = { x: e.clientX, y: e.clientY, id: e.pointerId };
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        if (marking) {
            const p = imagePoint(e);
            setStroke(p ? [p] : []);
        }
    };
    const onPointerMove = (e: React.PointerEvent) => {
        if (!start.current || start.current.id !== e.pointerId) return;
        if (marking) {
            const p = imagePoint(e);
            if (p) setStroke((s) => [...s, p]);
            return;
        }
        setDrag({ dx: e.clientX - start.current.x, dy: e.clientY - start.current.y });
    };
    const onPointerUp = (e: React.PointerEvent) => {
        if (!start.current || start.current.id !== e.pointerId) return;
        start.current = null;
        if (marking) {
            // one stroke, then back to swiping; too small a scribble counts as no circle
            setMarking(false);
            if (!regionOf(stroke)) setStroke([]);
            return;
        }
        const decision = drag ? directionOf(drag.dx, drag.dy) : null;
        if (decision) decide(decision);
        else setDrag(null);
    };

    const addFolder = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!folder.trim()) return;
        try {
            await api.addTriageFolder(folder.trim());
            setFolder('');
            setShowFolders(false);
            setItems([]);
            load();
        } catch (err) {
            console.error(err);
            setError('That folder wasn\u2019t found on this computer');
        }
    };

    const removeFolder = async (path: string) => {
        await api.removeTriageFolder(path);
        setItems([]);
        load();
    };

    const process = async () => {
        const result = await api.processTriage();
        setMessage(result.status === 'started' ? `Reading ${result.pending_count} kept screenshot(s)…`
            : result.status === 'refused' ? result.message ?? 'Busy'
            : result.status === 'no_pending' ? 'Nothing waiting to be read' : 'Already reading');
        load();
    };

    const region = regionOf(stroke);
    const hint = drag ? directionOf(drag.dx, drag.dy) : leaving;
    const transform = leaving
        ? leaving === 'keep' ? 'translate(120vw, 0) rotate(18deg)'
            : leaving === 'skip' ? 'translate(-120vw, 0) rotate(-18deg)'
            : 'translate(0, -120vh)'
        : drag ? `translate(${drag.dx}px, ${Math.min(drag.dy, 40)}px) rotate(${drag.dx / 20}deg)` : undefined;
    const waiting = state ? (state.queue.pending ?? 0) : 0;
    const noFolders = state && state.folders.length === 0;

    return (
        <div className="triage">
            <header className="triage-bar">
                <Link href="/images" className="btn btn-secondary">←</Link>
                <div className="triage-counts">
                    <strong>{state ? state.remaining : '…'}</strong> to go
                    {state && (
                        <span className="text-muted">
                            {' '}· kept {state.decided.keep + state.decided.keep_image} · discarded {state.decided.skip}
                        </span>
                    )}
                </div>
                <button className="btn btn-secondary" onClick={process} disabled={!waiting || state?.processing}
                        title="Read the kept screenshots into the knowledge base (uses the GPU)">
                    {state?.processing ? 'Reading…' : `Read ${waiting} kept`}
                </button>
                <button className="btn btn-secondary" aria-pressed={showFolders} onClick={() => setShowFolders(!showFolders)}>
                    Folders
                </button>
            </header>

            {(showFolders || noFolders) && (
                <section className="triage-folders">
                    {noFolders && <p>Add a folder of screenshots to start. The files themselves are never changed or deleted.</p>}
                    <ul>
                        {state?.folders.map((f) => (
                            <li key={f}>
                                <span title={f}>{f}</span>
                                <button className="btn btn-secondary" onClick={() => removeFolder(f)}>Remove</button>
                            </li>
                        ))}
                    </ul>
                    <form onSubmit={addFolder}>
                        <input className="input" value={folder} onChange={(e) => setFolder(e.target.value)}
                               placeholder="C:\Users\you\OneDrive\Pictures\Screenshots" spellCheck={false} />
                        <button className="btn btn-primary" type="submit">Add folder</button>
                    </form>
                </section>
            )}

            {error && <div className="triage-error" role="alert">{error}</div>}

            <main className="triage-stage">
                {items[1] && (
                    // the next one, loaded behind the current card
                    // eslint-disable-next-line @next/next/no-img-element
                    <img className="triage-card triage-next" src={api.triageImageUrl(items[1].path)} alt="" draggable={false} />
                )}
                {current ? (
                    <div
                        className={`triage-card ${zoomed ? 'zoomed' : ''} ${leaving ? 'leaving' : ''} ${drag ? 'dragging' : ''} ${marking ? 'marking' : ''}`}
                        style={{ transform }}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={onPointerUp}
                        onPointerCancel={() => { start.current = null; setDrag(null); }}
                        onDoubleClick={() => setZoomed((z) => !z)}
                    >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img ref={imgRef} src={api.triageImageUrl(current.path)} alt={current.name} draggable={false}
                             onLoad={(e) => {
                                 const img = e.currentTarget;
                                 setFrame({ left: img.offsetLeft, top: img.offsetTop, width: img.offsetWidth, height: img.offsetHeight });
                             }} />
                        {frame && stroke.length > 1 && (
                            <svg className="triage-mark" viewBox="0 0 1 1" preserveAspectRatio="none"
                                 style={{ left: frame.left, top: frame.top, width: frame.width, height: frame.height }}>
                                <polyline points={stroke.map((p) => p.join(',')).join(' ')}
                                          className={region && !marking ? 'closed' : ''} />
                            </svg>
                        )}
                    </div>
                ) : (
                    state && !noFolders && (
                        <div className="triage-done">
                            <p>All caught up.</p>
                            {waiting > 0 && <p className="text-muted">{waiting} kept screenshot(s) waiting to be read.</p>}
                        </div>
                    )
                )}
                {hint && <div className={`triage-stamp stamp-${hint}`}>{ACTIONS.find((a) => a.decision === hint)?.label}</div>}
            </main>

            {current && (
                <div className="triage-meta text-muted">
                    {marking ? 'Draw around the part that matters'
                        : region ? 'Circled: keep it and only that part is read · Esc clears'
                        : `${current.folder} · ${current.name} · ${new Date(current.modified).toLocaleDateString()}`}
                </div>
            )}

            <footer className="triage-actions">
                <button className="btn btn-secondary" onClick={undo} disabled={!history.length} title="Undo (Backspace)">
                    ↶
                </button>
                <button className={`btn btn-secondary triage-circle ${marking || region ? 'active' : ''}`}
                        onClick={() => (marking ? setMarking(false) : startMarking())} disabled={!current || zoomed}
                        title="Circle the part that matters (M)">
                    ✎
                </button>
                {ACTIONS.map((a) => (
                    <button key={a.decision} className={`triage-btn btn-${a.decision}`} onClick={() => decide(a.decision)}
                            disabled={!current} title={`${a.hint} (${a.key})`}>
                        <span className="triage-key">{a.key}</span>
                        {a.label}
                    </button>
                ))}
            </footer>

            {message && <div className="triage-toast" role="status">{message}</div>}
        </div>
    );
}
