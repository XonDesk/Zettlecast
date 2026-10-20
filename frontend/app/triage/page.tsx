'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { api, type TriageDecision, type TriageItem, type TriageState } from '@/lib/api';

// Screenshot triage: one screenshot at a time; swipe (phone) or arrow keys
//   ← / swipe left   discard: nothing is added
//   → / swipe right  keep: the information goes into the knowledge base, not the image
//   ↑ / swipe up     keep with the image (it matters visually)
//   Backspace / Z    undo the last decision

const SWIPE_PX = 90; // how far a card must travel to count
const REFILL_AT = 5; // fetch more when this few are left
const LEAVE_MS = 180;

const ACTIONS: { decision: TriageDecision; key: string; label: string; hint: string }[] = [
    { decision: 'skip', key: '←', label: 'Discard', hint: 'Not added' },
    { decision: 'keep_image', key: '↑', label: 'Keep + image', hint: 'Information and the image' },
    { decision: 'keep', key: '→', label: 'Keep', hint: 'Information only' },
];

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

    const decide = useCallback((decision: TriageDecision) => {
        const item = items[0];
        if (!item || busy.current) return;
        busy.current = true;
        inFlight.current.add(item.path);
        setLeaving(decision);
        if (items.length - 1 < REFILL_AT && state && state.remaining > items.length) load();
        setTimeout(() => {
            setItems((list) => list.slice(1));
            setLeaving(null);
            setDrag(null);
            setZoomed(false);
            busy.current = false;
        }, LEAVE_MS);
        setHistory((h) => [...h.slice(-49), { item, decision }]);
        setState((s) => s && {
            ...s,
            remaining: Math.max(0, s.remaining - 1),
            decided: { ...s.decided, [decision]: (s.decided[decision] ?? 0) + 1 },
            queue: decision === 'skip' ? s.queue : { ...s.queue, pending: (s.queue.pending ?? 0) + 1 },
        });
        api.decideTriage(item.path, decision).catch((e) => {
            console.error(e);
            setError(`Couldn't save the decision on ${item.name}; it will come up again`);
            inFlight.current.delete(item.path);
        });
    }, [items, state, load]);

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
            else return;
            e.preventDefault();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [decide, undo]);

    useEffect(() => {
        if (!message) return;
        const timer = setTimeout(() => setMessage(null), 4000);
        return () => clearTimeout(timer);
    }, [message]);

    // --- Swipe ---

    const onPointerDown = (e: React.PointerEvent) => {
        if (zoomed || busy.current) return;
        start.current = { x: e.clientX, y: e.clientY, id: e.pointerId };
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    };
    const onPointerMove = (e: React.PointerEvent) => {
        if (!start.current || start.current.id !== e.pointerId) return;
        setDrag({ dx: e.clientX - start.current.x, dy: e.clientY - start.current.y });
    };
    const onPointerUp = (e: React.PointerEvent) => {
        if (!start.current || start.current.id !== e.pointerId) return;
        start.current = null;
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
                        className={`triage-card ${zoomed ? 'zoomed' : ''} ${leaving ? 'leaving' : ''} ${drag ? 'dragging' : ''}`}
                        style={{ transform }}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={onPointerUp}
                        onPointerCancel={() => { start.current = null; setDrag(null); }}
                        onDoubleClick={() => setZoomed((z) => !z)}
                    >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={api.triageImageUrl(current.path)} alt={current.name} draggable={false} />
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
                    {current.folder} · {current.name} · {new Date(current.modified).toLocaleDateString()}
                </div>
            )}

            <footer className="triage-actions">
                <button className="btn btn-secondary" onClick={undo} disabled={!history.length} title="Undo (Backspace)">
                    ↶
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
