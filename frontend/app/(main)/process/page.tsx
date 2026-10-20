'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { Alarm, SystemStatus } from '@/lib/types';
import './hmi.css';

const POLL_MS = 2000;
const COMM_FAIL_MS = 6000; // no fresh reading for this long: values on screen are stale
const ARM_MS = 4000; // a dangerous button stays armed (press again to confirm) this long

const PASS1 = [
    { id: 'decode', name: 'DECODE', note: '16 kHz mono' },
    { id: 'asr', name: 'ASR', note: 'Parakeet' },
    { id: 'diarize', name: 'DIARIZE', note: 'Sortformer' },
    { id: 'align', name: 'ALIGN', note: 'words → turns' },
];
const PASS2 = [
    { id: 'analyse', name: 'ANALYSE', note: 'LLM' },
    { id: 'suspects', name: 'SUSPECTS', note: 'LLM' },
    { id: 'verify', name: 'VERIFY', note: 'audio judge' },
    { id: 'cleanup', name: 'CLEANUP', note: 'LLM + judge' },
    { id: 'save', name: 'SAVE', note: 'note .md' },
    { id: 'index', name: 'INDEX', note: 'embed' },
    { id: 'link', name: 'LINK', note: 'graph edges' },
];
const LLM_STAGES = new Set(['analyse', 'suspects', 'cleanup']);
const PRIORITY_ORDER = { high: 0, medium: 1, low: 2 } as const;

function since(iso?: string | null, now = Date.now()): string {
    if (!iso) return '—';
    const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const pad = (n: number) => String(n).padStart(2, '0');
    return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

function gb(mib?: number | null): string {
    return mib === null || mib === undefined ? '—' : (mib / 1024).toFixed(1);
}

/** A button that must be pressed twice within ARM_MS (destructive or disruptive actions). */
function ConfirmButton({ label, confirmLabel, disabled, onConfirm, title }: {
    label: string; confirmLabel: string; disabled?: boolean; onConfirm: () => void; title?: string;
}) {
    const [armed, setArmed] = useState(false);
    useEffect(() => {
        if (!armed) return;
        const t = setTimeout(() => setArmed(false), ARM_MS);
        return () => clearTimeout(t);
    }, [armed]);
    return (
        <button
            className="hmi-btn"
            data-armed={armed}
            disabled={disabled}
            title={title}
            onClick={() => {
                if (armed) {
                    setArmed(false);
                    onConfirm();
                } else {
                    setArmed(true);
                }
            }}
        >
            {armed ? confirmLabel : label}
        </button>
    );
}

function Analog({ value, max, warn, alarm, unit, label }: {
    value: number | null | undefined; max: number; warn?: number; alarm?: number; unit: string; label: string;
}) {
    const v = value ?? 0;
    const pct = (x: number) => `${Math.min(100, Math.max(0, (x / max) * 100))}%`;
    const level = alarm !== undefined && v >= alarm ? 'high' : warn !== undefined && v >= warn ? 'medium' : 'normal';
    return (
        <div className="hmi-analog" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={v}>
            <div className="hmi-row">
                <span>{label}</span>
                <span>{value === null || value === undefined ? '—' : `${v.toFixed(unit === 'GB' ? 1 : 0)} ${unit}`}</span>
            </div>
            <div className="hmi-analog-track">
                {warn !== undefined && (
                    <div className="hmi-analog-zone warn" style={{ left: pct(warn), width: `calc(${pct(alarm ?? max)} - ${pct(warn)})` }} />
                )}
                {alarm !== undefined && <div className="hmi-analog-zone alarm" style={{ left: pct(alarm), right: 0 }} />}
                <div className="hmi-analog-fill" data-level={level} style={{ width: pct(v) }} />
            </div>
            <div className="hmi-analog-scale">
                <span>0</span>
                {warn !== undefined && <span>normal &lt; {warn.toFixed(unit === 'GB' ? 1 : 0)}</span>}
                <span>{max.toFixed(0)} {unit}</span>
            </div>
        </div>
    );
}

function Unit({ x, y, name, note, state, value, progress }: {
    x: number; y: number; name: string; note: string; state: 'idle' | 'running' | 'fault';
    value?: string; progress?: number | null;
}) {
    return (
        <g className={`unit ${state}`} transform={`translate(${x} ${y})`}>
            <rect className="body" width={110} height={54} />
            <text className="name" x={8} y={17}>{name}</text>
            <text x={8} y={32}>{note}</text>
            <text x={8} y={46}>{value ?? (state === 'running' ? 'RUNNING' : '')}</text>
            {progress !== null && progress !== undefined && (
                <>
                    <rect className="bar-bg" x={66} y={40} width={36} height={6} />
                    <rect className="bar" x={66} y={40} width={36 * Math.min(1, progress)} height={6} />
                </>
            )}
        </g>
    );
}

function Mimic({ s, now }: { s: SystemStatus; now: number }) {
    const run = s.run;
    const q = s.queues.podcast.by_status;
    const total = Math.max(1, s.queues.podcast.total);
    const pending = q.pending ?? 0;
    const pass1Active = run.alive && run.phase === 'transcribe';
    const pass2Active = run.alive && (run.phase === 'enhance' || run.phase === 'running');
    const graphActive = run.alive && run.phase === 'graph';
    const llmDown = s.alarms.some((a) => a.id === 'llm-down');
    const vramAlarm = s.alarms.find((a) => a.id === 'vram');
    const stageValue = (id: string) => {
        if (run.stage !== id) return undefined;
        const t = since(run.stage_started_at, now);
        return run.chunks ? `${run.chunk}/${run.chunks} · ${t}` : t;
    };
    const unitState = (id: string, active: boolean): 'idle' | 'running' | 'fault' => {
        if (active && run.stage === id) return llmDown && LLM_STAGES.has(id) ? 'fault' : 'running';
        return 'idle';
    };
    const server = s.llm.server;
    const tankH = 90;
    const level = (pending / total) * (tankH - 4);
    const phaseCount = run.phase_total ? ` · ${run.phase_done ?? 0}/${run.phase_total}` : '';

    return (
        <svg viewBox="0 0 1180 470" role="img" aria-label="Process diagram: sources, queue, pass 1 transcription, saved transcriptions, pass 2 analysis, knowledge store, graph">
            <defs>
                <marker id="hmi-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M0 0 L10 5 L0 10 z" fill="#5a5e66" />
                </marker>
            </defs>

            {/* Row 1: sources -> queue -> pass 1 -> saved transcriptions */}
            <g className="unit idle" transform="translate(20 58)">
                <rect className="body" width={110} height={54} />
                <text className="name" x={8} y={17}>SOURCES</text>
                <text x={8} y={32}>RSS feed import</text>
                <text x={8} y={46}>uploads · storage</text>
            </g>
            <path className="pipe" d="M130 85 H168" markerEnd="url(#hmi-arrow)" />

            <rect className="tank" x={170} y={40} width={70} height={tankH} rx={6} />
            <rect className="tank-level" x={172} y={40 + tankH - 2 - level} width={66} height={level} rx={4} />
            <text className="label" x={205} y={32} textAnchor="middle">QUEUE</text>
            <text className="value" x={205} y={150} textAnchor="middle">{pending}</text>
            <text className="label" x={205} y={165} textAnchor="middle">pending of {s.queues.podcast.total}</text>

            <path className={`pipe ${pass1Active ? 'flow' : ''}`} d="M240 85 H288" markerEnd="url(#hmi-arrow)" />
            <rect className="frame" x={280} y={28} width={530} height={116} />
            <text className="frame-label" x={290} y={22}>PASS 1 · TRANSCRIPTION{pass1Active ? phaseCount : ''}</text>
            {PASS1.map((u, i) => (
                <Unit key={u.id} x={295 + i * 128} y={58} name={u.name} note={u.note}
                      state={unitState(u.id, pass1Active)} value={pass1Active ? stageValue(u.id) : undefined}
                      progress={pass1Active && u.id === 'asr' && run.stage === 'asr' && run.chunks ? (run.chunk ?? 0) / run.chunks : null} />
            ))}
            {PASS1.slice(0, -1).map((u, i) => (
                <path key={u.id} className="pipe" d={`M${405 + i * 128} 85 H${421 + i * 128}`} markerEnd="url(#hmi-arrow)" />
            ))}
            <path className={`pipe ${pass1Active ? 'flow' : ''}`} d="M789 85 H838" markerEnd="url(#hmi-arrow)" />

            <rect className="tank" x={840} y={40} width={80} height={tankH} rx={6} />
            <text className="label" x={880} y={32} textAnchor="middle">SAVED ASR</text>
            <text className="value" x={880} y={82} textAnchor="middle">{s.queues.podcast.total - (q.cancelled ?? 0)}</text>
            <text className="label" x={880} y={98} textAnchor="middle">.asr.json</text>

            {/* LLM server (utility for pass 2) */}
            <g className={`unit ${server.state === 'ready' ? 'running' : llmDown ? 'fault' : 'idle'}`} transform="translate(975 40)">
                <rect className="body" width={185} height={90} />
                <text className="name" x={10} y={18}>LLM SERVER</text>
                <text x={10} y={36}>{s.llm.backend === 'llama-server' ? 'llama-server' : 'Ollama'} · {s.llm.model}</text>
                <text x={10} y={54}>{server.state.toUpperCase()}{server.vision ? ' · vision' : ''}</text>
                <text x={10} y={72}>GPU {gb(s.gpu?.memory_used_mib)} / {gb(s.gpu?.memory_total_mib)} GB</text>
            </g>
            {vramAlarm && <rect className="alarm-marker" x={1146} y={46} width={9} height={9} transform="rotate(45 1150 50)" />}
            <path className="pipe" d="M1067 130 V180 H905 V198" strokeDasharray="3 4" markerEnd="url(#hmi-arrow)" />
            <text className="label" x={985} y={176}>LLM service</text>

            {/* ASR results -> pass 2 */}
            <path className={`pipe ${pass2Active ? 'flow' : ''}`} d="M880 130 V172 H40 V238" markerEnd="url(#hmi-arrow)" />

            {/* Row 2: pass 2 -> store -> graph */}
            <rect className="frame" x={20} y={200} width={900} height={118} />
            <text className="frame-label" x={58} y={194}>
                PASS 2 · ANALYSIS & CORRECTIONS{pass2Active ? phaseCount : ''}{run.legacy ? ' · RUNNING (no stage data from this run)' : ''}
            </text>
            {PASS2.map((u, i) => (
                <Unit key={u.id} x={32 + i * 126} y={240} name={u.name} note={u.note}
                      state={unitState(u.id, pass2Active)} value={pass2Active ? stageValue(u.id) : undefined} />
            ))}
            {PASS2.slice(0, -1).map((u, i) => (
                <path key={u.id} className="pipe" d={`M${142 + i * 126} 267 H${156 + i * 126}`} markerEnd="url(#hmi-arrow)" />
            ))}
            <path className={`pipe ${pass2Active ? 'flow' : ''}`} d="M898 267 H948" markerEnd="url(#hmi-arrow)" />

            <g transform="translate(950 214)">
                <rect className="tank" width={100} height={104} rx={8} />
                <text className="label" x={50} y={-6} textAnchor="middle">KNOWLEDGE</text>
                <text className="value" x={50} y={26} textAnchor="middle">{s.database.notes ?? '—'}</text>
                <text className="label" x={50} y={40} textAnchor="middle">notes</text>
                <text className="label" x={50} y={62} textAnchor="middle">{s.database.chunks ?? '—'} chunks</text>
                <text className="label" x={50} y={78} textAnchor="middle">{s.database.edges ?? '—'} links</text>
            </g>
            <path className={`pipe ${graphActive ? 'flow' : ''}`} d="M1050 267 H1068" markerEnd="url(#hmi-arrow)" />
            <g className={`unit ${graphActive ? 'running' : 'idle'}`} transform="translate(1070 240)">
                <rect className="body" width={95} height={54} />
                <text className="name" x={8} y={17}>GRAPH</text>
                <text x={8} y={32}>{s.database.communities ?? 0} communities</text>
                <text x={8} y={46}>{graphActive ? 'REBUILDING' : 'Leiden'}</text>
            </g>

            {/* Row 3: image and web lines into the store */}
            <text className="frame-label" x={30} y={360}>IMAGES</text>
            <g className="unit idle" transform="translate(20 370)">
                <rect className="body" width={110} height={54} />
                <text className="name" x={8} y={17}>IMAGE QUEUE</text>
                <text x={8} y={32}>{s.queues.images.by_status?.pending ?? 0} pending</text>
                <text x={8} y={46}>{s.queues.images.by_status?.failed ?? 0} failed</text>
            </g>
            <path className="pipe" d="M130 397 H158" markerEnd="url(#hmi-arrow)" />
            <Unit x={160} y={370} name="VISION" note="LLM + OCR" state="idle" />
            <path className="pipe" d="M270 397 H300 V346 H985 V320" markerEnd="url(#hmi-arrow)" />

            <text className="frame-label" x={360} y={360}>WEB & DOCUMENTS</text>
            <Unit x={350} y={370} name="FETCH" note="web · PDF · md" state="idle" />
            <path className="pipe" d="M460 397 H488" markerEnd="url(#hmi-arrow)" />
            <Unit x={490} y={370} name="PARSE" note="Trafilatura · Marker" state="idle" />
            <path className="pipe" d="M600 397 H628" markerEnd="url(#hmi-arrow)" />
            <Unit x={630} y={370} name="ENRICH" note="LLM tags" state="idle" />
            <path className="pipe" d="M740 397 H1015 V320" markerEnd="url(#hmi-arrow)" />
        </svg>
    );
}

export default function ProcessPage() {
    const [status, setStatus] = useState<SystemStatus | null>(null);
    const [lastOk, setLastOk] = useState(0);
    const [now, setNow] = useState(() => Date.now());
    const [message, setMessage] = useState('');
    const [busy, setBusy] = useState(false);
    const [limit, setLimit] = useState(10);
    const [feedUrl, setFeedUrl] = useState('');
    const [feedLimit, setFeedLimit] = useState(10);
    const [acked, setAcked] = useState<Set<string>>(new Set());
    const timer = useRef<ReturnType<typeof setInterval> | null>(null);

    const poll = useCallback(async () => {
        try {
            setStatus(await api.getSystemStatus());
            setLastOk(Date.now());
        } catch {
            /* shown as COMM FAIL once readings go stale */
        }
        setNow(Date.now());
    }, []);

    useEffect(() => {
        poll();
        timer.current = setInterval(poll, POLL_MS);
        return () => {
            if (timer.current) clearInterval(timer.current);
        };
    }, [poll]);

    // acknowledged alarms survive a reload within the session
    useEffect(() => {
        try {
            const saved = sessionStorage.getItem('hmi-acked');
            if (saved) setAcked(new Set(JSON.parse(saved)));
        } catch {
            /* storage unavailable: acks last until reload */
        }
    }, []);
    const ack = (ids: string[]) => {
        const next = new Set([...acked, ...ids]);
        setAcked(next);
        try {
            sessionStorage.setItem('hmi-acked', JSON.stringify([...next]));
        } catch {
            /* ignore */
        }
    };

    const act = async (label: string, fn: () => Promise<{ status: string; message?: string }>) => {
        setBusy(true);
        try {
            const r = await fn();
            setMessage(`${label}: ${r.message ?? r.status}`);
        } catch (e) {
            setMessage(`${label}: failed (${e instanceof Error ? e.message : 'no response'})`);
        } finally {
            setBusy(false);
            poll();
        }
    };

    const commFail = !lastOk || now - lastOk > COMM_FAIL_MS;

    if (!status) {
        return (
            <div className="hmi">
                <div className="hmi-header">
                    <span className="hmi-title">PROCESS</span>
                    <span className="hmi-comm-fail">{commFail && lastOk === 0 ? 'CONNECTING…' : 'COMM FAIL'}</span>
                </div>
            </div>
        );
    }

    const run = status.run;
    const alarms = [...status.alarms].sort((a, b) => PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]);
    const fault = alarms.some((a) => a.id === 'run-died' || a.id === 'llm-down');
    const plantState = fault ? 'fault' : run.alive ? (run.stop_requested ? 'stopping' : 'running') : 'idle';
    const stateLabel = { fault: 'FAULT', stopping: 'STOPPING', running: 'RUNNING', idle: 'IDLE' }[plantState];
    const q = status.queues.podcast.by_status;
    const gpu = status.gpu;
    const vramMax = gpu?.memory_total_mib ? gpu.memory_total_mib / 1024 : 32;
    const counts = (p: Alarm['priority']) => alarms.filter((a) => a.priority === p).length;
    const phaseName = { transcribe: 'Pass 1 · transcription', enhance: 'Pass 2 · analysis', graph: 'Graph rebuild',
        starting: 'Starting', running: 'Running', idle: 'Idle' }[run.phase] ?? run.phase;
    const recent = run.recent ?? [];
    const task = status.task;

    return (
        <div className="hmi">
            <header className="hmi-header">
                <span className="hmi-title">PROCESS</span>
                <span className="hmi-state" data-state={plantState}>{stateLabel}</span>
                {run.alive && <span className="hmi-muted">{phaseName}{run.source ? ` · ${run.source.toUpperCase()} pid ${run.pid}` : ''}</span>}
                <div className="hmi-header-right">
                    <span className="hmi-alarm-counts" aria-label="Active alarms by priority">
                        {(['high', 'medium', 'low'] as const).map((p) => (
                            <span key={p} className="hmi-alarm-count" title={`${p} priority alarms`}>
                                <span className="hmi-pri" data-p={p} /> {counts(p)}
                            </span>
                        ))}
                    </span>
                    {commFail ? <span className="hmi-comm-fail">COMM FAIL · data {since(new Date(lastOk).toISOString(), now)} old</span>
                        : <span>{new Date(status.time).toLocaleTimeString()}</span>}
                </div>
            </header>

            <section className="hmi-mimic" aria-label="Process overview">
                <Mimic s={status} now={now} />
            </section>

            <div className="hmi-grid">
                <section className="hmi-panel" aria-label="Podcast run">
                    <h2>Podcast run</h2>
                    <div className="hmi-row"><span>State</span><span>{stateLabel}{run.alive ? ` · ${phaseName}` : ''}</span></div>
                    {run.alive && !!run.phase_total && (
                        <>
                            <div className="hmi-row"><span>Progress</span><span>{run.phase_done} / {run.phase_total} episodes</span></div>
                            <div className="hmi-progress"><div style={{ width: `${(100 * (run.phase_done ?? 0)) / run.phase_total}%` }} /></div>
                        </>
                    )}
                    <div className="hmi-row"><span>Episode</span><span title={run.episode_title ?? ''}>{run.episode_title ?? '—'}</span></div>
                    <div className="hmi-row"><span>Step</span><span>{run.stage ? `${run.stage.toUpperCase()} · ${since(run.stage_started_at, now)}` : run.legacy ? 'not reported by this run' : '—'}</span></div>
                    <div className="hmi-row"><span>Episode time</span><span>{since(run.episode_started_at, now)}</span></div>
                    <div className="hmi-row"><span>This run</span><span>{run.completed ?? 0} done · {run.failed ?? 0} failed</span></div>
                    <div className="hmi-field">
                        <label className="hmi-muted" htmlFor="hmi-limit">Episodes</label>
                        <input id="hmi-limit" className="hmi-input num" type="number" min={1} value={limit}
                               onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))} />
                        <button className="hmi-btn" disabled={busy || run.alive || !q.pending}
                                onClick={() => act('Start', () => api.runPodcasts(limit))}>START</button>
                        {run.stop_requested
                            ? <button className="hmi-btn" disabled={busy || !run.alive} onClick={() => act('Resume', api.resumeRun)}>CANCEL STOP</button>
                            : <button className="hmi-btn" disabled={busy || !run.alive || run.legacy}
                                      title={run.legacy ? 'This run predates stop requests: only abort can end it' : 'Finish the current episode, then stop'}
                                      onClick={() => act('Stop', () => api.stopRun('after_episode'))}>STOP AFTER EPISODE</button>}
                        <ConfirmButton label="ABORT" confirmLabel="CONFIRM ABORT" disabled={busy || !run.alive}
                                       title="End the run now; the current episode goes back to pending"
                                       onConfirm={() => act('Abort', () => api.stopRun('abort'))} />
                    </div>
                </section>

                <section className="hmi-panel" aria-label="Equipment">
                    <h2>Equipment</h2>
                    <div className="hmi-row"><span>GPU</span><span>{gpu?.name ?? 'not found'}</span></div>
                    <Analog label="GPU memory" unit="GB" value={gpu?.memory_used_mib ? gpu.memory_used_mib / 1024 : null} max={vramMax}
                            warn={vramMax * status.limits.vram_warn} alarm={vramMax * status.limits.vram_alarm} />
                    <Analog label="GPU load" unit="%" value={gpu?.utilization} max={100} />
                    <div className="hmi-row"><span>Temperature · power</span><span>{gpu?.temperature ?? '—'} °C · {gpu?.power_w?.toFixed(0) ?? '—'} W</span></div>
                    <div className="hmi-row"><span>LLM server</span><span>{status.llm.server.installed ? `${status.llm.server.state.toUpperCase()}${status.llm.server.pid ? ` · pid ${status.llm.server.pid}` : ''}` : 'not installed (Ollama)'}</span></div>
                    <div className="hmi-row"><span>Ollama</span><span>{status.llm.ollama.reachable ? (status.llm.ollama.loaded.join(', ') || 'up, nothing loaded') : 'not reachable'}</span></div>
                    <div className="hmi-row"><span>API models</span><span>{Object.entries(status.models).filter(([, on]) => on).map(([k]) => k.replace('_', ' ')).join(', ') || 'none loaded'}</span></div>
                    <div className="hmi-buttons">
                        <ConfirmButton label="FREE GPU" confirmLabel="CONFIRM FREE GPU" disabled={busy || run.alive}
                                       title={run.alive ? 'Not during a run: stop or abort it first' : 'Stop the LLM server and unload every model'}
                                       onConfirm={() => act('Free GPU', api.unloadModels)} />
                    </div>
                </section>

                <section className="hmi-panel" aria-label="Queue and maintenance">
                    <h2>Queue & maintenance</h2>
                    <div className="hmi-row"><span>Podcast queue</span><span>{q.pending ?? 0} pending · {q.processing ?? 0} active · {q.completed ?? 0} done · {q.failed ?? 0} failed</span></div>
                    <div className="hmi-buttons">
                        <button className="hmi-btn" disabled={busy || !q.failed} onClick={() => act('Retry failed', api.retryFailedPodcasts)}>RETRY FAILED</button>
                        <button className="hmi-btn" disabled={busy || run.alive || !q.processing} onClick={() => act('Reset stuck', api.resetStuckPodcasts)}>RESET STUCK</button>
                        <button className="hmi-btn" disabled={busy} onClick={() => act('Sync storage', api.syncPodcastQueue)}>SYNC STORAGE</button>
                    </div>
                    <div className="hmi-field">
                        <input className="hmi-input" style={{ flex: 1 }} placeholder="RSS feed URL" value={feedUrl}
                               onChange={(e) => setFeedUrl(e.target.value)} aria-label="RSS feed URL" />
                        <input className="hmi-input num" type="number" min={1} value={feedLimit} aria-label="Episodes to import"
                               onChange={(e) => setFeedLimit(Math.max(1, Number(e.target.value) || 1))} />
                        <button className="hmi-btn" disabled={busy || !feedUrl.trim()}
                                onClick={() => act('Import feed', () => api.importPodcastFeed(feedUrl.trim(), feedLimit))}>IMPORT</button>
                    </div>
                    <div className="hmi-row"><span>Knowledge</span><span>{status.database.notes ?? '—'} notes · {status.database.edges ?? '—'} links · {status.database.communities ?? 0} communities</span></div>
                    <div className="hmi-buttons">
                        <button className="hmi-btn" disabled={busy || run.alive} onClick={() => act('Rebuild graph', async () => {
                            const r = await api.rebuildGraph();
                            return { status: r.status, message: `${r.edges} links` };
                        })}>REBUILD GRAPH</button>
                        <ConfirmButton label="REINDEX" confirmLabel="CONFIRM REINDEX" disabled={busy || run.alive || task.running}
                                       title="Re-embed every note and rebuild the graph (minutes, uses the GPU)"
                                       onConfirm={() => act('Reindex', api.reindex)} />
                        <button className="hmi-btn" disabled={busy || !(status.queues.images.by_status?.pending)}
                                onClick={() => act('Run images', () => api.runImages(20))}>RUN IMAGES</button>
                    </div>
                    {task.name && (
                        <div className="hmi-row"><span>Task</span>
                            <span>{task.name.toUpperCase()} · {task.running ? `running ${since(task.started_at, now)}` : task.error ? `failed: ${task.error}` : 'finished'}</span>
                        </div>
                    )}
                    <div className="hmi-row"><span>Disk free</span><span>{status.disk.free_gb.toFixed(0)} of {status.disk.total_gb.toFixed(0)} GB</span></div>
                    <div className="hmi-message" role="status" aria-live="polite">{message}</div>
                </section>
            </div>

            <div className="hmi-grid">
                <section className="hmi-panel" aria-label="Alarms">
                    <h2>Alarms</h2>
                    {alarms.length === 0 ? <span className="hmi-muted">No active alarms.</span> : (
                        <>
                            <ul className="hmi-alarms">
                                {alarms.map((a) => (
                                    <li key={a.id} className="hmi-alarm" data-p={a.priority} data-acked={acked.has(a.id)}>
                                        <span className="hmi-pri" data-p={a.priority} aria-label={`${a.priority} priority`} />
                                        <span className="text">{a.text}</span>
                                        {!acked.has(a.id) && <button className="hmi-btn" onClick={() => ack([a.id])}>ACK</button>}
                                    </li>
                                ))}
                            </ul>
                            {alarms.some((a) => !acked.has(a.id)) && (
                                <div className="hmi-buttons">
                                    <button className="hmi-btn" onClick={() => ack(alarms.map((a) => a.id))}>ACK ALL</button>
                                </div>
                            )}
                        </>
                    )}
                    {status.queues.podcast.failed.length > 0 && (
                        <>
                            <h2>Failed episodes</h2>
                            {status.queues.podcast.failed.map((f) => (
                                <div key={f.id} className="hmi-row"><span title={f.title}>{f.title.slice(0, 50)}</span><span title={f.error}>{f.error}</span></div>
                            ))}
                        </>
                    )}
                </section>

                <section className="hmi-panel" aria-label="Recent episodes">
                    <h2>Recent episodes</h2>
                    {recent.length === 0 ? <span className="hmi-muted">None yet in this run.</span> : (
                        <table className="hmi-table">
                            <thead>
                                <tr><th>Episode</th><th className="num">Audio</th><th className="num">Took</th><th className="num">s / audio min</th><th>Result</th></tr>
                            </thead>
                            <tbody>
                                {recent.map((r) => (
                                    <tr key={`${r.finished_at}-${r.title}`}>
                                        <td className="title" title={r.title}>{r.title}</td>
                                        <td className="num">{r.audio_minutes.toFixed(0)} min</td>
                                        <td className="num">{r.seconds.toFixed(0)} s</td>
                                        <td className="num">{r.audio_minutes ? (r.seconds / r.audio_minutes).toFixed(2) : '—'}</td>
                                        <td>{r.status}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </section>
            </div>
        </div>
    );
}
