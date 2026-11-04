'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { Alarm, DesktopStatus, SystemStatus } from '@/lib/types';
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
const CYCLE_WINDOW = 10; // episodes in the rolling cycle-time average

function clock(seconds: number): string {
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const pad = (n: number) => String(n).padStart(2, '0');
    return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

function roughly(seconds: number): string {
    if (seconds < 90) return `${Math.round(seconds)} s`;
    const m = Math.round(seconds / 60);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
}

interface Timing {
    current: number | null; // seconds into the current episode
    last: number | null;
    average: number | null; // rolling, this phase
    samples: number;
    eta: number | null; // seconds until the run ends (phases still to come estimated from earlier runs)
    unestimated: string[]; // later phases with no estimate yet (left out of the ETA)
    estimatedFromHistory: boolean;
}

/** Cycle times and a rough ETA from the run state (falls back on older runs' published data). */
function timing(run: SystemStatus['run'], now: number): Timing {
    const none: Timing = { current: null, last: null, average: null, samples: 0, eta: null, unestimated: [], estimatedFromHistory: false };
    if (!run.alive) return none;
    const phase = run.phase as 'transcribe' | 'enhance' | 'graph';
    // a run started before cycle times were published: its finished pass-2 episodes are in `recent`
    const cycles = run.cycles ?? (phase === 'enhance'
        ? (run.recent ?? []).filter((r) => r.status === 'completed' && (!run.started_at || r.finished_at >= run.started_at))
            .map((r) => r.seconds).reverse()
        : []);
    const window = cycles.slice(-CYCLE_WINDOW);
    const history = run.averages ?? {};
    const rolling = window.length ? window.reduce((a, b) => a + b, 0) / window.length : null;
    const average = rolling ?? history[phase] ?? null;
    const current = run.episode_started_at ? (now - new Date(run.episode_started_at).getTime()) / 1000 : null;
    const out: Timing = { ...none, current, last: cycles.length ? cycles[cycles.length - 1] : null,
        average: rolling, samples: window.length, estimatedFromHistory: rolling === null && average !== null };

    if (phase === 'graph') {
        const took = run.phase_started_at ? (now - new Date(run.phase_started_at).getTime()) / 1000 : 0;
        out.eta = history.graph !== undefined ? Math.max(0, history.graph - took) : null;
        return out;
    }
    if (!run.phase_total || average === null) return out;
    const left = Math.max(0, run.phase_total - (run.phase_done ?? 0));
    const inFlight = run.episode_started_at && left > 0 ? 1 : 0;
    let eta = inFlight * Math.max(0, average - (current ?? 0)) + (left - inFlight) * average;
    const later: ('enhance' | 'graph')[] = phase === 'transcribe' ? ['enhance', 'graph'] : ['graph'];
    for (const p of later) {
        const per = history[p];
        if (per === undefined) out.unestimated.push(p === 'graph' ? 'graph rebuild' : 'pass 2');
        else eta += p === 'graph' ? per : per * run.phase_total;
    }
    out.eta = eta;
    return out;
}

function since(iso?: string | null, now = Date.now()): string {
    if (!iso) return '—';
    const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const pad = (n: number) => String(n).padStart(2, '0');
    return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

interface Pace {
    per: number | null; // seconds per item (parallel slots included: items finished per second, inverted)
    samples: number; // finishes the rolling pace is from (0: since the run started)
    left: number | null; // seconds until the last item is done
}

const PACE_WINDOW = 12; // finishes in the rolling pace

/**
 * Pace from when the last items finished; until there are a few, from the start of the run
 * (which includes starting the LLM server). The time since the last finish counts towards the next.
 */
function paceOf(doneTimes: string[] | undefined, startedAt: string | null | undefined, done: number, total: number,
                now: number, minSamples = 4): Pace {
    const times = (doneTimes ?? []).slice(-PACE_WINDOW).map((t) => new Date(t).getTime());
    let per: number | null = null;
    let samples = 0;
    if (times.length >= minSamples) {
        per = (times[times.length - 1] - times[0]) / 1000 / (times.length - 1);
        samples = times.length;
    } else if (done && startedAt) {
        per = (now - new Date(startedAt).getTime()) / 1000 / done;
    }
    if (per === null) return { per, samples, left: null };
    const sinceLast = times.length ? (now - times[times.length - 1]) / 1000 : 0;
    return { per, samples, left: Math.max(0, Math.max(0, total - done) * per - Math.min(sinceLast, per)) };
}

/** The image run: reading pace, then the one graph rebuild (as long as the last one took). */
function imageTiming(img: SystemStatus['images'], graphAverage: number | undefined, now: number) {
    const none = { pace: { per: null, samples: 0, left: null } as Pace, link: null as number | null, left: null as number | null };
    if (!img?.is_running) return none;
    const link = img.link_seconds ?? graphAverage ?? null;
    if (img.phase === 'linking') {
        const took = img.phase_started_at ? (now - new Date(img.phase_started_at).getTime()) / 1000 : 0;
        return { ...none, link, left: link === null ? null : Math.max(0, link - took) };
    }
    const pace = paceOf(img.done_times, img.started_at, img.processed_count + img.error_count, img.total, now);
    return { pace, link, left: pace.left === null ? null : pace.left + (link ?? 0) };
}

/** Pages and documents being added: each as long as the recent ones of its kind took. */
function webTiming(web: SystemStatus['web'], now: number): { left: number | null; average: number | null; samples: number } {
    const recent = (web?.recent ?? []).filter((r) => r.status === 'success');
    const avgOf = (rows: typeof recent) => rows.length ? rows.reduce((a, r) => a + r.seconds, 0) / rows.length : null;
    const inFlight = web?.in_flight ?? [];
    const average = avgOf(recent);
    if (!inFlight.length) return { left: null, average, samples: recent.length };
    let left: number | null = null;
    for (const item of inFlight) {
        const avg = avgOf(recent.filter((r) => r.kind === item.kind)) ?? average;
        if (avg === null) continue;
        const remaining = Math.max(0, avg - (now - new Date(item.started_at).getTime()) / 1000);
        left = Math.max(left ?? 0, remaining); // added side by side
    }
    return { left, average, samples: recent.length };
}

function clockAt(now: number, seconds: number | null): string {
    return seconds === null ? '' : new Date(now + seconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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

function Mimic({ s, now, eta }: { s: SystemStatus; now: number; eta: string }) {
    const run = s.run;
    const img = s.images;
    const reading = !!img?.is_running && img.phase === 'reading';
    const linking = !!img?.is_running && img.phase === 'linking';
    const imgQ = s.queues.images.by_status ?? {};
    const tri = s.triage;
    const kept = tri ? (tri.decided.keep ?? 0) + (tri.decided.keep_image ?? 0) : null;
    const imgTankH = 74;
    const imgEta = clockAt(now, imageTiming(img, run.averages?.graph, now).left);
    const web = s.web;
    const webT = webTiming(web, now);
    const rp = s.reddit?.progress;
    const redditOn = !!rp?.running;
    const redditEta = redditOn && rp!.total ? clockAt(now, paceOf(rp!.done_times, null, rp!.done, rp!.total, now, 2).left) : '';
    const adding = web?.in_flight ?? [];
    const webLabel = [
        adding.length ? `ADDING ${adding.length}${webT.left !== null ? ` · ETA ${clockAt(now, webT.left)}` : ''}` : '',
        redditOn ? `REDDIT ${rp!.phase === 'tag' ? 'TAGGING' : 'SYNC'} ${rp!.done}/${rp!.total || '?'}${redditEta ? ` · ETA ${redditEta}` : ''}` : '',
    ].filter(Boolean).join(' · ');
    const imgLevel = Math.min(1, (imgQ.pending ?? 0) / Math.max(1, s.queues.images.total)) * (imgTankH - 4);
    const q = s.queues.podcast.by_status;
    const total = Math.max(1, s.queues.podcast.total);
    // undefined from an API server started before the count existed: all pending shown as queued
    const transcribed = s.queues.podcast.transcribed;
    const toTranscribe = Math.max(0, (q.pending ?? 0) - (transcribed ?? 0));
    const pass1Active = run.alive && run.phase === 'transcribe';
    const pass2Active = run.alive && (run.phase === 'enhance' || run.phase === 'running');
    const graphActive = (run.alive && run.phase === 'graph') || linking;
    const llmDown = s.alarms.some((a) => a.id === 'llm-down');
    const vramAlarm = s.alarms.find((a) => a.id === 'vram');
    const stageValue = (id: string) => {
        if (run.stage !== id) {
            // finished steps of this episode, else the previous episode's time for the step
            const done = run.stage_times?.[id];
            if (done !== undefined) return `done ${clock(done)}`;
            const last = run.last_stages?.[id];
            return last !== undefined ? `last ${clock(last)}` : undefined;
        }
        const t = since(run.stage_started_at, now);
        return run.chunks ? `${run.chunk}/${run.chunks} · ${t}` : t;
    };
    const unitState = (id: string, active: boolean): 'idle' | 'running' | 'fault' => {
        if (active && run.stage === id) return llmDown && LLM_STAGES.has(id) ? 'fault' : 'running';
        return 'idle';
    };
    const server = s.llm.server;
    const tankH = 90;
    const level = (toTranscribe / total) * (tankH - 4);
    const savedLevel = ((transcribed ?? 0) / total) * (tankH - 4);
    const phaseCount = (run.phase_total ? ` · ${run.phase_done ?? 0}/${run.phase_total}` : '') + (eta ? ` · ETA ${eta}` : '');

    return (
        <svg viewBox="0 0 1180 545" role="img" aria-label="Process diagram: sources, queue, pass 1 transcription, saved transcriptions, pass 2 analysis, knowledge store, graph; screenshots, sorting, image queue, vision reading; web and documents">
            <defs>
                <marker id="hmi-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M0 0 L10 5 L0 10 z" fill="context-stroke" />
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

            <g data-group="queue">
            <rect className="tank" x={170} y={40} width={70} height={tankH} rx={6} />
            <rect className="tank-level" x={172} y={40 + tankH - 2 - level} width={66} height={level} rx={4} />
            <text className="label" x={205} y={32} textAnchor="middle">QUEUE</text>
            <text className="value" x={205} y={150} textAnchor="middle">{toTranscribe}</text>
            <text className="label" x={205} y={165} textAnchor="middle">to transcribe of {s.queues.podcast.total}</text>
            </g>

            <g data-group="pass1">
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
            </g>

            {/* transcribed episodes waiting for pass 2 (their .asr.json saved) */}
            <rect className="tank" data-group="pass1" x={840} y={40} width={80} height={tankH} rx={6} />
            <rect className="tank-level" x={842} y={40 + tankH - 2 - savedLevel} width={76} height={savedLevel} rx={4} />
            <text className="label" x={880} y={32} textAnchor="middle">SAVED ASR</text>
            <text className="value" x={880} y={82} textAnchor="middle">{transcribed ?? '—'}</text>
            <text className="label" x={880} y={98} textAnchor="middle">for pass 2</text>

            {/* LLM server (utility for pass 2) */}
            <g data-group="llm" className={`unit ${server.state === 'ready' ? 'running' : llmDown ? 'fault' : 'idle'}`} transform="translate(975 40)">
                <rect className="body" width={185} height={90} />
                <text className="name" x={10} y={18}>LLM SERVER</text>
                <text x={10} y={36}>{s.llm.backend === 'llama-server' ? 'llama-server' : 'Ollama'} · {s.llm.model}</text>
                <text x={10} y={54}>{server.state.toUpperCase()}{server.vision ? ' · vision' : ''}</text>
                <text x={10} y={72}>GPU {gb(s.gpu?.memory_used_mib)} / {gb(s.gpu?.memory_total_mib)} GB</text>
            </g>
            {vramAlarm && <rect className="alarm-marker" x={1146} y={46} width={9} height={9} transform="rotate(45 1150 50)" />}
            <g data-group="llm">
            <path className={`pipe service ${server.state === 'ready' ? 'live' : ''}`} d="M1067 130 V180 H905 V198" markerEnd="url(#hmi-arrow)" />
            <text className="label accent" x={985} y={176}>LLM service</text>
            </g>

            {/* ASR results -> pass 2 */}
            <g data-group="pass2">
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
            </g>

            <g data-group="store">
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
            </g>

            {/* Row 3: screenshots -> sorting -> image queue -> vision -> store (linked by one rebuild at the end) */}
            <g data-group="images">
            <text className="frame-label" x={30} y={348}>
                IMAGES{img?.is_running ? ` · ${linking ? 'LINKING' : `READING ${img.processed_count}/${img.total}`}${img.stop_requested ? ' · STOPPING' : ''}${imgEta ? ` · ETA ${imgEta}` : ''}` : ''}
            </text>
            <g className="unit idle" transform="translate(20 370)">
                <rect className="body" width={110} height={54} />
                <text className="name" x={8} y={17}>SCREENSHOTS</text>
                <text x={8} y={32}>{tri ? `${tri.remaining} to sort` : '—'}</text>
                <text x={8} y={46}>{tri ? `${tri.folders} folder${tri.folders === 1 ? '' : 's'}` : ''}</text>
            </g>
            <path className="pipe" d="M130 397 H158" markerEnd="url(#hmi-arrow)" />
            <g className="unit idle" transform="translate(160 370)">
                <rect className="body" width={110} height={54} />
                <text className="name" x={8} y={17}>SORT</text>
                <text x={8} y={32}>you · swipe / keys</text>
                <text x={8} y={46}>{tri ? `${kept} kept · ${tri.decided.skip ?? 0} del` : ''}</text>
            </g>
            <path className="pipe" d="M215 424 V446" markerEnd="url(#hmi-arrow)" />
            <text className="label" x={215} y={460} textAnchor="middle">
                Recycle Bin{tri?.to_recycle ? ` · ${tri.to_recycle} in 10 min` : ''}
            </text>
            <path className="pipe" d="M270 397 H298" markerEnd="url(#hmi-arrow)" />

            <rect className="tank" x={300} y={362} width={70} height={imgTankH} rx={6} />
            <rect className="tank-level" x={302} y={362 + imgTankH - 2 - imgLevel} width={66} height={imgLevel} rx={4} />
            <text className="value" x={335} y={398} textAnchor="middle">{imgQ.pending ?? 0}</text>
            <text className="label" x={335} y={412} textAnchor="middle">to read</text>
            <text className="label" x={335} y={452} textAnchor="middle">
                IMAGE QUEUE{imgQ.failed ? ` · ${imgQ.failed} failed` : ''}
            </text>

            <path className={`pipe ${reading ? 'flow' : ''}`} d="M370 397 H398" markerEnd="url(#hmi-arrow)" />
            <Unit x={400} y={370} name="READ" note="LLM vision · slots"
                  state={reading ? (s.alarms.some((a) => a.id === 'llm-down') ? 'fault' : 'running') : 'idle'}
                  value={reading ? `${img!.in_flight.length} at once` : undefined}
                  progress={img?.is_running && img.total ? img.processed_count / img.total : null} />
            <path className={`pipe ${reading ? 'flow' : ''}`} d="M510 397 H528" markerEnd="url(#hmi-arrow)" />
            <Unit x={530} y={370} name="STORE" note="note · embed"
                  state={reading ? 'running' : 'idle'}
                  value={img?.is_running ? `${img.processed_count} stored` : undefined} />
            <path className={`pipe ${reading ? 'flow' : ''}`} d="M640 397 H690 V340 H985 V320" markerEnd="url(#hmi-arrow)" />
            <text className="label accent" x={700} y={334}>linked by one graph rebuild at the end</text>
            </g>

            {/* Row 4: web and documents into the store */}
            <g data-group="io">
            <text className="frame-label" x={30} y={478}>WEB & DOCUMENTS{webLabel ? ` · ${webLabel}` : ''}</text>
            <Unit x={20} y={486} name="FETCH" note="web · PDF · md"
                  state={adding.length || (redditOn && rp!.phase !== 'tag') ? 'running' : 'idle'}
                  value={adding.length ? adding[0].title.slice(0, 16) : redditOn && rp!.phase !== 'tag' ? 'Reddit saves' : undefined} />
            <path className={`pipe ${adding.length || redditOn ? 'flow' : ''}`} d="M130 513 H158" markerEnd="url(#hmi-arrow)" />
            <Unit x={160} y={486} name="PARSE" note="Trafilatura · Marker" state={adding.length ? 'running' : 'idle'}
                  value={webT.average !== null && !adding.length ? `avg ${clock(webT.average)}` : undefined} />
            <path className={`pipe ${adding.length || redditOn ? 'flow' : ''}`} d="M270 513 H298" markerEnd="url(#hmi-arrow)" />
            <Unit x={300} y={486} name="ENRICH" note="LLM tags" state={adding.length || redditOn ? 'running' : 'idle'}
                  progress={redditOn && rp!.total ? rp!.done / rp!.total : null} />
            <path className={`pipe ${adding.length || redditOn ? 'flow' : ''}`} d="M410 513 H1015 V320" markerEnd="url(#hmi-arrow)" />
            </g>
        </svg>
    );
}

/** The worker PC (on the primary): on or off, jobs waiting for it, wake, pause, stay on */
function DesktopPanel({ d, now, busy, act }: {
    d: DesktopStatus; now: number; busy: boolean;
    act: (label: string, fn: () => Promise<{ status: string; message?: string }>) => Promise<void>;
}) {
    const label = { offline: 'OFF', waking: 'WAKING', idle: 'ON · IDLE', working: 'ON · WORKING' }[d.state];
    const waiting = [
        d.pending.podcasts && `${d.pending.podcasts} episode${d.pending.podcasts > 1 ? 's' : ''}`,
        d.pending.images && `${d.pending.images} image${d.pending.images > 1 ? 's' : ''}`,
        d.pending.ingest && `${d.pending.ingest} page${d.pending.ingest > 1 ? 's' : ''} or file${d.pending.ingest > 1 ? 's' : ''} to add`,
        d.pending.tags && `${d.pending.tags} note${d.pending.tags > 1 ? 's' : ''} to tag or index`,
    ].filter(Boolean).join(' · ');
    const wakeNote = !d.wake.configured ? 'not set up (DESKTOP_MAC_ADDRESS)'
        : d.wake.gave_up ? `gave up after ${d.wake.misses} tries`
        : d.wake.sent_at ? `sent ${since(d.wake.sent_at, now)} ago${d.wake.reason ? ` (${d.wake.reason})` : ''}`
        : `automatic${d.wake.window ? ` ${d.wake.window}` : ''}`;
    return (
        <section className="hmi-panel" data-accent="desktop" aria-label="Desktop (GPU worker)">
            <h2>Desktop</h2>
            <div className="hmi-row"><span>State</span><span>{label}{d.host ? ` · ${d.host}` : ''}{d.paused ? ' · PAUSED' : ''}</span></div>
            <div className="hmi-row"><span>Last seen</span><span>{d.online ? 'now' : d.last_seen_at ? `${since(d.last_seen_at, now)} ago` : 'never'}</span></div>
            <div className="hmi-row"><span>Waiting for it</span><span>{waiting || 'nothing'}</span></div>
            <div className="hmi-row"><span>Wake-on-LAN</span><span>{wakeNote}</span></div>
            <div className="hmi-row"><span>When idle</span>
                <span>{d.stay_on ? 'stays on' : d.idle_shutdown_minutes > 0 ? `shuts down after ${d.idle_shutdown_minutes} min (nobody signed in)` : 'stays on'}</span>
            </div>
            <div className="hmi-buttons">
                <button className="hmi-btn" disabled={busy || d.online || !d.wake.configured}
                        title={d.online ? 'The desktop is on' : 'Send Wake-on-LAN now'}
                        onClick={() => act('Wake', api.wakeDesktop)}>WAKE</button>
                {d.paused && <button className="hmi-btn" disabled={busy} title="Let the desktop take new jobs again"
                                     onClick={() => act('Resume', api.resumeRun)}>RESUME</button>}
                <button className="hmi-btn" disabled={busy}
                        title={d.stay_on ? 'Let it shut down again when idle' : 'Keep it on when idle (e.g. to work on it remotely)'}
                        onClick={() => act(d.stay_on ? 'Allow shutdown' : 'Stay on',
                            async () => { await api.setDesktopStayOn(!d.stay_on); return { status: d.stay_on ? 'may shut down when idle' : 'stays on' }; })}>
                    {d.stay_on ? 'ALLOW SHUTDOWN' : 'STAY ON'}
                </button>
            </div>
        </section>
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
    const img = status.images;
    const imgRunning = !!img?.is_running;
    const imgQ = status.queues.images.by_status ?? {};
    const tri = status.triage;
    const imgT = imageTiming(img, run.averages?.graph, now);
    const webT = webTiming(status.web, now);
    const rp = status.reddit?.progress;
    const redditPace = rp?.running && rp.total ? paceOf(rp.done_times, null, rp.done, rp.total, now, 2) : null;
    const lastWeb = status.web?.recent.length ? status.web.recent[status.web.recent.length - 1] : null;
    const stopping = (run.alive && run.stop_requested) || (imgRunning && img!.stop_requested);
    const plantState = fault ? 'fault' : run.alive || imgRunning ? (stopping ? 'stopping' : 'running') : 'idle';
    const stateLabel = { fault: 'FAULT', stopping: 'STOPPING', running: 'RUNNING', idle: 'IDLE' }[plantState];
    const q = status.queues.podcast.by_status;
    const gpu = status.gpu;
    const vramMax = gpu?.memory_total_mib ? gpu.memory_total_mib / 1024 : 32;
    const counts = (p: Alarm['priority']) => alarms.filter((a) => a.priority === p).length;
    const phaseName = { transcribe: 'Pass 1 · transcription', enhance: 'Pass 2 · analysis', graph: 'Graph rebuild',
        starting: 'Starting', running: 'Running', idle: 'Idle' }[run.phase] ?? run.phase;
    const recent = run.recent ?? [];
    const task = status.task;
    const t = timing(run, now);
    const desk = status.desktop;
    const etaAt = t.eta !== null ? new Date(now + t.eta * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

    return (
        <div className="hmi">
            <header className="hmi-header">
                <span className="hmi-title">PROCESS</span>
                <span className="hmi-state" data-state={plantState}>{stateLabel}</span>
                {run.alive && <span className="hmi-muted">{phaseName}{run.source ? ` · ${run.source.toUpperCase()} pid ${run.pid}` : ''}</span>}
                {run.alive && etaAt && <span className="hmi-muted" title="Rough: average cycle time × episodes left">ETA {etaAt}{t.unestimated.length ? '+' : ''}</span>}
                {imgRunning && <span className="hmi-muted">Images · {img!.phase === 'linking' ? 'linking' : `reading ${img!.processed_count}/${img!.total}`}</span>}
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
                <Mimic s={status} now={now} eta={etaAt ? `${etaAt}${t.unestimated.length ? '+' : ''}` : ''} />
            </section>

            <div className="hmi-grid">
                {desk && <DesktopPanel d={desk} now={now} busy={busy} act={act} />}
                <section className="hmi-panel" data-accent="run" aria-label="Podcast run">
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
                    <div className="hmi-row"><span>Cycle time</span>
                        <span>
                            {run.alive && t.current !== null ? `now ${clock(t.current)}` : '—'}
                            {t.last !== null && ` · last ${clock(t.last)}`}
                            {t.average !== null && ` · avg ${clock(t.average)} (${t.samples})`}
                        </span>
                    </div>
                    <div className="hmi-row"><span>ETA</span>
                        <span title="Average cycle time × episodes left; later passes from the previous run's averages">
                            {t.eta === null ? (run.alive ? 'after the first episode' : '—')
                                : `~${roughly(t.eta)} · done ≈ ${etaAt}${t.unestimated.length ? ` + ${t.unestimated.join(', ')}` : ''}`}
                            {t.estimatedFromHistory && ' · from last run'}
                        </span>
                    </div>
                    <div className="hmi-row"><span>This run</span><span>{run.completed ?? 0} done · {run.failed ?? 0} failed</span></div>
                    <div className="hmi-field">
                        <label className="hmi-muted" htmlFor="hmi-limit">Episodes</label>
                        <input id="hmi-limit" className="hmi-input num" type="number" min={1} value={limit}
                               onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))} />
                        <button className="hmi-btn" disabled={busy || run.alive || imgRunning || !q.pending}
                                title={imgRunning ? 'Images are being read: stop that first' : undefined}
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

                <section className="hmi-panel" data-accent="equipment" aria-label="Equipment">
                    <h2>Equipment</h2>
                    <div className="hmi-row"><span>GPU</span><span>{gpu?.name ?? (desk && !desk.online ? 'desktop off' : 'not found')}</span></div>
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

                <section className="hmi-panel" data-accent="queue" aria-label="Queue and maintenance">
                    <h2>Queue & maintenance</h2>
                    <div className="hmi-row"><span>Podcast queue</span><span>{q.pending ?? 0} pending ({status.queues.podcast.transcribed ?? '—'} transcribed) · {q.processing ?? 0} active · {q.completed ?? 0} done · {q.failed ?? 0} failed</span></div>
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
                    </div>
                    {status.reddit && (
                        <>
                            <div className="hmi-row"><span>Reddit saves</span>
                                <span>
                                    {!status.reddit.configured ? 'no feed (Settings)'
                                        : status.reddit.progress.running
                                            ? `${rp!.phase === 'tag' ? 'tagging' : 'syncing'} ${rp!.done}/${rp!.total || '?'}`
                                              + (redditPace?.left != null ? ` · ~${roughly(redditPace.left)} left (done ≈ ${clockAt(now, redditPace.left)})` : '')
                                              + (rp!.current ? ` · ${rp!.current}` : '')
                                            : `${status.reddit.imported} imported${status.reddit.partial ? ` (${status.reddit.partial} without comments)` : ''}`
                                              + (status.reddit.untagged ? ` · ${status.reddit.untagged} untagged` : '')
                                              + ` · last sync ${status.reddit.last_sync ? new Date(status.reddit.last_sync).toLocaleString() : 'never'}`}
                                </span>
                            </div>
                            <div className="hmi-buttons">
                                <button className="hmi-btn" disabled={busy || run.alive || task.running || !status.reddit.configured}
                                        title={run.alive ? 'Not during a run (the LLM tags each save)' : 'Import saves not imported yet'}
                                        onClick={() => act('Sync Reddit', () => api.syncReddit())}>SYNC REDDIT</button>
                                <button className="hmi-btn" disabled={busy || run.alive || task.running || !status.reddit.configured}
                                        title="Import every new save now from the saved feed: no LLM tags (TAG SAVES) and no comments (FETCH COMMENTS) yet"
                                        onClick={() => act('Quick sync', () => api.syncReddit(false, false, false))}>QUICK SYNC</button>
                                {!!status.reddit.untagged && (
                                    <button className="hmi-btn" disabled={busy || run.alive || task.running}
                                            title="Add LLM tags and summaries to the saves imported without them (no Reddit requests)"
                                            onClick={() => act('Tag saves', api.tagReddit)}>TAG SAVES</button>
                                )}
                                {status.reddit.progress.running && (
                                    <button className="hmi-btn" disabled={busy}
                                            title="Stop after the current save; what's imported stays"
                                            onClick={() => act('Stop Reddit', api.stopReddit)}>STOP SYNC</button>
                                )}
                                {status.reddit.partial > 0 && (
                                    <button className="hmi-btn" disabled={busy || run.alive || task.running}
                                            title="Add the top comments to saves imported without their thread (about one a minute: Reddit's limit without a login); tags are kept"
                                            onClick={() => act('Fetch comments', () => api.syncReddit(true, false))}>FETCH COMMENTS</button>
                                )}
                            </div>
                        </>
                    )}
                    <div className="hmi-row"><span>Web & docs</span>
                        <span title="Pages and files added through the web UI or bookmarklet; each estimated from the recent ones of its kind">
                            {status.web?.in_flight.length
                                ? `adding ${status.web.in_flight.map((i) => i.title).join(', ')}`
                                  + (webT.left !== null ? ` · ~${roughly(webT.left)} left` : ' · no estimate yet')
                                : lastWeb
                                    ? `last: ${lastWeb.title} took ${roughly(lastWeb.seconds)}`
                                      + (webT.average !== null ? ` · avg ${roughly(webT.average)} (${webT.samples})` : '')
                                    : 'nothing added since the server started'}
                        </span>
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
                <section className="hmi-panel" data-accent="images" aria-label="Images">
                    <h2>Images</h2>
                    <div className="hmi-row"><span>State</span>
                        <span>
                            {!imgRunning ? `IDLE${img?.finished_at ? ` · last run ended ${new Date(img.finished_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : ''}`
                                : img!.phase === 'linking' ? 'LINKING · one graph rebuild for the new notes'
                                    : `READING · ${img!.in_flight.length} at once${img!.stop_requested ? ' · STOPPING after these' : ''}`}
                        </span>
                    </div>
                    {imgRunning && !!img!.total && (
                        <>
                            <div className="hmi-row"><span>Progress</span><span>{img!.processed_count} / {img!.total} read{img!.error_count ? ` · ${img!.error_count} failed` : ''}</span></div>
                            <div className="hmi-progress"><div style={{ width: `${(100 * img!.processed_count) / img!.total}%` }} /></div>
                        </>
                    )}
                    {imgRunning && img!.phase !== 'linking' && (
                        <div className="hmi-row"><span>Pace</span>
                            <span title={imgT.pace.samples ? `From the last ${imgT.pace.samples} images to finish (read side by side, so per image finished)`
                                : 'Since the run started, LLM server start included'}>
                                {imgT.pace.per === null ? 'after the first image'
                                    : `${imgT.pace.per.toFixed(1)} s / image${imgT.pace.samples ? ` (last ${imgT.pace.samples})` : ' (so far)'}`}
                            </span>
                        </div>
                    )}
                    {imgRunning && (
                        <div className="hmi-row"><span>ETA</span>
                            <span title="Images left × pace, then one graph rebuild as long as the last one took">
                                {imgT.left === null ? 'after the first image'
                                    : `~${roughly(imgT.left)} · done ≈ ${clockAt(now, imgT.left)}`
                                      + (img!.phase === 'linking' ? ' · linking' : imgT.link !== null ? ` · incl. linking ~${roughly(imgT.link)}` : ' + linking')}
                            </span>
                        </div>
                    )}
                    {imgRunning && img!.in_flight.length > 0 && (
                        <div className="hmi-row"><span>Reading</span><span title={img!.in_flight.join('\n')}>{img!.in_flight.join(' · ')}</span></div>
                    )}
                    <div className="hmi-row"><span>Image queue</span><span>{imgQ.pending ?? 0} to read · {imgQ.processing ?? 0} active · {imgQ.completed ?? 0} done · {imgQ.failed ?? 0} failed</span></div>
                    <div className="hmi-row"><span>Screenshots</span>
                        <span>
                            {tri ? `${tri.remaining} to sort · ${(tri.decided.keep ?? 0) + (tri.decided.keep_image ?? 0)} kept · ${tri.decided.skip ?? 0} deleted`
                                + (tri.to_recycle ? ` (${tri.to_recycle} to the Recycle Bin soon)` : '') : '—'}
                        </span>
                    </div>
                    <div className="hmi-buttons">
                        <button className="hmi-btn" disabled={busy || imgRunning || run.alive || !imgQ.pending}
                                title={run.alive ? 'Not during a podcast run (it needs the GPU)' : 'Read every image waiting in the queue'}
                                onClick={() => act('Read images', api.processTriage)}>START</button>
                        <button className="hmi-btn" disabled={busy || !imgRunning || img!.stop_requested || img!.phase === 'linking'}
                                title="Finish the images being read, start no more, link what was stored"
                                onClick={() => act('Stop images', api.stopImages)}>STOP</button>
                        <button className="hmi-btn" disabled={busy || imgRunning || !imgQ.failed}
                                onClick={() => act('Retry failed images', api.retryFailedImages)}>RETRY FAILED</button>
                        <Link className="hmi-btn" href="/triage">SORT SCREENSHOTS</Link>
                    </div>
                </section>

                <section className="hmi-panel" data-accent="alarms" aria-label="Alarms">
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

                <section className="hmi-panel" data-accent="recent" aria-label="Recent episodes">
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
