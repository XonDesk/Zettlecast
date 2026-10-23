'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import type { InsightCard, InsightSide, TodayResponse } from '@/lib/types';

// A few connections from the graph each day (insights.py picks them; the LLM says what links
// them). Each card: ← not related (the link leaves the graph), → good link, ↑ write a note.

const SWIPE_PX = 90;
const POLL_MS = 8000; // while the LLM prepares today's cards
const TYPE_LABELS: Record<string, string> = { audio: 'Podcast', image: 'Screenshot', reddit: 'Reddit save', web: 'Web page', pdf: 'PDF', markdown: 'Note' };
const RELATIONS: Record<string, string> = { agrees: 'Agree', contradicts: 'Contradict', extends: 'Builds on it', applies: 'In practice' };

type Action = 'rejected' | 'good' | 'note';

function clock(seconds: number): string {
    const s = Math.floor(seconds);
    const pad = (n: number) => String(n).padStart(2, '0');
    return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

function Side({ side }: { side: InsightSide }) {
    const href = `/notes/${side.uuid}${side.time ? `?t=${Math.floor(side.time)}` : ''}`;
    return (
        <article className="today-side">
            <div className="today-side-meta">
                {[TYPE_LABELS[side.source_type] ?? side.source_type, side.community, side.date].filter(Boolean).join(' · ')}
            </div>
            <Link href={href} className="today-side-title">{side.title}</Link>
            <div className="today-side-body">
                {side.image && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img className="today-thumb" src={api.getNoteThumbUrl(side.uuid)} alt=""
                         onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                )}
                <blockquote className={side.passage ? 'today-passage' : 'today-passage today-summary'}>
                    {side.time !== null && <Link href={href} className="today-time">{clock(side.time)}</Link>}
                    {side.text}
                </blockquote>
            </div>
        </article>
    );
}

export default function TodayPage() {
    const [today, setToday] = useState<TodayResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [writing, setWriting] = useState<{ title: string; text: string } | null>(null);
    const [leaving, setLeaving] = useState<Action | null>(null);
    const [drag, setDrag] = useState<number | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const start = useRef<{ x: number; id: number } | null>(null);
    const busy = useRef(false);

    const load = useCallback(() => {
        api.getToday()
            .then((t) => {
                setToday(t);
                setError(null);
            })
            .catch((e) => {
                console.error(e);
                setError('Could not load today’s cards');
            });
    }, []);

    useEffect(load, [load]);

    // the LLM's take arrives card by card: look again while it works
    const preparing = !!today && today.preparing > 0 && today.generating;
    useEffect(() => {
        if (!preparing) return;
        const timer = setInterval(load, POLL_MS);
        return () => clearInterval(timer);
    }, [preparing, load]);

    useEffect(() => {
        if (!message) return;
        const timer = setTimeout(() => setMessage(null), 4000);
        return () => clearTimeout(timer);
    }, [message]);

    const open = today?.cards.filter((c) => !c.verdict) ?? [];
    const card: InsightCard | undefined = open[0];
    const done = today ? today.cards.length - open.length : 0;

    const settle = (key: string, verdict: InsightCard['verdict'], note?: string) =>
        setToday((t) => t && { ...t, cards: t.cards.map((c) => (c.key === key ? { ...c, verdict, note } : c)) });

    const act = useCallback(async (action: Action) => {
        if (!card || busy.current) return;
        if (action === 'note') {
            setDrag(null);
            setWriting({
                title: card.insight?.note_title ?? '',
                text: card.insight ? `${card.insight.insight}\n\n${card.insight.question}` : '',
            });
            return;
        }
        busy.current = true;
        setLeaving(action);
        try {
            await api.decideInsight(card.key, action);
            await new Promise((r) => setTimeout(r, 180)); // the card slides away
            settle(card.key, action);
            if (action === 'rejected') setMessage('Unlinked: these two won’t be linked again');
        } catch (e) {
            console.error(e);
            setError('That didn’t save');
        } finally {
            setLeaving(null);
            setDrag(null);
            busy.current = false;
        }
    }, [card]);

    const saveNote = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!card || !writing || busy.current) return;
        busy.current = true;
        try {
            const { uuid } = await api.noteFromInsight(card.key, writing.title, writing.text);
            settle(card.key, 'noted', uuid);
            setWriting(null);
            setMessage('Note saved and linked to both');
        } catch (err) {
            console.error(err);
            setError('The note didn’t save');
        } finally {
            busy.current = false;
        }
    };

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (writing || e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
            if (e.key === 'ArrowLeft') act('rejected');
            else if (e.key === 'ArrowRight') act('good');
            else if (e.key === 'ArrowUp') act('note');
            else return;
            e.preventDefault();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [act, writing]);

    // swipe on a touch screen (a mouse selects text: it has the buttons and arrow keys)
    const onPointerDown = (e: React.PointerEvent) => {
        if (e.pointerType !== 'touch' || writing) return;
        start.current = { x: e.clientX, id: e.pointerId };
    };
    const onPointerMove = (e: React.PointerEvent) => {
        if (start.current?.id === e.pointerId) setDrag(e.clientX - start.current.x);
    };
    const onPointerUp = (e: React.PointerEvent) => {
        if (start.current?.id !== e.pointerId) return;
        start.current = null;
        if (drag !== null && drag > SWIPE_PX) act('good');
        else if (drag !== null && drag < -SWIPE_PX) act('rejected');
        else setDrag(null);
    };

    const transform = leaving === 'good' ? 'translateX(120vw) rotate(8deg)'
        : leaving === 'rejected' ? 'translateX(-120vw) rotate(-8deg)'
        : drag ? `translateX(${drag}px) rotate(${drag / 40}deg)` : undefined;

    return (
        <div className="today">
            <header className="page-header today-header">
                <h1 className="page-title">💡 Today</h1>
                {today && today.cards.length > 0 && (
                    <span className="text-muted">{Math.min(done + 1, today.cards.length)} of {today.cards.length}</span>
                )}
            </header>

            {error && <div className="today-error" role="alert">{error}</div>}
            {!today && !error && <p className="text-muted">Loading…</p>}

            {today && today.cards.length === 0 && (
                <div className="card today-done">
                    <p>No connections to show yet: they come from the graph&apos;s links. Add notes, then rebuild the graph.</p>
                </div>
            )}

            {today && today.cards.length > 0 && !card && (
                <div className="card today-done">
                    <h2>That&apos;s today&apos;s set</h2>
                    <p className="text-muted">
                        {today.cards.filter((c) => c.verdict === 'good').length} good ·{' '}
                        {today.cards.filter((c) => c.verdict === 'noted').length} written up ·{' '}
                        {today.cards.filter((c) => c.verdict === 'rejected').length} unlinked
                    </p>
                    <p>New connections tomorrow. <Link href="/graph">Explore the graph</Link></p>
                </div>
            )}

            {card && (
                <section key={card.key} className={`card today-card${drag !== null && !leaving ? ' dragging' : ''}`}
                         style={{ transform }}
                         onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
                         onPointerCancel={() => { start.current = null; setDrag(null); }}>
                    <div className="today-kind">
                        {card.kind === 'echo' ? 'Echo: something new meets something older' : 'Unexpected link'}
                        {card.insight && RELATIONS[card.insight.relation] && (
                            <span className={`today-relation today-${card.insight.relation}`}>{RELATIONS[card.insight.relation]}</span>
                        )}
                    </div>
                    {card.insight ? (
                        <div className="today-insight">
                            <p>{card.insight.insight}</p>
                            <p className="today-question">{card.insight.question}</p>
                        </div>
                    ) : (
                        <p className="today-insight today-preparing text-muted">
                            {today?.generating ? 'Working out what links these two…' : 'The insight for this card is still to come.'}
                        </p>
                    )}
                    <div className="today-sides">
                        <Side side={card.a} />
                        <Side side={card.b} />
                    </div>
                    {card.shared.length > 0 && (
                        <div className="today-shared">
                            {card.shared.map((s) => <span key={s} className="graph-brief-chip">{s}</span>)}
                        </div>
                    )}

                    {writing ? (
                        <form className="today-write" onSubmit={saveNote}>
                            <input className="input" value={writing.title} placeholder="Title: the idea, as a claim"
                                   onChange={(e) => setWriting({ ...writing, title: e.target.value })} autoFocus />
                            <textarea className="input" rows={6} value={writing.text}
                                      placeholder="What you make of it. Both notes are linked at the end."
                                      onChange={(e) => setWriting({ ...writing, text: e.target.value })} />
                            <div className="today-write-actions">
                                <button type="submit" className="btn btn-primary" disabled={!writing.text.trim()}>Save note</button>
                                <button type="button" className="btn btn-secondary" onClick={() => setWriting(null)}>Cancel</button>
                            </div>
                        </form>
                    ) : (
                        <footer className="today-actions">
                            <button className="btn btn-secondary today-reject" onClick={() => act('rejected')}>
                                <span className="today-key">←</span> Not related
                            </button>
                            <button className="btn btn-secondary today-note" onClick={() => act('note')}>
                                <span className="today-key">↑</span> Write a note
                            </button>
                            <button className="btn btn-secondary today-good" onClick={() => act('good')}>
                                Good link <span className="today-key">→</span>
                            </button>
                        </footer>
                    )}
                </section>
            )}

            {message && <div className="triage-toast" role="status">{message}</div>}
        </div>
    );
}
