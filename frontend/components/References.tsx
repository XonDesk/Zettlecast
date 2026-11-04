'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import type { Reference, ReferenceCandidate } from '@/lib/types';

type UrlRef = Extract<Reference, { kind: 'url' }>;
type NameRef = Extract<Reference, { kind: 'name' }>;

/** What a note links to and mentions; any of it fetched on request as a note linked to this one */
export default function References({ uuid, onFetched }: { uuid: string; onFetched: () => void }) {
    const [refs, setRefs] = useState<Reference[] | null>(null);
    const [fetched, setFetched] = useState<Record<string, string>>({}); // url -> note uuid
    const [busy, setBusy] = useState<string | null>(null);
    const [errors, setErrors] = useState<Record<string, string>>({});
    const [candidates, setCandidates] = useState<Record<string, ReferenceCandidate[]>>({}); // query -> pages
    const [showLow, setShowLow] = useState(false);

    useEffect(() => {
        setRefs(null);
        api.getReferences(uuid).then((r) => setRefs(r.references)).catch(() => setRefs([]));
    }, [uuid]);

    const fetchUrl = async (url: string, key = url) => {
        setBusy(key);
        setErrors((e) => ({ ...e, [key]: '' }));
        try {
            const res = await api.fetchReference(uuid, url);
            if (res.uuid) {
                setFetched((f) => ({ ...f, [key]: res.uuid! }));
                onFetched();
            } else {
                setErrors((e) => ({ ...e, [key]: res.error || 'Could not read the page' }));
            }
        } catch {
            setErrors((e) => ({ ...e, [key]: 'Could not read the page' }));
        } finally {
            setBusy(null);
        }
    };

    const find = async (query: string) => {
        setBusy(query);
        setErrors((e) => ({ ...e, [query]: '' }));
        try {
            const res = await api.lookupReference(query);
            setCandidates((c) => ({ ...c, [query]: res.candidates }));
            if (!res.candidates.length) setErrors((e) => ({ ...e, [query]: 'Nothing found on Wikipedia' }));
        } catch {
            setErrors((e) => ({ ...e, [query]: 'Wikipedia lookup failed' }));
        } finally {
            setBusy(null);
        }
    };

    if (refs === null) return null;
    const links = refs.filter((r): r is UrlRef => r.kind === 'url');
    const names = refs.filter((r): r is NameRef => r.kind === 'name');
    if (!links.length && !names.length) return null;
    const shown = links.filter((l) => !l.low_value || showLow);
    const lowCount = links.length - links.filter((l) => !l.low_value).length;

    const fetchButton = (url: string, existing: string | null, key = url) => {
        const target = fetched[key] || existing;
        if (target) return <Link href={`/notes/${target}`} className="btn btn-secondary">Open</Link>;
        return (
            <button className="btn btn-primary" disabled={busy !== null} onClick={() => fetchUrl(url, key)}>
                {busy === key ? 'Fetching…' : 'Fetch'}
            </button>
        );
    };

    return (
        <div className="card">
            <h3 className="card-title mb-2">References</h3>
            {links.length > 0 && (
                <>
                    <p className="text-muted mb-1">Links</p>
                    <div className="suggestions-list">
                        {shown.map((l) => (
                            <div key={l.url} className="suggestion-card" title={l.context}>
                                <a href={l.url} target="_blank" rel="noopener noreferrer" className="suggestion-title"
                                   style={{ display: 'block', wordBreak: 'break-word' }}>
                                    {l.label}
                                </a>
                                <div className="suggestion-score">{l.domain}</div>
                                {errors[l.url] && <div className="suggestion-reason">{errors[l.url]}</div>}
                                <div className="suggestion-actions">{fetchButton(l.url, l.existing_uuid)}</div>
                            </div>
                        ))}
                    </div>
                    {lowCount > 0 && (
                        <button className="btn btn-secondary mt-2" onClick={() => setShowLow(!showLow)}>
                            {showLow ? 'Hide' : 'Show'} {lowCount} video, social and short links
                        </button>
                    )}
                </>
            )}
            {names.length > 0 && (
                <>
                    <p className="text-muted mb-1 mt-2">Mentioned</p>
                    <div className="suggestions-list">
                        {names.map((n) => (
                            <div key={n.label} className="suggestion-card">
                                <div className="suggestion-title">{n.label}</div>
                                <div className="suggestion-score">{n.what}</div>
                                {errors[n.query] && <div className="suggestion-reason">{errors[n.query]}</div>}
                                {candidates[n.query] ? (
                                    candidates[n.query].map((c) => (
                                        <div key={c.url} className="mb-1">
                                            <div className="suggestion-reason">
                                                <a href={c.url} target="_blank" rel="noopener noreferrer">{c.title}</a>
                                                {c.description && ` · ${c.description}`}
                                            </div>
                                            {errors[c.url] && <div className="suggestion-reason">{errors[c.url]}</div>}
                                            <div className="suggestion-actions">{fetchButton(c.url, null)}</div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="suggestion-actions">
                                        <button className="btn btn-secondary" disabled={busy !== null}
                                                onClick={() => find(n.query)}>
                                            {busy === n.query ? 'Looking up…' : 'Find'}
                                        </button>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
