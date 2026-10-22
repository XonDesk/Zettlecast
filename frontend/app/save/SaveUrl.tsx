'use client';

import { useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';

const CLOSE_AFTER_MS = 2500;

export default function SaveUrl({ url }: { url: string }) {
    const [message, setMessage] = useState(url ? 'Saving…' : 'No page to save.');
    const started = useRef(false);

    useEffect(() => {
        if (!url || started.current) return;
        started.current = true; // once, even when React runs effects twice in development
        api.ingest(url)
            .then((r) => {
                if (r.status === 'success') setMessage(`Saved: ${r.title ?? url}`);
                else if (r.status === 'duplicate') setMessage(`Already saved: ${r.title ?? url}`);
                else setMessage(`Not saved: ${r.error ?? r.status}`);
                if (r.status === 'success' || r.status === 'duplicate') {
                    setTimeout(() => window.close(), CLOSE_AFTER_MS); // a window the bookmarklet opened
                }
            })
            .catch((e) => setMessage(`Not saved: ${e instanceof Error ? e.message : String(e)}`));
    }, [url]);

    return (
        <>
            <p role="status">{message}</p>
            {url && <p className="text-muted" style={{ wordBreak: 'break-all', fontSize: '0.8rem' }}>{url}</p>}
        </>
    );
}
