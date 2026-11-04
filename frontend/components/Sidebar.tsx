'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { api } from '@/lib/api';

const navItems = [
    { href: '/today', label: '💡 Today', icon: '💡' },
    { href: '/notes', label: '📚 Notes', icon: '📚' },
    { href: '/podcasts', label: '🎙️ Podcasts', icon: '🎙️' },
    { href: '/images', label: '🖼️ Images', icon: '🖼️' },
    // { href: '/search', label: '🔍 Search', icon: '🔍' }, // hidden: the graph's search box does passage search now
    { href: '/graph', label: '📊 Graph', icon: '📊' },
    { href: '/process', label: '🏭 Process', icon: '🏭' },
    { href: '/settings', label: '⚙️ Settings', icon: '⚙️' },
];

// The sidebar is pinned beside the page on a wide screen (the « button slides it away, ☰ brings it
// back; remembered). On a narrow one (globals.css: max-width 900px) it's a drawer over the page:
// ☰ or a swipe from the left edge opens it, a link, the backdrop, a swipe left or Esc closes it.
const NARROW = '(max-width: 900px)';
const COLLAPSED_KEY = 'zettlecast.sidebar.collapsed';
const EDGE_PX = 24; // a swipe from this close to the left edge opens the drawer
const SWIPE_PX = 60;

export default function Sidebar() {
    const pathname = usePathname();
    const [url, setUrl] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const [open, setOpen] = useState(false); // the drawer, on a narrow screen
    const [collapsed, setCollapsed] = useState(false); // slid away, on a wide screen
    const swipe = useRef<{ x: number; y: number; opening: boolean } | null>(null);

    const narrow = () => window.matchMedia(NARROW).matches;

    useEffect(() => {
        try {
            setCollapsed(window.localStorage.getItem(COLLAPSED_KEY) === '1');
        } catch {
            // storage blocked: pinned
        }
    }, []);

    // the page's left margin follows (globals.css: body[data-sidebar])
    useEffect(() => {
        document.body.dataset.sidebar = collapsed ? 'collapsed' : 'pinned';
        return () => {
            delete document.body.dataset.sidebar;
        };
    }, [collapsed]);

    const setPinned = useCallback((pinned: boolean) => {
        setCollapsed(!pinned);
        try {
            window.localStorage.setItem(COLLAPSED_KEY, pinned ? '0' : '1');
        } catch {
            // storage blocked: this visit only
        }
    }, []);

    // a link was followed: the drawer has done its job
    useEffect(() => setOpen(false), [pathname]);

    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open]);

    // swipe right from the left edge opens the drawer, swipe left anywhere closes it
    useEffect(() => {
        const onStart = (e: TouchEvent) => {
            const t = e.touches[0];
            if (!narrow() || e.touches.length > 1) return;
            if (open || t.clientX <= EDGE_PX) swipe.current = { x: t.clientX, y: t.clientY, opening: !open };
        };
        const onEnd = (e: TouchEvent) => {
            const s = swipe.current;
            swipe.current = null;
            const t = e.changedTouches[0];
            if (!s || !t) return;
            const dx = t.clientX - s.x;
            if (Math.abs(dx) < Math.abs(t.clientY - s.y)) return; // scrolling, not swiping
            if (s.opening && dx > SWIPE_PX) setOpen(true);
            if (!s.opening && dx < -SWIPE_PX) setOpen(false);
        };
        window.addEventListener('touchstart', onStart, { passive: true });
        window.addEventListener('touchend', onEnd, { passive: true });
        return () => {
            window.removeEventListener('touchstart', onStart);
            window.removeEventListener('touchend', onEnd);
        };
    }, [open]);

    const menu = () => (narrow() ? setOpen((o) => !o) : setPinned(true));

    const handleQuickAdd = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!url.trim()) return;

        setIsAdding(true);
        setMessage(null);

        try {
            const result = await api.ingest(url);
            if (result.status === 'success') {
                setMessage({ type: 'success', text: `Added: ${result.title || 'Untitled'}` });
                setUrl('');
            } else if (result.status === 'queued') {
                // the desktop reads it: listed now, searchable once it has run
                setMessage({ type: 'success', text: 'Saved. The desktop adds it the next time it runs.' });
                setUrl('');
            } else if (result.status === 'duplicate') {
                setMessage({ type: 'error', text: result.error || 'Already exists' });
            } else {
                setMessage({ type: 'error', text: result.error || 'Failed to add' });
            }
        } catch {
            setMessage({ type: 'error', text: 'Failed to connect to API' });
        } finally {
            setIsAdding(false);
        }
    };

    const current = navItems.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`));

    return (
        <>
            <header className="top-bar">
                <button className="top-bar-menu" onClick={menu} aria-label="Menu" aria-expanded={open}>☰</button>
                <span className="top-bar-title">{current ? current.label : '🧠 Zettlecast'}</span>
            </header>

            <div className={clsx('sidebar-backdrop', open && 'open')} onClick={() => setOpen(false)} aria-hidden="true" />

            <aside className={clsx('sidebar', open && 'open')}>
                <div className="sidebar-header">
                    <div>
                        <h1 className="sidebar-title">🧠 Zettlecast</h1>
                        <p className="sidebar-subtitle">Digital Zettelkasten</p>
                    </div>
                    <button className="sidebar-hide" onClick={() => (narrow() ? setOpen(false) : setPinned(false))}
                            aria-label="Hide the sidebar" title="Hide the sidebar">«</button>
                </div>

                <nav className="sidebar-nav">
                    {navItems.map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={clsx('nav-item', item === current && 'active')}
                        >
                            {item.label}
                        </Link>
                    ))}
                </nav>

                <div className="sidebar-divider" />

                <div className="quick-add">
                    <h3 className="quick-add-title">Quick Add</h3>
                    <form onSubmit={handleQuickAdd}>
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://example.com/article"
                            className="quick-add-input"
                            disabled={isAdding}
                        />
                        <button
                            type="submit"
                            className="quick-add-button"
                            disabled={isAdding || !url.trim()}
                        >
                            {isAdding ? 'Adding...' : '➕ Add URL'}
                        </button>
                    </form>
                    {message && (
                        <p className={clsx('quick-add-message', message.type)}>
                            {message.text}
                        </p>
                    )}
                </div>
            </aside>
        </>
    );
}
