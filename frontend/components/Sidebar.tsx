'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import clsx from 'clsx';
import { api } from '@/lib/api';

const navItems = [
    { href: '/notes', label: '📚 Notes', icon: '📚' },
    { href: '/podcasts', label: '🎙️ Podcasts', icon: '🎙️' },
    { href: '/images', label: '🖼️ Images', icon: '🖼️' },
    { href: '/search', label: '🔍 Search', icon: '🔍' },
    { href: '/graph', label: '📊 Graph', icon: '📊' },
    { href: '/process', label: '🏭 Process', icon: '🏭' },
    { href: '/settings', label: '⚙️ Settings', icon: '⚙️' },
];

export default function Sidebar() {
    const pathname = usePathname();
    const [url, setUrl] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

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
            } else if (result.status === 'duplicate') {
                setMessage({ type: 'error', text: result.error || 'Already exists' });
            } else {
                setMessage({ type: 'error', text: result.error || 'Failed to add' });
            }
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to connect to API' });
        } finally {
            setIsAdding(false);
        }
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <h1 className="sidebar-title">🧠 Zettlecast</h1>
                <p className="sidebar-subtitle">Digital Zettelkasten</p>
            </div>

            <nav className="sidebar-nav">
                {navItems.map((item) => (
                    <Link
                        key={item.href}
                        href={item.href}
                        className={clsx('nav-item', pathname === item.href && 'active')}
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
    );
}
