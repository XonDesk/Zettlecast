'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';
import type { Note } from '@/lib/types';

export default function NotesPage() {
    const [notes, setNotes] = useState<Note[]>([]);
    const [loading, setLoading] = useState(true);
    const [statusFilter, setStatusFilter] = useState<string>('');

    useEffect(() => {
        const fetchNotes = async () => {
            try {
                setLoading(true);
                const params: { status?: string; limit: number } = { limit: 100 };
                if (statusFilter) params.status = statusFilter;

                const data = await api.listNotes(params);
                setNotes(data.notes);
            } catch (err) {
                console.error('Failed to fetch notes:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchNotes();
    }, [statusFilter]);

    const getBadgeClass = (status: string) => {
        const classes: Record<string, string> = {
            inbox: 'badge badge-inbox',
            reviewed: 'badge badge-reviewed',
            archived: 'badge badge-archived',
        };
        return classes[status] || 'badge';
    };

    const getTypeEmoji = (type: string) => {
        const emojis: Record<string, string> = {
            pdf: '📄',
            web: '🌐',
            audio: '🎙️',
            markdown: '📝',
            rss: '📰',
        };
        return emojis[type] || '📋';
    };

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">📚 Notes</h1>
            </div>

            <div className="filters-row">
                <select
                    className="select"
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                >
                    <option value="">All Status</option>
                    <option value="inbox">Inbox</option>
                    <option value="reviewed">Reviewed</option>
                    <option value="archived">Archived</option>
                </select>
                <span className="text-muted">{notes.length} notes</span>
            </div>

            {loading ? (
                <div className="loading">Loading notes...</div>
            ) : notes.length === 0 ? (
                <div className="empty-state">
                    <p>No notes yet. Add some content using the sidebar!</p>
                </div>
            ) : (
                notes.map((note) => (
                    <Link key={note.uuid} href={`/notes/${note.uuid}`} style={{ textDecoration: 'none' }}>
                        <div className="card">
                            <div className="card-title">
                                {getTypeEmoji(note.source_type)} {note.title}
                            </div>
                            <div className="card-meta">
                                <span className={getBadgeClass(note.status)}>{note.status}</span>
                                {' · '}
                                {note.source_type}
                                {' · '}
                                {formatDate(note.created_at)}
                            </div>
                        </div>
                    </Link>
                ))
            )}
        </div>
    );
}
