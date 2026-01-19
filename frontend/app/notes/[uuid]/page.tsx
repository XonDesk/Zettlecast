'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';
import type { NoteDetail as NoteDetailType } from '@/lib/types';

export default function NoteDetailPage() {
    const params = useParams();
    const router = useRouter();
    const uuid = params.uuid as string;

    const [note, setNote] = useState<NoteDetailType | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchNote = async () => {
            try {
                setLoading(true);
                const data = await api.getNote(uuid);
                setNote(data);
            } catch (err) {
                setError('Failed to load note');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        if (uuid) fetchNote();
    }, [uuid]);

    const handleAcceptLink = async (targetUuid: string) => {
        try {
            await api.manageLink(uuid, targetUuid, 'accept');
            // Refresh to update suggestions
            const data = await api.getNote(uuid);
            setNote(data);
        } catch (err) {
            console.error('Failed to accept link:', err);
        }
    };

    const handleRejectLink = async (targetUuid: string) => {
        try {
            await api.manageLink(uuid, targetUuid, 'reject');
            // Refresh to update suggestions
            const data = await api.getNote(uuid);
            setNote(data);
        } catch (err) {
            console.error('Failed to reject link:', err);
        }
    };

    const handleDelete = async () => {
        if (!confirm('Are you sure you want to delete this note?')) return;

        try {
            await api.deleteNote(uuid);
            router.push('/notes');
        } catch (err) {
            console.error('Failed to delete note:', err);
        }
    };

    if (loading) return <div className="loading">Loading note...</div>;
    if (error) return <div className="empty-state">{error}</div>;
    if (!note) return <div className="empty-state">Note not found</div>;

    return (
        <div>
            <div className="page-header">
                <button className="btn btn-secondary mb-4" onClick={() => router.push('/notes')}>
                    ← Back to Notes
                </button>
                <h1 className="page-title">📝 {note.title}</h1>
                <div className="card-meta mb-4">
                    <span className={`badge badge-${note.status}`}>{note.status}</span>
                    {' · '}
                    {note.source_type}
                    {' · '}
                    {formatDate(note.created_at)}
                </div>
            </div>

            <div className="note-detail">
                <div className="note-content">
                    {note.full_text.length > 5000
                        ? note.full_text.substring(0, 5000) + '\n\n... [truncated]'
                        : note.full_text
                    }
                </div>

                <div className="note-sidebar">
                    <div className="card">
                        <h3 className="card-title mb-2">Metadata</h3>
                        {note.metadata.author && (
                            <p className="text-muted mb-1">Author: {note.metadata.author}</p>
                        )}
                        {note.metadata.tags.length > 0 && (
                            <p className="text-muted mb-1">Tags: {note.metadata.tags.join(', ')}</p>
                        )}
                        {note.metadata.word_count && (
                            <p className="text-muted mb-1">Words: {note.metadata.word_count}</p>
                        )}
                        {note.metadata.language && (
                            <p className="text-muted">Language: {note.metadata.language}</p>
                        )}
                    </div>

                    <div className="card">
                        <h3 className="card-title mb-2">💡 Suggested Links</h3>
                        {note.suggestions && note.suggestions.length > 0 ? (
                            <div className="suggestions-list">
                                {note.suggestions.map((s) => (
                                    <div key={s.uuid} className="suggestion-card">
                                        <div className="suggestion-title">{s.title}</div>
                                        <div className="suggestion-score">Score: {s.score.toFixed(3)}</div>
                                        <div className="suggestion-actions">
                                            <button
                                                className="btn btn-primary"
                                                onClick={() => handleAcceptLink(s.uuid)}
                                            >
                                                ✅ Accept
                                            </button>
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => handleRejectLink(s.uuid)}
                                            >
                                                ❌ Reject
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-muted">No suggestions yet</p>
                        )}
                    </div>

                    <button className="btn btn-danger" onClick={handleDelete}>
                        🗑️ Delete Note
                    </button>
                </div>
            </div>
        </div>
    );
}
