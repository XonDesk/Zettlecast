'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';
import MarkdownContent from '@/components/MarkdownContent';
import type { NoteDetail as NoteDetailType } from '@/lib/types';

function formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins >= 60) {
        const hrs = Math.floor(mins / 60);
        const remainMins = mins % 60;
        return `${hrs}h ${remainMins}m`;
    }
    return `${mins}m ${secs.toString().padStart(2, '0')}s`;
}

function getTypeEmoji(type: string): string {
    const emojis: Record<string, string> = {
        pdf: '\u{1F4C4}',
        web: '\u{1F310}',
        audio: '\u{1F399}\uFE0F',
        markdown: '\u{1F4DD}',
        rss: '\u{1F4F0}',
        image: '\u{1F5BC}\uFE0F',
    };
    return emojis[type] || '\u{1F4CB}';
}

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
            const data = await api.getNote(uuid);
            setNote(data);
        } catch (err) {
            console.error('Failed to accept link:', err);
        }
    };

    const handleRejectLink = async (targetUuid: string) => {
        try {
            await api.manageLink(uuid, targetUuid, 'reject');
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
                    &larr; Back to Notes
                </button>
                <h1 className="page-title">{getTypeEmoji(note.source_type)} {note.title}</h1>
                <div className="card-meta mb-4">
                    <span className={`badge badge-${note.status}`}>{note.status}</span>
                    {' \u00B7 '}
                    {note.source_type}
                    {' \u00B7 '}
                    {formatDate(note.created_at)}
                </div>
            </div>

            <div className="note-detail">
                <div className="note-content">
                    <MarkdownContent content={note.full_text} />
                </div>

                <div className="note-sidebar">
                    <div className="card">
                        <h3 className="card-title mb-2">📎 Files</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            <a
                                href={api.getNoteSourceUrl(uuid)}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="btn btn-secondary"
                                style={{ textDecoration: 'none', textAlign: 'center' }}
                            >
                                📁 View Source File
                            </a>
                            <a
                                href={api.getNoteMarkdownUrl(uuid)}
                                download
                                className="btn btn-secondary"
                                style={{ textDecoration: 'none', textAlign: 'center' }}
                            >
                                📄 Download Markdown
                            </a>
                        </div>
                        {note.source_path && (
                            <p className="text-muted mt-2" style={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>
                                {note.source_path.startsWith('http') ? (
                                    <a href={note.source_path} target="_blank" rel="noopener noreferrer" style={{ color: '#3b82f6' }}>
                                        {note.source_path}
                                    </a>
                                ) : (
                                    <span title={note.source_path}>{note.source_path}</span>
                                )}
                            </p>
                        )}
                    </div>

                    <div className="card">
                        <h3 className="card-title mb-2">Metadata</h3>
                        {note.metadata.author && (
                            <p className="text-muted mb-1">Author: {note.metadata.author}</p>
                        )}
                        {note.metadata.tags.length > 0 && (
                            <div className="mb-1">
                                <p className="text-muted mb-1">Tags:</p>
                                <div className="tag-list">
                                    {note.metadata.tags.map((tag, i) => (
                                        <span key={i} className="tag-badge">{tag}</span>
                                    ))}
                                </div>
                            </div>
                        )}
                        {note.metadata.word_count && (
                            <p className="text-muted mb-1">Words: {note.metadata.word_count.toLocaleString()}</p>
                        )}
                        {note.metadata.duration_seconds && (
                            <p className="text-muted mb-1">Duration: {formatDuration(note.metadata.duration_seconds)}</p>
                        )}
                        {note.metadata.page_count && (
                            <p className="text-muted mb-1">Pages: {note.metadata.page_count}</p>
                        )}
                        {note.metadata.language && (
                            <p className="text-muted">Language: {note.metadata.language}</p>
                        )}
                    </div>

                    <div className="card">
                        <h3 className="card-title mb-2">Suggested Links</h3>
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
                                                Accept
                                            </button>
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => handleRejectLink(s.uuid)}
                                            >
                                                Reject
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
                        Delete Note
                    </button>
                </div>
            </div>
        </div>
    );
}
