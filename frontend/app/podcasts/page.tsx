'use client';

import { useEffect, useState, useCallback } from 'react';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';

interface PodcastItem {
    job_id: string;
    podcast_name: string;
    episode_title: string;
    status: string;
    added_at: string;
    error_message: string | null;
    attempts: number;
}

interface QueueStatus {
    by_status: {
        pending: number;
        processing: number;
        completed: number;
        review: number;
        failed: number;
    };
    total: number;
    estimated_remaining: string;
    items: PodcastItem[];
}

interface RunningStatus {
    is_running: boolean;
    current_episode: string | null;
    processed_count: number;
    error_count: number;
    started_at: string | null;
}

export default function PodcastsPage() {
    const [status, setStatus] = useState<QueueStatus | null>(null);
    const [runningStatus, setRunningStatus] = useState<RunningStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    // Import form state
    const [feedUrl, setFeedUrl] = useState('');
    const [episodeLimit, setEpisodeLimit] = useState(5);
    const [importing, setImporting] = useState(false);

    // Run settings
    const [runLimit, setRunLimit] = useState(5);
    const [starting, setStarting] = useState(false);

    // Action loading states
    const [syncing, setSyncing] = useState(false);
    const [retrying, setRetrying] = useState(false);
    const [resetting, setResetting] = useState(false);

    const fetchStatus = useCallback(async () => {
        try {
            const [queueData, runData] = await Promise.all([
                api.getPodcastStatus(),
                api.getRunningStatus(),
            ]);
            setStatus(queueData);
            setRunningStatus(runData);
            setError(null);
        } catch (err) {
            if (err instanceof Error && err.message.includes('503')) {
                setError('Podcast module not installed. Install with: pip install -e ".[podcast]"');
            } else {
                setError('Failed to load queue status. Is the API running?');
            }
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        // Faster refresh when processing is running
        const interval = setInterval(fetchStatus, runningStatus?.is_running ? 3000 : 10000);
        return () => clearInterval(interval);
    }, [fetchStatus, runningStatus?.is_running]);

    const handleImport = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!feedUrl.trim()) return;

        setImporting(true);
        setMessage(null);

        try {
            const result = await api.importPodcastFeed(feedUrl, episodeLimit);
            setMessage({ type: 'success', text: result.message });
            setFeedUrl('');
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to import feed' });
            console.error(err);
        } finally {
            setImporting(false);
        }
    };

    const handleStartProcessing = async () => {
        setStarting(true);
        setMessage(null);

        try {
            const result = await api.runPodcasts(runLimit);
            if (result.status === 'started') {
                setMessage({ type: 'success', text: result.message });
            } else if (result.status === 'already_running') {
                setMessage({ type: 'error', text: 'Processing is already running' });
            } else {
                setMessage({ type: 'error', text: result.message });
            }
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to start processing' });
            console.error(err);
        } finally {
            setStarting(false);
        }
    };

    const handleSync = async () => {
        setSyncing(true);
        setMessage(null);

        try {
            const result = await api.syncPodcastQueue();
            setMessage({ type: 'success', text: result.message });
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to sync queue' });
            console.error(err);
        } finally {
            setSyncing(false);
        }
    };

    const handleRetry = async () => {
        setRetrying(true);
        setMessage(null);

        try {
            const result = await api.retryFailedPodcasts();
            setMessage({ type: 'success', text: result.message });
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to retry failed items' });
            console.error(err);
        } finally {
            setRetrying(false);
        }
    };

    const handleResetStuck = async () => {
        setResetting(true);
        setMessage(null);

        try {
            const result = await api.resetStuckPodcasts();
            setMessage({ type: 'success', text: result.message });
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to reset stuck items' });
            console.error(err);
        } finally {
            setResetting(false);
        }
    };

    const getStatusBadge = (itemStatus: string) => {
        const badges: Record<string, string> = {
            pending: 'badge badge-pending',
            processing: 'badge badge-processing',
            completed: 'badge badge-completed',
            failed: 'badge badge-failed',
            review: 'badge badge-failed',
        };
        return badges[itemStatus] || 'badge';
    };

    if (loading) return <div className="loading">Loading podcast queue...</div>;

    if (error) {
        return (
            <div>
                <div className="page-header">
                    <h1 className="page-title">🎙️ Podcast Manager</h1>
                </div>
                <div className="card" style={{ borderColor: '#ef4444' }}>
                    <p className="text-error">{error}</p>
                </div>
            </div>
        );
    }

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">🎙️ Podcast Manager</h1>
                <p className="text-muted">
                    Transcribe audio with speaker diarization
                </p>
            </div>

            {message && (
                <div className={`card mb-4`} style={{
                    borderColor: message.type === 'success' ? '#22c55e' : '#ef4444',
                    background: message.type === 'success' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'
                }}>
                    <p style={{ color: message.type === 'success' ? '#22c55e' : '#ef4444' }}>
                        {message.text}
                    </p>
                </div>
            )}

            {/* Processing Status */}
            {runningStatus?.is_running && (
                <div className="card mb-4" style={{ borderColor: '#f97316', background: 'rgba(249, 115, 22, 0.1)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ fontSize: '1.5rem' }}>⚙️</div>
                        <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                                Transcription in Progress
                            </div>
                            <div className="text-muted" style={{ fontSize: '0.875rem' }}>
                                {runningStatus.current_episode || 'Initializing...'}
                            </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <div style={{ fontWeight: 600, color: '#22c55e' }}>
                                {runningStatus.processed_count} done
                            </div>
                            {runningStatus.error_count > 0 && (
                                <div style={{ fontSize: '0.75rem', color: '#ef4444' }}>
                                    {runningStatus.error_count} errors
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Queue Status Summary */}
            <div className="settings-grid mb-4">
                <div className="setting-item">
                    <div className="setting-label">⏳ Pending</div>
                    <div className="setting-value" style={{ fontSize: '1.5rem' }}>
                        {status?.by_status.pending || 0}
                    </div>
                </div>
                <div className="setting-item">
                    <div className="setting-label">⚙️ Processing</div>
                    <div className="setting-value" style={{ fontSize: '1.5rem', color: '#f97316' }}>
                        {status?.by_status.processing || 0}
                    </div>
                </div>
                <div className="setting-item">
                    <div className="setting-label">✅ Completed</div>
                    <div className="setting-value" style={{ fontSize: '1.5rem', color: '#22c55e' }}>
                        {status?.by_status.completed || 0}
                    </div>
                </div>
                <div className="setting-item">
                    <div className="setting-label">❌ Failed</div>
                    <div className="setting-value" style={{ fontSize: '1.5rem', color: '#ef4444' }}>
                        {(status?.by_status.review || 0) + (status?.by_status.failed || 0)}
                    </div>
                </div>
            </div>

            {status && status.estimated_remaining !== 'N/A' && status.by_status.pending > 0 && (
                <p className="text-muted mb-4">
                    ⏱️ Estimated time remaining: <strong>{status.estimated_remaining}</strong>
                </p>
            )}

            {/* Start Processing */}
            <div className="card mb-4" style={{ borderColor: '#3b82f6' }}>
                <h3 className="card-title mb-2">▶️ Run Processing</h3>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                    <select
                        className="select"
                        value={runLimit}
                        onChange={(e) => setRunLimit(Number(e.target.value))}
                        disabled={runningStatus?.is_running || starting}
                    >
                        <option value={1}>1 episode</option>
                        <option value={3}>3 episodes</option>
                        <option value={5}>5 episodes</option>
                        <option value={10}>10 episodes</option>
                        <option value={25}>25 episodes</option>
                        <option value={100}>All pending</option>
                    </select>
                    <button
                        className="btn btn-primary"
                        onClick={handleStartProcessing}
                        disabled={runningStatus?.is_running || starting || (status?.by_status.pending || 0) === 0}
                        style={{ flex: 1, maxWidth: '200px' }}
                    >
                        {runningStatus?.is_running ? '⏳ Running...' : starting ? 'Starting...' : '🚀 Start Processing'}
                    </button>
                </div>
                {(status?.by_status.pending || 0) === 0 && !runningStatus?.is_running && (
                    <p className="text-muted mt-2" style={{ fontSize: '0.75rem' }}>
                        No pending episodes. Import from an RSS feed below.
                    </p>
                )}
            </div>

            {/* Import Form */}
            <div className="card mb-4">
                <h3 className="card-title mb-2">📥 Import from RSS Feed</h3>
                <form onSubmit={handleImport}>
                    <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        <input
                            type="url"
                            className="input"
                            style={{ flex: 1 }}
                            value={feedUrl}
                            onChange={(e) => setFeedUrl(e.target.value)}
                            placeholder="https://example.com/podcast/feed.xml"
                            required
                        />
                        <select
                            className="select"
                            value={episodeLimit}
                            onChange={(e) => setEpisodeLimit(Number(e.target.value))}
                            style={{ width: '100px' }}
                        >
                            <option value={1}>1 ep</option>
                            <option value={3}>3 eps</option>
                            <option value={5}>5 eps</option>
                            <option value={10}>10 eps</option>
                            <option value={25}>25 eps</option>
                        </select>
                    </div>
                    <button type="submit" className="btn btn-primary" disabled={importing || !feedUrl.trim()}>
                        {importing ? 'Importing...' : '➕ Add to Queue'}
                    </button>
                </form>
            </div>

            {/* Queue Actions */}
            <div className="card mb-4">
                <h3 className="card-title mb-2">🔧 Queue Actions</h3>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={handleSync} disabled={syncing}>
                        {syncing ? 'Syncing...' : '🔄 Sync with Storage'}
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={handleRetry}
                        disabled={retrying || ((status?.by_status.review || 0) === 0)}
                    >
                        {retrying ? 'Retrying...' : '🔁 Retry Failed'}
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={handleResetStuck}
                        disabled={resetting}
                    >
                        {resetting ? 'Resetting...' : '⚡ Reset Stuck'}
                    </button>
                </div>
            </div>

            {/* Queue Items */}
            <h3 className="card-title mb-4">📋 Recent Episodes</h3>

            {status?.items.length === 0 ? (
                <div className="empty-state">
                    <p>No episodes in queue. Import from an RSS feed above!</p>
                </div>
            ) : (
                <div>
                    {status?.items.map((item) => (
                        <div key={item.job_id} className="card" style={{ marginBottom: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div>
                                    <div className="card-title" style={{ marginBottom: '0.25rem' }}>
                                        {item.episode_title}
                                    </div>
                                    <div className="card-meta">
                                        {item.podcast_name} · {formatDate(item.added_at)}
                                        {item.attempts > 1 && ` · ${item.attempts} attempts`}
                                    </div>
                                </div>
                                <span className={getStatusBadge(item.status)}>{item.status}</span>
                            </div>
                            {item.error_message && (
                                <p className="text-error mt-2" style={{ fontSize: '0.75rem' }}>
                                    {item.error_message}
                                </p>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
