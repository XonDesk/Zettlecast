'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
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
    audio_path?: string;
    queue_position?: number;
}

interface QueueStatus {
    by_status: {
        pending: number;
        processing: number;
        completed: number;
        review: number;
        failed: number;
        cancelled?: number;
    };
    total: number;
    estimated_remaining: string;
    items: PodcastItem[];
}

interface RunningStatus {
    is_running: boolean;
    current_episode: string | null;
    current_episode_id: string | null;
    current_stage: string | null;
    current_chunk: number;
    total_chunks: number;
    device: string | null;
    chunk_device: string | null;
    processed_count: number;
    error_count: number;
    started_at: string | null;
    episode_started_at: string | null;
}

const STAGES = ['chunking', 'transcribing', 'diarizing', 'aligning', 'enhancing', 'saving'];
const STAGE_LABELS: Record<string, string> = {
    chunking: 'Chunk',
    transcribing: 'Transcribe',
    diarizing: 'Diarize',
    aligning: 'Align',
    enhancing: 'Enhance',
    saving: 'Save',
};

function StagePipeline({ currentStage }: { currentStage: string | null }) {
    const currentIdx = currentStage ? STAGES.indexOf(currentStage) : -1;

    return (
        <div className="stage-pipeline">
            {STAGES.map((stage, idx) => {
                let stateClass = 'pending';
                if (idx < currentIdx) stateClass = 'completed';
                else if (idx === currentIdx) stateClass = 'active';

                return (
                    <div key={stage} className="stage-pipeline-item">
                        <div className={`stage-dot ${stateClass}`} />
                        {idx < STAGES.length - 1 && (
                            <div className={`stage-connector ${idx < currentIdx ? 'completed' : ''}`} />
                        )}
                        <div className={`stage-label ${stateClass}`}>
                            {STAGE_LABELS[stage]}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function DeviceBadge({ device, chunkDevice }: { device: string | null; chunkDevice: string | null }) {
    if (!device) return null;

    const isGpu = device === 'cuda';
    const isFallback = isGpu && chunkDevice === 'cpu';

    if (isFallback) {
        return <span className="device-badge device-cpu">CPU fallback</span>;
    }
    if (isGpu) {
        return <span className="device-badge device-gpu">GPU (CUDA)</span>;
    }
    return <span className="device-badge device-cpu">CPU</span>;
}

function ElapsedTimer({ startedAt }: { startedAt: string | null }) {
    const [elapsed, setElapsed] = useState('0:00');
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        if (!startedAt) {
            setElapsed('0:00');
            return;
        }

        const update = () => {
            const start = new Date(startedAt).getTime();
            const diff = Math.floor((Date.now() - start) / 1000);
            const mins = Math.floor(diff / 60);
            const secs = diff % 60;
            setElapsed(`${mins}:${secs.toString().padStart(2, '0')}`);
        };

        update();
        intervalRef.current = setInterval(update, 1000);
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [startedAt]);

    return <span className="elapsed-timer">{elapsed}</span>;
}

export default function PodcastsPage() {
    const [status, setStatus] = useState<QueueStatus | null>(null);
    const [runningStatus, setRunningStatus] = useState<RunningStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    // Import form state
    const [showImport, setShowImport] = useState(false);
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
    const [cancelling, setCancelling] = useState<string | null>(null);

    // Completed section collapsed
    const [showCompleted, setShowCompleted] = useState(false);

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
            setShowImport(false);
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

    const handleCancel = async (episodeId: string) => {
        setCancelling(episodeId);
        try {
            const result = await api.cancelPodcastEpisode(episodeId);
            if (result.status === 'not_found') {
                setMessage({ type: 'error', text: 'Episode not found in queue' });
            } else {
                setMessage({ type: 'success', text: result.message || 'Cancelling...' });
            }
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to cancel episode' });
            console.error(err);
        } finally {
            setCancelling(null);
        }
    };

    // Derived lists
    const pendingItems = (status?.items || [])
        .filter(i => i.status === 'pending')
        .sort((a, b) => (a.queue_position || 999) - (b.queue_position || 999));

    const completedItems = (status?.items || [])
        .filter(i => i.status === 'completed')
        .slice(0, 20);

    const failedItems = (status?.items || [])
        .filter(i => i.status === 'failed' || i.status === 'review');

    const cancelledCount = status?.by_status.cancelled || 0;

    // Progress percentage
    const progressPercent = runningStatus?.total_chunks
        ? Math.round(((runningStatus.current_chunk - 1) / runningStatus.total_chunks) * 100)
        : 0;

    if (loading) return <div className="loading">Loading podcast queue...</div>;

    if (error) {
        return (
            <div>
                <div className="page-header">
                    <h1 className="page-title">Podcast Manager</h1>
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
                <h1 className="page-title">Podcast Manager</h1>
                <p className="text-muted">
                    Transcription render queue with speaker diarization
                </p>
            </div>

            {message && (
                <div className={`card mb-4`} style={{
                    borderColor: message.type === 'success' ? '#22c55e' : '#ef4444',
                    background: message.type === 'success' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'
                }}>
                    <p style={{ color: message.type === 'success' ? '#22c55e' : '#ef4444', margin: 0 }}>
                        {message.text}
                    </p>
                </div>
            )}

            {/* Actions Toolbar */}
            <div className="actions-toolbar mb-4">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <select
                        className="select"
                        value={runLimit}
                        onChange={(e) => setRunLimit(Number(e.target.value))}
                        disabled={runningStatus?.is_running || starting}
                        style={{ width: '120px' }}
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
                    >
                        {runningStatus?.is_running ? 'Running...' : starting ? 'Starting...' : 'Start Processing'}
                    </button>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-sm" onClick={() => setShowImport(!showImport)}>
                        {showImport ? 'Hide Import' : 'Import RSS'}
                    </button>
                    <button className="btn btn-sm" onClick={handleSync} disabled={syncing}>
                        {syncing ? 'Syncing...' : 'Sync'}
                    </button>
                    <button
                        className="btn btn-sm"
                        onClick={handleRetry}
                        disabled={retrying || failedItems.length === 0}
                    >
                        {retrying ? 'Retrying...' : 'Retry Failed'}
                    </button>
                    <button className="btn btn-sm" onClick={handleResetStuck} disabled={resetting}>
                        {resetting ? 'Resetting...' : 'Reset Stuck'}
                    </button>
                </div>
            </div>

            {/* Import Form (expandable) */}
            {showImport && (
                <div className="card mb-4">
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
                                style={{ width: '90px' }}
                            >
                                <option value={1}>1 ep</option>
                                <option value={3}>3 eps</option>
                                <option value={5}>5 eps</option>
                                <option value={10}>10 eps</option>
                                <option value={25}>25 eps</option>
                            </select>
                        </div>
                        <button type="submit" className="btn btn-primary" disabled={importing || !feedUrl.trim()}>
                            {importing ? 'Importing...' : 'Add to Queue'}
                        </button>
                    </form>
                </div>
            )}

            {/* Now Processing */}
            {runningStatus?.is_running && runningStatus.current_episode && (
                <div className="render-queue-section">
                    <div className="section-title">Now Processing</div>
                    <div className="now-processing-card">
                        <div className="now-processing-header">
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div className="now-processing-title">
                                    {runningStatus.current_episode}
                                </div>
                            </div>
                            <div className="now-processing-controls">
                                <DeviceBadge
                                    device={runningStatus.device}
                                    chunkDevice={runningStatus.chunk_device}
                                />
                                <ElapsedTimer startedAt={runningStatus.episode_started_at} />
                                {runningStatus.current_episode_id && (
                                    <button
                                        className="btn btn-sm"
                                        style={{ color: '#ef4444', borderColor: '#ef4444' }}
                                        onClick={() => handleCancel(runningStatus.current_episode_id!)}
                                        disabled={cancelling === runningStatus.current_episode_id}
                                    >
                                        {cancelling === runningStatus.current_episode_id ? 'Cancelling...' : 'Cancel'}
                                    </button>
                                )}
                            </div>
                        </div>

                        {/* Progress Bar */}
                        {runningStatus.total_chunks > 0 && (
                            <div className="progress-bar-container">
                                <div
                                    className="progress-bar-fill"
                                    style={{ width: `${Math.max(progressPercent, 2)}%` }}
                                />
                                <span className="progress-bar-text">
                                    Chunk {runningStatus.current_chunk} of {runningStatus.total_chunks}
                                    {' '}({progressPercent}%)
                                </span>
                            </div>
                        )}

                        {/* Stage Pipeline */}
                        <StagePipeline currentStage={runningStatus.current_stage} />

                        {/* Run stats */}
                        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.8rem' }}>
                            <span style={{ color: '#22c55e' }}>
                                {runningStatus.processed_count} completed
                            </span>
                            {runningStatus.error_count > 0 && (
                                <span style={{ color: '#ef4444' }}>
                                    {runningStatus.error_count} errors
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Up Next */}
            {pendingItems.length > 0 && (
                <div className="render-queue-section">
                    <div className="section-title">Up Next ({pendingItems.length})</div>
                    <div className="up-next-list">
                        {pendingItems.map((item) => (
                            <div key={item.job_id} className="up-next-item">
                                <span className="queue-position">{item.queue_position || '-'}</span>
                                <div className="up-next-info">
                                    <div className="up-next-title">{item.episode_title}</div>
                                    <div className="up-next-meta">{item.podcast_name}</div>
                                </div>
                                <button
                                    className="btn-icon"
                                    onClick={() => handleCancel(item.job_id)}
                                    disabled={cancelling === item.job_id}
                                    title="Cancel"
                                >
                                    {cancelling === item.job_id ? '...' : '\u00D7'}
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Status Summary Bar */}
            <div className="status-summary-bar mb-4">
                <div className="status-summary-item">
                    <span className="badge badge-pending">{status?.by_status.pending || 0}</span>
                    <span>Pending</span>
                </div>
                <div className="status-summary-item">
                    <span className="badge badge-processing">{status?.by_status.processing || 0}</span>
                    <span>Processing</span>
                </div>
                <div className="status-summary-item">
                    <span className="badge badge-completed">{status?.by_status.completed || 0}</span>
                    <span>Completed</span>
                </div>
                <div className="status-summary-item">
                    <span className="badge badge-failed">{(status?.by_status.failed || 0) + (status?.by_status.review || 0)}</span>
                    <span>Failed</span>
                </div>
                {cancelledCount > 0 && (
                    <div className="status-summary-item">
                        <span className="badge badge-cancelled">{cancelledCount}</span>
                        <span>Cancelled</span>
                    </div>
                )}
            </div>

            {/* Failed Items */}
            {failedItems.length > 0 && (
                <div className="render-queue-section">
                    <div className="section-title" style={{ color: '#ef4444' }}>
                        Failed ({failedItems.length})
                    </div>
                    {failedItems.map((item) => (
                        <div key={item.job_id} className="card" style={{ marginBottom: '0.5rem', borderColor: '#ef4444' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div style={{ flex: 1 }}>
                                    <div className="card-title" style={{ marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                                        {item.episode_title}
                                    </div>
                                    <div className="card-meta">
                                        {item.podcast_name}
                                        {item.attempts > 1 && ` \u00B7 ${item.attempts} attempts`}
                                    </div>
                                    {item.error_message && (
                                        <p className="text-error" style={{ fontSize: '0.75rem', marginTop: '0.25rem', marginBottom: 0 }}>
                                            {item.error_message}
                                        </p>
                                    )}
                                </div>
                                <span className="badge badge-failed">{item.status}</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Completed Items */}
            {completedItems.length > 0 && (
                <div className="render-queue-section">
                    <div
                        className="section-title"
                        style={{ cursor: 'pointer', userSelect: 'none' }}
                        onClick={() => setShowCompleted(!showCompleted)}
                    >
                        Completed ({status?.by_status.completed || 0})
                        <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>
                            {showCompleted ? '\u25B2' : '\u25BC'}
                        </span>
                    </div>
                    {showCompleted && completedItems.map((item) => (
                        <div key={item.job_id} className="card" style={{ marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div className="card-title" style={{ marginBottom: '0.15rem', fontSize: '0.9rem' }}>
                                        {item.episode_title}
                                    </div>
                                    <div className="card-meta">{item.podcast_name} &middot; {formatDate(item.added_at)}</div>
                                </div>
                                <span className="badge badge-completed">done</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Empty state */}
            {(status?.total || 0) === 0 && (
                <div className="empty-state">
                    <p>No episodes in queue. Import from an RSS feed to get started!</p>
                </div>
            )}
        </div>
    );
}
