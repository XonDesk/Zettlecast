'use client';

import { useEffect, useState, useCallback } from 'react';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';

interface ImageItem {
    job_id: string;
    image_title: string;
    collection_name: string;
    status: string;
    added_at: string;
    error_message: string | null;
    attempts: number;
    megapixels: number | null;
    image_path: string;
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
    items: ImageItem[];
}

interface RunningStatus {
    is_running: boolean;
    current_image: string | null;
    processed_count: number;
    error_count: number;
    started_at: string | null;
}

interface ScannedImage {
    path: string;
    name: string;
    size_mb: number;
}

interface ScanResult {
    status: string;
    path: string;
    total_count: number;
    images: ScannedImage[];
    has_more: boolean;
}

export default function ImagesPage() {
    const [status, setStatus] = useState<QueueStatus | null>(null);
    const [runningStatus, setRunningStatus] = useState<RunningStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    // Add form state
    const [imagePath, setImagePath] = useState('');
    const [collectionName, setCollectionName] = useState('');
    const [recursive, setRecursive] = useState(true);
    const [extensions, setExtensions] = useState({
        png: true,
        jpg: true,
        jpeg: true,
        gif: true,
        webp: true,
        bmp: true,
    });
    
    // Scan state
    const [scanning, setScanning] = useState(false);
    const [scanResult, setScanResult] = useState<ScanResult | null>(null);
    const [adding, setAdding] = useState(false);

    // Run settings
    const [runLimit, setRunLimit] = useState(5);
    const [starting, setStarting] = useState(false);

    // Action loading states
    const [retrying, setRetrying] = useState(false);

    const fetchStatus = useCallback(async () => {
        try {
            const [queueData, runData] = await Promise.all([
                api.getImageStatus(),
                api.getImageRunningStatus(),
            ]);
            setStatus(queueData);
            setRunningStatus(runData);
            setError(null);
        } catch (err) {
            if (err instanceof Error && err.message.includes('503')) {
                setError('Image module not installed. Install dependencies first.');
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

    const getSelectedExtensions = () => {
        return Object.entries(extensions)
            .filter(([_, enabled]) => enabled)
            .map(([ext, _]) => ext);
    };

    const handleScan = async () => {
        if (!imagePath.trim()) return;

        setScanning(true);
        setMessage(null);
        setScanResult(null);

        try {
            const selectedExts = getSelectedExtensions();
            const result = await api.scanImages(imagePath, recursive, selectedExts);
            setScanResult(result);
            if (result.total_count === 0) {
                setMessage({ type: 'error', text: 'No images found in directory' });
            }
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to scan directory. Check the path.' });
            console.error(err);
        } finally {
            setScanning(false);
        }
    };

    const handleAdd = async () => {
        if (!imagePath.trim()) return;

        setAdding(true);
        setMessage(null);

        try {
            const selectedExts = getSelectedExtensions();
            const result = await api.addImages(imagePath, collectionName || undefined, recursive, selectedExts);
            setMessage({ type: 'success', text: result.message });
            setImagePath('');
            setCollectionName('');
            setScanResult(null);
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to add images' });
            console.error(err);
        } finally {
            setAdding(false);
        }
    };

    const handleStartProcessing = async () => {
        setStarting(true);
        setMessage(null);

        try {
            const result = await api.runImages(runLimit);
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

    const handleRetry = async () => {
        setRetrying(true);
        setMessage(null);

        try {
            const result = await api.retryFailedImages();
            setMessage({ type: 'success', text: result.message });
            await fetchStatus();
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to retry failed items' });
            console.error(err);
        } finally {
            setRetrying(false);
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

    if (loading) return <div className="loading">Loading image queue...</div>;

    if (error) {
        return (
            <div>
                <div className="page-header">
                    <h1 className="page-title">🖼️ Image Manager</h1>
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
                <h1 className="page-title">🖼️ Image Manager</h1>
                <p className="text-muted">
                    Extract descriptions, OCR, and concepts from images
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
                                Image Processing in Progress
                            </div>
                            <div className="text-muted" style={{ fontSize: '0.875rem' }}>
                                {runningStatus.current_image || 'Initializing...'}
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

            {status && status.estimated_remaining !== '0 minutes' && status.by_status.pending > 0 && (
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
                        <option value={1}>1 image</option>
                        <option value={3}>3 images</option>
                        <option value={5}>5 images</option>
                        <option value={10}>10 images</option>
                        <option value={25}>25 images</option>
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
                        No pending images. Add images using the form below.
                    </p>
                )}
            </div>

            {/* Add Images Form */}
            <div className="card mb-4">
                <h3 className="card-title mb-2">📥 Add Images</h3>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {/* Path Input */}
                    <div>
                        <label className="text-muted" style={{ fontSize: '0.875rem', marginBottom: '0.25rem', display: 'block' }}>
                            Directory Path
                        </label>
                        <input
                            type="text"
                            className="input"
                            value={imagePath}
                            onChange={(e) => setImagePath(e.target.value)}
                            placeholder="/path/to/images/"
                            style={{ width: '100%' }}
                        />
                    </div>

                    {/* Collection Name */}
                    <div>
                        <label className="text-muted" style={{ fontSize: '0.875rem', marginBottom: '0.25rem', display: 'block' }}>
                            Collection Name (optional)
                        </label>
                        <input
                            type="text"
                            className="input"
                            value={collectionName}
                            onChange={(e) => setCollectionName(e.target.value)}
                            placeholder="e.g., Screenshots, Diagrams"
                            style={{ width: '100%' }}
                        />
                    </div>

                    {/* Options Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {/* Recursive Toggle */}
                        <div>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={recursive}
                                    onChange={(e) => setRecursive(e.target.checked)}
                                    style={{ cursor: 'pointer' }}
                                />
                                <span className="text-muted" style={{ fontSize: '0.875rem' }}>
                                    📁 Include subfolders
                                </span>
                            </label>
                        </div>

                        {/* File Extensions */}
                        <div>
                            <label className="text-muted" style={{ fontSize: '0.875rem', marginBottom: '0.25rem', display: 'block' }}>
                                File Types
                            </label>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                {Object.entries(extensions).map(([ext, enabled]) => (
                                    <label key={ext} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer' }}>
                                        <input
                                            type="checkbox"
                                            checked={enabled}
                                            onChange={(e) => setExtensions({ ...extensions, [ext]: e.target.checked })}
                                            style={{ cursor: 'pointer' }}
                                        />
                                        <span style={{ fontSize: '0.75rem', textTransform: 'uppercase' }}>{ext}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button
                            className="btn btn-secondary"
                            onClick={handleScan}
                            disabled={scanning || !imagePath.trim()}
                            style={{ flex: 1 }}
                        >
                            {scanning ? '🔍 Scanning...' : '🔍 Scan Preview'}
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleAdd}
                            disabled={adding || !imagePath.trim()}
                            style={{ flex: 1 }}
                        >
                            {adding ? 'Adding...' : '➕ Add to Queue'}
                        </button>
                    </div>
                </div>

                {/* Scan Results */}
                {scanResult && (
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px' }}>
                        <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: '#3b82f6' }}>
                            📊 Found {scanResult.total_count} image{scanResult.total_count !== 1 ? 's' : ''}
                        </div>
                        {scanResult.images.length > 0 && (
                            <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.75rem' }}>
                                {scanResult.images.map((img, idx) => (
                                    <div key={idx} style={{ padding: '0.25rem 0', display: 'flex', justifyContent: 'space-between' }}>
                                        <span className="text-muted">{img.name}</span>
                                        <span className="text-muted">{img.size_mb} MB</span>
                                    </div>
                                ))}
                                {scanResult.has_more && (
                                    <div style={{ padding: '0.25rem 0', fontStyle: 'italic', color: '#6b7280' }}>
                                        ...and {scanResult.total_count - 100} more
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Queue Actions */}
            <div className="card mb-4">
                <h3 className="card-title mb-2">🔧 Queue Actions</h3>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                        className="btn btn-secondary"
                        onClick={handleRetry}
                        disabled={retrying || ((status?.by_status.review || 0) + (status?.by_status.failed || 0) === 0)}
                    >
                        {retrying ? 'Retrying...' : '🔁 Retry Failed'}
                    </button>
                </div>
            </div>

            {/* Queue Items */}
            <h3 className="card-title mb-4">📋 Recent Images</h3>

            {status?.items.length === 0 ? (
                <div className="empty-state">
                    <p>No images in queue. Add images using the form above!</p>
                </div>
            ) : (
                <div>
                    {status?.items.map((item) => (
                        <div key={item.job_id} className="card" style={{ marginBottom: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div style={{ flex: 1 }}>
                                    <div className="card-title" style={{ marginBottom: '0.25rem' }}>
                                        {item.image_title}
                                    </div>
                                    <div className="card-meta">
                                        {item.collection_name} · {formatDate(item.added_at)}
                                        {item.megapixels && ` · ${item.megapixels.toFixed(2)} MP`}
                                        {item.attempts > 1 && ` · ${item.attempts} attempts`}
                                    </div>
                                    {item.image_path && (
                                        <div className="text-muted mt-1" style={{ fontSize: '0.7rem', wordBreak: 'break-all' }}>
                                            📁 {item.image_path}
                                        </div>
                                    )}
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
