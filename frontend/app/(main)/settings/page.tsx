'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { Settings } from '@/lib/types';

type TextKey = 'ollama_model' | 'vision_model' | 'nemo_parakeet_model' | 'nemo_diarization_model'
    | 'embedding_model' | 'reranker_model';

const TEXT_FIELDS: { section: string; fields: { key: TextKey; label: string; hint: string }[] }[] = [
    {
        section: 'LLM (Ollama)',
        fields: [
            { key: 'ollama_model', label: 'Text Model', hint: 'Transcript summaries, cleanup and tagging' },
            { key: 'vision_model', label: 'Vision Model', hint: 'Image description and OCR' },
        ],
    },
    {
        section: 'Podcast Transcription (NeMo)',
        fields: [
            { key: 'nemo_parakeet_model', label: 'ASR Model', hint: 'nvidia/parakeet-tdt-0.6b-v3 for 25 European languages' },
            { key: 'nemo_diarization_model', label: 'Diarization Model', hint: 'Speaker separation' },
        ],
    },
    {
        section: 'Search',
        fields: [
            { key: 'embedding_model', label: 'Embedding Model', hint: 'Changing it requires: zettlecast reindex' },
            { key: 'reranker_model', label: 'Reranker Model', hint: 'Cross-encoder for result ordering' },
        ],
    },
];

export default function SettingsPage() {
    const [settings, setSettings] = useState<Settings | null>(null);
    const [form, setForm] = useState<Settings | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        api.getSettings()
            .then((data) => {
                setSettings(data);
                setForm({ ...data, hf_token: '' });
            })
            .catch(() => setError('Failed to load settings. Is the API running?'))
            .finally(() => setLoading(false));
    }, []);

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!settings || !form) return;
        setMessage(null);

        const updates: Partial<Settings> = {};
        (Object.keys(form) as (keyof Settings)[]).forEach((key) => {
            if (key === 'hf_token') return;
            if (form[key] !== settings[key]) (updates as Record<string, unknown>)[key] = form[key];
        });
        if (form.hf_token) updates.hf_token = form.hf_token;

        if (Object.keys(updates).length === 0) {
            setMessage({ type: 'error', text: 'No changes to save' });
            return;
        }

        setSaving(true);
        try {
            const result = await api.updateSettings(updates);
            setMessage({ type: 'success', text: result.message });
            const fresh = await api.getSettings();
            setSettings(fresh);
            setForm({ ...fresh, hf_token: '' });
        } catch {
            setMessage({ type: 'error', text: 'Failed to save settings' });
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <div className="loading">Loading settings...</div>;
    if (error) return <div className="empty-state">{error}</div>;
    if (!settings || !form) return <div className="empty-state">No settings available</div>;

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const bookmarklet = `javascript:(function(){fetch('${apiUrl}/ingest?token=${process.env.NEXT_PUBLIC_API_TOKEN}&url='+encodeURIComponent(location.href),{method:'POST'}).then(r=>r.json()).then(j=>alert('Zettlecast: '+j.status))})();`;

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">⚙️ Settings</h1>
            </div>

            {message && (
                <div className="card mb-4" style={{
                    borderColor: message.type === 'success' ? '#22c55e' : '#ef4444',
                    background: message.type === 'success' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'
                }}>
                    <p style={{ color: message.type === 'success' ? '#22c55e' : '#ef4444' }}>{message.text}</p>
                </div>
            )}

            <form onSubmit={handleSave}>
                {TEXT_FIELDS.map(({ section, fields }) => (
                    <div key={section}>
                        <h2 className="card-title mb-4">{section}</h2>
                        <div className="settings-grid mb-4">
                            {fields.map(({ key, label, hint }) => (
                                <div className="setting-item" key={key}>
                                    <label className="setting-label">{label}</label>
                                    <input
                                        type="text"
                                        className="input"
                                        style={{ width: '100%' }}
                                        value={form[key]}
                                        onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                                    />
                                    <div className="text-muted" style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>{hint}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}

                <h2 className="card-title mb-4">Processing</h2>
                <div className="settings-grid mb-4">
                    <div className="setting-item">
                        <label className="setting-label">Chunk Size (characters)</label>
                        <input
                            type="number"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.chunk_size}
                            onChange={(e) => setForm({ ...form, chunk_size: parseInt(e.target.value) || 1200 })}
                            min={200}
                            max={4000}
                        />
                    </div>
                    <div className="setting-item">
                        <label className="setting-label" style={{ marginBottom: '0.5rem', display: 'block' }}>
                            Auto-tagging
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                            <input
                                type="checkbox"
                                checked={form.enable_auto_tagging}
                                onChange={(e) => setForm({ ...form, enable_auto_tagging: e.target.checked })}
                            />
                            <span>LLM tags + summary for PDFs, web pages and notes</span>
                        </label>
                    </div>
                    <div className="setting-item" style={{ gridColumn: 'span 2' }}>
                        <label className="setting-label">Hugging Face Token</label>
                        <input
                            type="password"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.hf_token}
                            onChange={(e) => setForm({ ...form, hf_token: e.target.value })}
                            placeholder={settings.hf_token ? 'Set (leave empty to keep)' : 'hf_... (only needed for gated models)'}
                        />
                    </div>
                </div>

                <button type="submit" className="btn btn-primary" disabled={saving}>
                    {saving ? 'Saving...' : '💾 Save Settings'}
                </button>
                <span className="text-muted" style={{ marginLeft: '1rem', fontSize: '0.875rem' }}>
                    Changes apply after restarting the server.
                </span>
            </form>

            <div style={{ marginTop: '3rem' }}>
                <h2 className="card-title mb-4">Storage</h2>
                <div className="setting-item">
                    <div className="setting-value">{settings.storage_path}</div>
                </div>
            </div>

            <div style={{ marginTop: '3rem' }}>
                <h2 className="card-title mb-4">Bookmarklet</h2>
                <div className="card">
                    <p className="text-muted mb-2">
                        Drag this link to your bookmarks bar, then click it on any page to save it:{' '}
                        <a href={bookmarklet} className="btn btn-secondary" onClick={(e) => e.preventDefault()}>📌 Save to Zettlecast</a>
                    </p>
                    <code className="setting-value" style={{
                        display: 'block',
                        padding: '1rem',
                        background: 'var(--color-bg)',
                        borderRadius: '0.375rem',
                        wordBreak: 'break-all',
                        fontSize: '0.75rem'
                    }}>
                        {bookmarklet}
                    </code>
                </div>
            </div>
        </div>
    );
}
