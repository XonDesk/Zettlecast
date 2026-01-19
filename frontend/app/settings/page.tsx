'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { Settings } from '@/lib/types';

export default function SettingsPage() {
    const [settings, setSettings] = useState<Settings | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    // Editable form state
    const [form, setForm] = useState({
        llm_provider: '',
        ollama_model: '',
        whisper_model: '',
        asr_backend: '',
        enable_context_enrichment: false,
        chunk_size: 512,
        hf_token: '',
    });

    useEffect(() => {
        const fetchSettings = async () => {
            try {
                setLoading(true);
                const data = await api.getSettings();
                setSettings(data);
                setForm({
                    llm_provider: data.llm_provider,
                    ollama_model: data.ollama_model,
                    whisper_model: data.whisper_model,
                    asr_backend: data.asr_backend,
                    enable_context_enrichment: data.enable_context_enrichment,
                    chunk_size: data.chunk_size,
                    hf_token: data.hf_token === '***' ? '' : data.hf_token,
                });
            } catch (err) {
                setError('Failed to load settings. Is the API running?');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchSettings();
    }, []);

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        setMessage(null);
        setSaving(true);

        try {
            // Only send changed values
            const updates: Partial<Settings> = {};

            if (form.llm_provider !== settings?.llm_provider) {
                updates.llm_provider = form.llm_provider;
            }
            if (form.ollama_model !== settings?.ollama_model) {
                updates.ollama_model = form.ollama_model;
            }
            if (form.whisper_model !== settings?.whisper_model) {
                updates.whisper_model = form.whisper_model;
            }
            if (form.asr_backend !== settings?.asr_backend) {
                updates.asr_backend = form.asr_backend;
            }
            if (form.enable_context_enrichment !== settings?.enable_context_enrichment) {
                updates.enable_context_enrichment = form.enable_context_enrichment;
            }
            if (form.chunk_size !== settings?.chunk_size) {
                updates.chunk_size = form.chunk_size;
            }
            if (form.hf_token && form.hf_token !== '***') {
                updates.hf_token = form.hf_token;
            }

            if (Object.keys(updates).length === 0) {
                setMessage({ type: 'error', text: 'No changes to save' });
                return;
            }

            const result = await api.updateSettings(updates);
            setMessage({ type: 'success', text: result.message });

            // Refresh settings
            const newSettings = await api.getSettings();
            setSettings(newSettings);
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to save settings' });
            console.error(err);
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <div className="loading">Loading settings...</div>;
    if (error) return <div className="empty-state">{error}</div>;
    if (!settings) return <div className="empty-state">No settings available</div>;

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">⚙️ Settings</h1>
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

            <form onSubmit={handleSave}>
                <h2 className="card-title mb-4">LLM Configuration</h2>
                <div className="settings-grid mb-4">
                    <div className="setting-item">
                        <label className="setting-label">LLM Provider</label>
                        <select
                            className="select"
                            style={{ width: '100%' }}
                            value={form.llm_provider}
                            onChange={(e) => setForm({ ...form, llm_provider: e.target.value })}
                        >
                            <option value="ollama">Ollama (Local)</option>
                            <option value="openai">OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                        </select>
                    </div>
                    <div className="setting-item">
                        <label className="setting-label">Ollama Model</label>
                        <input
                            type="text"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.ollama_model}
                            onChange={(e) => setForm({ ...form, ollama_model: e.target.value })}
                            placeholder="llama3.2:3b"
                        />
                    </div>
                </div>

                <h2 className="card-title mb-4">Audio Transcription</h2>
                <div className="settings-grid mb-4">
                    <div className="setting-item">
                        <label className="setting-label">ASR Backend</label>
                        <select
                            className="select"
                            style={{ width: '100%' }}
                            value={form.asr_backend}
                            onChange={(e) => setForm({ ...form, asr_backend: e.target.value })}
                        >
                            <option value="auto">Auto (best for platform)</option>
                            <option value="parakeet-mlx">Parakeet-MLX (Mac)</option>
                            <option value="whisper">Whisper</option>
                            <option value="nemo">NeMo (NVIDIA)</option>
                        </select>
                    </div>
                    <div className="setting-item">
                        <label className="setting-label">Whisper Model</label>
                        <input
                            type="text"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.whisper_model}
                            onChange={(e) => setForm({ ...form, whisper_model: e.target.value })}
                            placeholder="large-v3-turbo"
                        />
                    </div>
                    <div className="setting-item" style={{ gridColumn: 'span 2' }}>
                        <label className="setting-label">HuggingFace Token (for speaker diarization)</label>
                        <input
                            type="password"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.hf_token}
                            onChange={(e) => setForm({ ...form, hf_token: e.target.value })}
                            placeholder="hf_xxx... (leave empty to keep current)"
                        />
                    </div>
                </div>

                <h2 className="card-title mb-4">Processing</h2>
                <div className="settings-grid mb-4">
                    <div className="setting-item">
                        <label className="setting-label">Chunk Size</label>
                        <input
                            type="number"
                            className="input"
                            style={{ width: '100%' }}
                            value={form.chunk_size}
                            onChange={(e) => setForm({ ...form, chunk_size: parseInt(e.target.value) || 512 })}
                            min={100}
                            max={2000}
                        />
                    </div>
                    <div className="setting-item">
                        <label className="setting-label" style={{ marginBottom: '0.5rem', display: 'block' }}>
                            Context Enrichment
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                            <input
                                type="checkbox"
                                checked={form.enable_context_enrichment}
                                onChange={(e) => setForm({ ...form, enable_context_enrichment: e.target.checked })}
                            />
                            <span>Enable LLM context enrichment</span>
                        </label>
                    </div>
                </div>

                <button type="submit" className="btn btn-primary" disabled={saving}>
                    {saving ? 'Saving...' : '💾 Save Settings'}
                </button>
            </form>

            <div style={{ marginTop: '3rem' }}>
                <h2 className="card-title mb-4">Read-Only Configuration</h2>
                <div className="settings-grid">
                    <div className="setting-item">
                        <div className="setting-label">Embedding Model</div>
                        <div className="setting-value">{settings.embedding_model}</div>
                    </div>
                    <div className="setting-item">
                        <div className="setting-label">Reranker Model</div>
                        <div className="setting-value">{settings.reranker_model}</div>
                    </div>
                    <div className="setting-item" style={{ gridColumn: 'span 2' }}>
                        <div className="setting-label">Storage Path</div>
                        <div className="setting-value">{settings.storage_path}</div>
                    </div>
                </div>
            </div>

            <div style={{ marginTop: '3rem' }}>
                <h2 className="card-title mb-4">Bookmarklet</h2>
                <div className="card">
                    <p className="text-muted mb-2">Drag this to your bookmarks bar to save pages:</p>
                    <code className="setting-value" style={{
                        display: 'block',
                        padding: '1rem',
                        background: 'var(--color-bg)',
                        borderRadius: '0.375rem',
                        wordBreak: 'break-all',
                        fontSize: '0.75rem'
                    }}>
                        javascript:(function()&#123;fetch(&apos;{process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/ingest?token={process.env.NEXT_PUBLIC_API_TOKEN}&url=&apos;+encodeURIComponent(location.href))&#125;)();
                    </code>
                </div>
            </div>
        </div>
    );
}
