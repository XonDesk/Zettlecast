'use client';

import { useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import type { SearchResult } from '@/lib/types';
import { formatTimestamp } from '@/lib/utils';

export default function SearchPage() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [searched, setSearched] = useState(false);
    const [topK, setTopK] = useState(5);
    const [useRerank, setUseRerank] = useState(true);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (query.length < 3) return;

        try {
            setLoading(true);
            const data = await api.search(query, topK, useRerank);
            setResults(data.results);
            setSearched(true);
        } catch (err) {
            console.error('Search failed:', err);
        } finally {
            setLoading(false);
        }
    };

    const getTypeEmoji = (type: string) => {
        const emojis: Record<string, string> = {
            pdf: '📄',
            web: '🌐',
            audio: '🎙️',
            markdown: '📝',
            image: '🖼️',
        };
        return emojis[type] || '📋';
    };

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">🔍 Search</h1>
            </div>

            <form onSubmit={handleSearch}>
                <div className="search-input-container">
                    <input
                        type="text"
                        className="input search-input"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="What are you looking for?"
                        minLength={3}
                    />
                </div>

                <div className="search-options">
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="text-muted">Results:</span>
                        <input
                            type="range"
                            min="1"
                            max="20"
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            style={{ width: '100px' }}
                        />
                        <span>{topK}</span>
                    </label>

                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            type="checkbox"
                            checked={useRerank}
                            onChange={(e) => setUseRerank(e.target.checked)}
                        />
                        <span>Use reranking</span>
                    </label>

                    <button type="submit" className="btn btn-primary" disabled={query.length < 3 || loading}>
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </div>
            </form>

            {query.length > 0 && query.length < 3 && (
                <p className="text-muted">Enter at least 3 characters to search</p>
            )}

            {searched && (
                <div className="mt-4">
                    {results.length === 0 ? (
                        <div className="empty-state">
                            <p>No results found. Try a different query.</p>
                        </div>
                    ) : (
                        <>
                            <p className="text-muted mb-4">Found {results.length} results</p>
                            {results.map((result, index) => (
                                <Link
                                    key={result.uuid}
                                    href={result.timestamp_seconds != null
                                        ? `/notes/${result.uuid}?t=${Math.floor(result.timestamp_seconds)}`
                                        : `/notes/${result.uuid}`}
                                    style={{ textDecoration: 'none' }}
                                >
                                    <div className="search-result">
                                        <div className="search-result-title">
                                            {index + 1}. {getTypeEmoji(result.source_type)} {result.title}
                                        </div>
                                        <div className="search-result-meta">
                                            Score: {result.score.toFixed(3)} · {result.source_type}
                                            {result.timestamp_seconds != null && (
                                                <> · ▶ {formatTimestamp(result.timestamp_seconds)}</>
                                            )}
                                        </div>
                                        <div className="search-result-snippet" style={{ whiteSpace: 'pre-line' }}>
                                            {result.snippet.length > 600 ? result.snippet.slice(0, 600) + '…' : result.snippet}
                                        </div>
                                    </div>
                                </Link>
                            ))}
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
