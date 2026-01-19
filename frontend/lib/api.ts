// Zettlecast API Client

import type {
    Note,
    NoteDetail,
    SearchResult,
    IngestResponse,
    GraphData,
    Settings,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN || '';

class APIError extends Error {
    constructor(public status: number, message: string) {
        super(message);
        this.name = 'APIError';
    }
}

async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const url = new URL(endpoint, API_URL);
    url.searchParams.set('token', API_TOKEN);

    const res = await fetch(url.toString(), {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!res.ok) {
        throw new APIError(res.status, `API Error: ${res.status} ${res.statusText}`);
    }

    return res.json();
}

export const api = {
    // Notes
    listNotes: async (params?: { status?: string; limit?: number; offset?: number }) => {
        const searchParams = new URLSearchParams();
        if (params?.status) searchParams.set('status', params.status);
        if (params?.limit) searchParams.set('limit', String(params.limit));
        if (params?.offset) searchParams.set('offset', String(params.offset));

        const query = searchParams.toString();
        return fetchAPI<{ notes: Note[]; count: number }>(
            `/notes${query ? `?${query}` : ''}`
        );
    },

    getNote: (uuid: string, includeSuggestions = true) =>
        fetchAPI<NoteDetail>(
            `/notes/${uuid}?include_suggestions=${includeSuggestions}`
        ),

    deleteNote: (uuid: string) =>
        fetchAPI<{ status: string; uuid: string }>(`/notes/${uuid}`, {
            method: 'DELETE',
        }),

    // Search
    search: (query: string, topK = 5, rerank = true) =>
        fetchAPI<{ query: string; results: SearchResult[]; count: number }>(
            `/search?q=${encodeURIComponent(query)}&top_k=${topK}&rerank=${rerank}`
        ),

    // Ingest
    ingest: (url: string) =>
        fetchAPI<IngestResponse>(
            `/ingest?url=${encodeURIComponent(url)}`,
            { method: 'POST' }
        ),

    // Graph
    getGraph: (limit = 2000) =>
        fetchAPI<GraphData>(`/graph?limit=${limit}`),

    // Links
    manageLink: (uuid: string, targetUuid: string, action: 'accept' | 'reject') =>
        fetchAPI<{ status: string }>(`/notes/${uuid}/link`, {
            method: 'POST',
            body: JSON.stringify({ target_uuid: targetUuid, action }),
        }),

    // Podcast
    getPodcastStatus: () =>
        fetchAPI<{
            by_status: { pending: number; processing: number; completed: number; review: number; failed: number };
            total: number;
            estimated_remaining: string;
            items: Array<{
                job_id: string;
                podcast_name: string;
                episode_title: string;
                status: string;
                added_at: string;
                error_message: string | null;
                attempts: number;
            }>;
        }>('/podcast/status'),

    importPodcastFeed: (feedUrl: string, limit: number = 5) =>
        fetchAPI<{ status: string; added_count: number; job_ids: string[]; message: string }>('/podcast/import', {
            method: 'POST',
            body: JSON.stringify({ feed_url: feedUrl, limit }),
        }),

    retryFailedPodcasts: () =>
        fetchAPI<{ status: string; retried_count: number; message: string }>('/podcast/retry', {
            method: 'POST',
        }),

    syncPodcastQueue: () =>
        fetchAPI<{ status: string; sync_stats: Record<string, number>; message: string }>('/podcast/sync', {
            method: 'POST',
        }),

    resetStuckPodcasts: () =>
        fetchAPI<{ status: string; reset_count: number; message: string }>('/podcast/reset-stuck', {
            method: 'POST',
        }),

    runPodcasts: (limit: number = 5, backend?: string) =>
        fetchAPI<{ status: string; pending_count?: number; limit?: number; message: string }>('/podcast/run', {
            method: 'POST',
            body: JSON.stringify({ limit, backend }),
        }),

    getRunningStatus: () =>
        fetchAPI<{
            is_running: boolean;
            current_episode: string | null;
            processed_count: number;
            error_count: number;
            started_at: string | null;
        }>('/podcast/running'),

    // Settings
    getSettings: () => fetchAPI<Settings>('/settings'),

    updateSettings: (settings: Partial<Settings>) =>
        fetchAPI<{ status: string; updated_keys: string[]; message: string }>('/settings', {
            method: 'POST',
            body: JSON.stringify(settings),
        }),

    // Health
    checkHealth: () =>
        fetchAPI<{ status: string; version: string }>('/health'),
};

export { APIError };

