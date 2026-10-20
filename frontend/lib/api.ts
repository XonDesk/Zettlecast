// Zettlecast API Client

import type {
    Note,
    NoteDetail,
    SearchResult,
    IngestResponse,
    GraphData,
    Settings,
    SystemStatus,
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
    listNotes: async (params?: { status?: string; source_type?: string; limit?: number; offset?: number }) => {
        const searchParams = new URLSearchParams();
        if (params?.status) searchParams.set('status', params.status);
        if (params?.source_type) searchParams.set('source_type', params.source_type);
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

    getNoteSourceUrl: (uuid: string) => {
        const url = new URL(`/notes/${uuid}/source`, API_URL);
        url.searchParams.set('token', API_TOKEN);
        return url.toString();
    },

    getNoteMarkdownUrl: (uuid: string) => {
        const url = new URL(`/notes/${uuid}/markdown`, API_URL);
        url.searchParams.set('token', API_TOKEN);
        return url.toString();
    },

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
            by_status: { pending: number; processing: number; completed: number; review: number; failed: number; cancelled?: number };
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
                queue_position?: number;
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

    runPodcasts: (limit: number = 5) =>
        fetchAPI<{ status: string; pending_count?: number; limit?: number; message: string }>('/podcast/run', {
            method: 'POST',
            body: JSON.stringify({ limit }),
        }),

    getRunningStatus: () =>
        fetchAPI<{
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
        }>('/podcast/running'),

    cancelPodcastEpisode: (episodeId: string) =>
        fetchAPI<{ status: string; message: string }>(`/podcast/cancel/${episodeId}`, {
            method: 'POST',
        }),

    // Image Processing
    getImageStatus: () =>
        fetchAPI<{
            by_status: { pending: number; processing: number; completed: number; review: number; failed: number };
            total: number;
            estimated_remaining: string;
            items: Array<{
                job_id: string;
                image_title: string;
                collection_name: string;
                status: string;
                added_at: string;
                error_message: string | null;
                attempts: number;
                megapixels: number | null;
                image_path: string;
            }>;
        }>('/image/status'),

    scanImages: (path: string, recursive: boolean = true, extensions: string[] = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']) =>
        fetchAPI<{
            status: string;
            path: string;
            total_count: number;
            images: Array<{ path: string; name: string; size_mb: number }>;
            has_more: boolean;
        }>('/image/scan', {
            method: 'POST',
            body: JSON.stringify({ path, recursive, extensions }),
        }),

    addImages: (
        path: string,
        collectionName?: string,
        recursive: boolean = true,
        extensions: string[] = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']
    ) =>
        fetchAPI<{ status: string; added_count: number; job_ids: string[]; message: string }>('/image/add', {
            method: 'POST',
            body: JSON.stringify({ path, collection_name: collectionName, recursive, extensions }),
        }),

    retryFailedImages: () =>
        fetchAPI<{ status: string; retried_count: number; message: string }>('/image/retry', {
            method: 'POST',
        }),

    runImages: (limit: number = 5) =>
        fetchAPI<{ status: string; pending_count?: number; limit?: number; message: string }>('/image/run', {
            method: 'POST',
            body: JSON.stringify({ limit }),
        }),

    getImageRunningStatus: () =>
        fetchAPI<{
            is_running: boolean;
            current_image: string | null;
            processed_count: number;
            error_count: number;
            started_at: string | null;
        }>('/image/running'),

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

    // System (process page)
    getSystemStatus: () => fetchAPI<SystemStatus>('/system/status'),

    stopRun: (mode: 'after_episode' | 'abort' = 'after_episode') =>
        fetchAPI<{ status: string; message?: string }>('/system/run/stop', {
            method: 'POST',
            body: JSON.stringify({ mode }),
        }),

    resumeRun: () => fetchAPI<{ status: string }>('/system/run/resume', { method: 'POST' }),

    unloadModels: () => fetchAPI<{ status: string; message?: string }>('/system/unload', { method: 'POST' }),

    reindex: () => fetchAPI<{ status: string; message?: string }>('/system/reindex', { method: 'POST' }),

    rebuildGraph: () => fetchAPI<{ status: string; edges: number }>('/graph/rebuild', { method: 'POST' }),

    // Screenshot triage
    getTriage: (limit = 20) => fetchAPI<TriageState>(`/image/triage?limit=${limit}`),

    addTriageFolder: (path: string) =>
        fetchAPI<{ folders: string[] }>('/image/triage/folders', { method: 'POST', body: JSON.stringify({ path }) }),

    removeTriageFolder: (path: string) =>
        fetchAPI<{ folders: string[] }>('/image/triage/folders', { method: 'DELETE', body: JSON.stringify({ path }) }),

    decideTriage: (path: string, decision: TriageDecision) =>
        fetchAPI<{ decision: TriageDecision; job_id: string | null }>('/image/triage/decide', {
            method: 'POST',
            body: JSON.stringify({ path, decision }),
        }),

    undoTriage: (path: string) =>
        fetchAPI<{ undone: { decision: TriageDecision; note_kept?: boolean } | null }>('/image/triage/undo', {
            method: 'POST',
            body: JSON.stringify({ path }),
        }),

    processTriage: () =>
        fetchAPI<{ status: string; pending_count?: number; message?: string }>('/image/triage/process', { method: 'POST' }),

    /** URL of a screenshot in a triage folder, for an <img> */
    triageImageUrl: (path: string) => {
        const url = new URL('/image/triage/file', API_URL);
        url.searchParams.set('path', path);
        url.searchParams.set('token', API_TOKEN);
        return url.toString();
    },
};

export type TriageDecision = 'keep' | 'keep_image' | 'skip';

export interface TriageItem {
    path: string;
    name: string;
    folder: string;
    size_kb: number;
    modified: string;
}

export interface TriageState {
    items: TriageItem[];
    remaining: number;
    decided: Record<TriageDecision, number>;
    folders: string[];
    queue: Record<string, number>;
    processing: boolean;
}

export { APIError };

