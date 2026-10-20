// TypeScript types for Zettlecast API

export interface Note {
    uuid: string;
    title: string;
    source_type: 'pdf' | 'web' | 'audio' | 'markdown' | 'image';
    status: 'inbox' | 'reviewed' | 'archived';
    created_at: string;
    tags?: string[];
}

export interface NoteDetail extends Note {
    source_path: string;
    full_text: string;
    metadata: NoteMetadata;
    suggestions?: LinkSuggestion[];
}

export interface NoteMetadata {
    author?: string;
    tags: string[];
    source_url?: string;
    language?: string;
    word_count?: number;
    page_count?: number;
    duration_seconds?: number;
    embedded_media: string[];
    custom: Record<string, unknown>;
}

export interface PassageSide {
    text: string;
    time: number | null;
}

export interface LinkSuggestion {
    uuid: string;
    title: string;
    score: number;
    reason?: string;
    kind?: 'similar' | 'duplicate' | 'link';
    /** Most alike passages, from the viewed note ("here") and the linked one ("there") */
    passage?: { here: PassageSide; there: PassageSide; similarity?: number } | null;
}

export interface SearchResult {
    uuid: string;
    title: string;
    score: number;
    snippet: string;
    source_type: string;
    timestamp_seconds?: number | null;
}

export interface IngestResponse {
    status: 'success' | 'queued' | 'failed' | 'duplicate';
    uuid?: string;
    title?: string;
    error?: string;
}

export interface GraphNode {
    id: string;
    name: string;
    source_type: string;
    val: number;
    community: number | null;
    tags: string[];
    people: string[];
    organisations: string[];
    source: string; // podcast show, website host, or type
    date: string | null; // YYYY-MM-DD: published (podcasts) or added
    duration: number | null; // seconds (audio)
}

export interface GraphLink {
    source: string;
    target: string;
    value: number;
    kind: 'similar' | 'duplicate' | 'link';
}

export interface GraphCommunity {
    id: number;
    label: string;
    size: number;
}

export interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
    communities: GraphCommunity[];
}

export interface Settings {
    embedding_model: string;
    reranker_model: string;
    ollama_model: string;
    vision_model: string;
    nemo_parakeet_model: string;
    nemo_diarization_model: string;
    enable_auto_tagging: boolean;
    chunk_size: number;
    storage_path: string;
    hf_token: string;
}

export interface PodcastEpisode {
    episode_title: string;
    podcast_name: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'review';
    added_at: string;
    audio_path: string;
}

export interface PodcastQueueStatus {
    by_status: {
        pending: number;
        processing: number;
        completed: number;
        review: number;
    };
    estimated_remaining: string;
}

export interface ImageItem {
    job_id: string;
    image_title: string;
    collection_name: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'review';
    added_at: string;
    error_message: string | null;
    attempts: number;
    megapixels: number | null;
    image_path: string;
}

export interface ImageQueueStatus {
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

// --- Process page (GET /system/status) ---

export interface RunRecent {
    title: string;
    status: string;
    seconds: number;
    audio_minutes: number;
    finished_at: string;
}

export interface RunState {
    phase: 'idle' | 'starting' | 'transcribe' | 'enhance' | 'graph' | 'running';
    alive: boolean;
    source?: 'cli' | 'api' | 'unknown';
    pid?: number;
    legacy?: boolean; // a run that doesn't publish stages or honour stop requests
    started_at?: string;
    phase_total?: number;
    phase_done?: number;
    episode_title?: string | null;
    episode_started_at?: string | null;
    stage?: string | null;
    stage_started_at?: string | null;
    chunk?: number;
    chunks?: number;
    completed?: number;
    failed?: number;
    recent?: RunRecent[];
    stop_requested?: boolean;
}

export interface Alarm {
    id: string;
    priority: 'high' | 'medium' | 'low';
    text: string;
}

export interface SystemStatus {
    time: string;
    run: RunState;
    queues: {
        podcast: {
            by_status: Record<string, number>;
            total: number;
            failed: { id: string; title: string; error: string }[];
            workers: { pid: number; title: string }[];
        };
        images: { total: number; by_status: Record<string, number> };
    };
    gpu: {
        name: string;
        memory_used_mib: number | null;
        memory_total_mib: number | null;
        utilization: number | null;
        temperature: number | null;
        power_w: number | null;
    } | null;
    llm: {
        backend: string;
        model: string;
        server: { installed: boolean; pid: number | null; vision: boolean; state: 'ready' | 'loading' | 'stopped' };
        ollama: { reachable: boolean; loaded: string[] };
    };
    models: { embedder: boolean; reranker: boolean; acoustic_judge: boolean };
    database: { notes?: number; chunks?: number; edges?: number; communities?: number; by_source_type?: Record<string, number> };
    disk: { free_gb: number; total_gb: number };
    task: { name: string | null; running: boolean; started_at: string | null; finished_at: string | null; error: string | null };
    alarms: Alarm[];
    limits: { vram_warn: number; vram_alarm: number };
}
