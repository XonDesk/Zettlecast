// TypeScript types for Zettlecast API

export interface Note {
    uuid: string;
    title: string;
    source_type: 'pdf' | 'web' | 'audio' | 'markdown' | 'image';
    status: 'inbox' | 'reviewed' | 'archived';
    created_at: string;
    tags?: string[];
    waiting?: boolean; // stored on the primary, not indexed yet: not searchable or linked until the desktop runs
    add_failed?: boolean; // the desktop couldn't read this page or file (the note says why)
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

/** What a note links to (url) or mentions (name, looked up on Wikipedia) */
export type Reference =
    | { kind: 'url'; url: string; label: string; domain: string; context: string; low_value: boolean; existing_uuid: string | null }
    | { kind: 'name'; label: string; query: string; what: string };

export interface ReferenceCandidate {
    title: string;
    url: string;
    description: string;
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
    brief?: string; // summary, image description or the start of the note
    image?: boolean; // a screenshot kept with its image: api.getNoteThumbUrl
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
    reddit_saved_feed: string; // "***" when set: the link is never sent back
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
    phase_started_at?: string | null;
    cycles?: number[]; // seconds per finished episode of this phase, newest last
    averages?: Partial<Record<'transcribe' | 'enhance' | 'graph', number>>; // from earlier runs
    stage_times?: Record<string, number>;
    last_stages?: Record<string, number>; // step times of the previous episode
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

/** The worker PC as the primary sees it (two-machine setup: ROLE=primary) */
export interface DesktopStatus {
    state: 'offline' | 'waking' | 'idle' | 'working';
    online: boolean;
    worker_id: string | null;
    host: string | null;
    last_seen_at: string | null;
    paused: boolean; // Stop on this page: no new work until Resume
    stay_on: boolean; // no idle shutdown
    wake: { configured: boolean; window: string; sent_at: string | null; misses: number; gave_up: boolean; reason: string | null };
    // graph: 1 when the graph is out of date (rebuilt last, after the rest; not in total: it doesn't wake the PC)
    pending: { podcasts: number; images: number; ingest: number; tags: number; insights: number; graph: number; total: number };
    idle_shutdown_minutes: number;
}

export interface SystemStatus {
    time: string;
    role?: 'standalone' | 'primary' | 'worker';
    desktop?: DesktopStatus; // only on the primary
    worker_disk?: { free_gb: number; total_gb: number } | null;
    run: RunState;
    queues: {
        podcast: {
            by_status: Record<string, number>;
            total: number;
            failed: { id: string; title: string; error: string }[];
            workers: { pid: number; title: string }[];
            transcribed?: number; // pending episodes with a saved transcription (waiting for pass 2)
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
    // image run in the API server (absent from an older API server)
    images?: {
        is_running: boolean;
        phase: 'reading' | 'linking' | null;
        processed_count: number;
        error_count: number;
        total: number;
        in_flight: string[];
        stop_requested: boolean;
        started_at: string | null;
        finished_at: string | null;
        done_times?: string[]; // when the last images finished (pace, ETA)
        phase_started_at?: string | null; // linking started
        link_seconds?: number | null; // the last run's graph rebuild
    };
    // web pages and documents being added through the API (absent from an older API server)
    web?: {
        in_flight: { title: string; kind: string; started_at: string }[];
        recent: { title: string; kind: string; status: string; seconds: number; finished_at: string }[];
    };
    // screenshot sorting (cached ~30 s); null when it couldn't be read
    triage?: { remaining: number; decided: Record<string, number>; folders: number; to_recycle: number } | null;
    // absent from an API server started before Reddit import existed
    reddit?: {
        configured: boolean;
        imported: number;
        partial: number; // imported without their thread (sync with refresh to retry)
        untagged?: number; // imported without LLM tags/summary (tag later)
        last_sync: string | null;
        progress: { running: boolean; done: number; total: number; current: string | null;
                    done_times?: string[]; phase?: 'sync' | 'tag' };
    } | null;
    alarms: Alarm[];
    limits: { vram_warn: number; vram_alarm: number };
}

// Today's insight cards (GET /today)
export interface InsightSide {
    uuid: string;
    title: string;
    source_type: string;
    community: string | null;
    date: string | null;
    text: string; // the passage matching the other note, or the note's summary
    time: number | null; // where the passage is, in a transcript
    passage: boolean;
    image: boolean;
}

export interface InsightCard {
    key: string;
    kind: 'echo' | 'surprise';
    weight: number;
    shared: string[];
    a: InsightSide;
    b: InsightSide;
    insight: { relation: 'same_subject' | 'agrees' | 'contradicts' | 'extends' | 'applies' | 'unrelated'; insight: string;
               question: string; note_title: string } | null; // null while the LLM prepares it
    verdict: 'good' | 'rejected' | 'noted' | null;
    note?: string; // uuid of the note written from it
}

export interface TodayResponse {
    date: string;
    cards: InsightCard[];
    preparing: number; // cards still waiting for the LLM
    generating: boolean;
}
