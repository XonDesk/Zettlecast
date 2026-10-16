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

export interface LinkSuggestion {
    uuid: string;
    title: string;
    score: number;
    reason?: string;
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
}

export interface GraphLink {
    source: string;
    target: string;
    value: number;
}

export interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
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
