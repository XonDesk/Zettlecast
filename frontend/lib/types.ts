// TypeScript types for Zettlecast API

export interface Note {
    uuid: string;
    title: string;
    source_type: 'pdf' | 'web' | 'audio' | 'markdown' | 'rss';
    status: 'inbox' | 'reviewed' | 'archived';
    created_at: string;
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
}

export interface IngestResponse {
    status: 'success' | 'partial' | 'failed' | 'duplicate';
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
    whisper_model: string;
    llm_provider: string;
    ollama_model: string;
    enable_context_enrichment: boolean;
    chunk_size: number;
    storage_path: string;
    asr_backend: string;
    hf_token: string;
}

export interface PodcastEpisode {
    episode_title: string;
    podcast_name: string;
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'review';
    added_at: string;
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
