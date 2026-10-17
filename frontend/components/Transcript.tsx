'use client';

import { formatTimestamp } from '@/lib/utils';

const TURN = /^\[([\d.]+)s - ([\d.]+)s\] (?:([^:]{1,60}): )?(.*)$/;
const SPEAKER_COLORS = ['#60a5fa', '#4ade80', '#f472b6', '#facc15', '#a78bfa', '#fb923c', '#2dd4bf', '#f87171'];
const SENTENCES_PER_PARAGRAPH = 4;

interface Turn {
    start: number;
    speaker: string;
    text: string;
}

export function parseTranscript(text: string): Turn[] {
    const turns: Turn[] = [];
    for (const line of text.split('\n')) {
        const m = TURN.exec(line.trim());
        if (m) {
            turns.push({ start: Number(m[1]), speaker: m[3] || '', text: m[4] });
        } else if (line.trim() && turns.length) {
            turns[turns.length - 1].text += ' ' + line.trim();
        }
    }
    return turns;
}

function paragraphs(text: string): string[] {
    const sentences = text.match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [text];
    const out: string[] = [];
    for (let i = 0; i < sentences.length; i += SENTENCES_PER_PARAGRAPH) {
        out.push(sentences.slice(i, i + SENTENCES_PER_PARAGRAPH).join('').trim());
    }
    return out;
}

export default function Transcript({ text, onSeek }: { text: string; onSeek?: (seconds: number) => void }) {
    const turns = parseTranscript(text);
    const speakers = Array.from(new Set(turns.map((t) => t.speaker)));
    const colorOf = (speaker: string) => SPEAKER_COLORS[speakers.indexOf(speaker) % SPEAKER_COLORS.length];

    return (
        <div className="transcript">
            {turns.map((turn, i) => (
                <div key={i} className="transcript-turn">
                    <div className="transcript-turn-header">
                        <button
                            className="transcript-time"
                            onClick={() => onSeek?.(turn.start)}
                            title="Play from here"
                            disabled={!onSeek}
                        >
                            {formatTimestamp(turn.start)}
                        </button>
                        {turn.speaker && (
                            <span className="transcript-speaker" style={{ color: colorOf(turn.speaker) }}>
                                {turn.speaker}
                            </span>
                        )}
                    </div>
                    {paragraphs(turn.text).map((p, j) => (
                        <p key={j}>{p}</p>
                    ))}
                </div>
            ))}
        </div>
    );
}
