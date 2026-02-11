'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function stripFrontmatter(text: string): string {
    if (text.startsWith('---')) {
        const endIndex = text.indexOf('---', 3);
        if (endIndex !== -1) {
            return text.slice(endIndex + 3).trimStart();
        }
    }
    return text;
}

export default function MarkdownContent({ content }: { content: string }) {
    const markdown = stripFrontmatter(content);
    return (
        <div className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {markdown}
            </ReactMarkdown>
        </div>
    );
}
