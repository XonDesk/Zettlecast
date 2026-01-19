'use client';

import { useRouter } from 'next/navigation';
import Graph from '@/components/Graph';
import type { GraphNode } from '@/lib/types';

export default function GraphPage() {
    const router = useRouter();

    const handleNodeClick = (node: GraphNode) => {
        router.push(`/notes/${node.id}`);
    };

    return (
        <div>
            <div className="page-header">
                <h1 className="page-title">📊 Knowledge Graph</h1>
                <p className="text-muted">
                    Click on a node to view the note. Drag to pan, scroll to zoom.
                </p>
            </div>

            <Graph onNodeClick={handleNodeClick} />
        </div>
    );
}
