'use client';

import dynamic from 'next/dynamic';

// Sigma uses WebGL, so render on the client only
const Graph = dynamic(() => import('@/components/Graph'), {
    ssr: false,
    loading: () => <div className="graph-loading">Loading graph...</div>,
});

export default function GraphPage() {
    return <Graph />;
}
