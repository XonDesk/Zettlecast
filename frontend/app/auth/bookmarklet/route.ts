import { NextResponse } from 'next/server';

// The "save this page" bookmarklet calls the API on this computer directly, so it carries the
// token; it's handed out only to a signed-in browser (proxy.ts guards this route).
export async function GET() {
    const api = 'http://localhost:8000';
    const token = process.env.ZC_API_TOKEN ?? '';
    const code = `javascript:(function(){fetch('${api}/ingest?token=${token}&url='+encodeURIComponent(location.href),{method:'POST'}).then(r=>r.json()).then(j=>alert('Zettlecast: '+j.status))})();`;
    return NextResponse.json({ bookmarklet: code });
}
