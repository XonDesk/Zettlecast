import { NextResponse, type NextRequest } from 'next/server';

// The "save this page" bookmarklet opens /save?url=<page> on this web UI, at the address the
// browser uses for it (this computer, its network address, over the VPN): the save goes through
// the sign-in like any other call, so the bookmarklet holds no token and works on every device.
export async function GET(request: NextRequest) {
    const host = request.headers.get('x-forwarded-host') ?? request.headers.get('host') ?? request.nextUrl.host;
    const proto = request.headers.get('x-forwarded-proto') ?? request.nextUrl.protocol.replace(':', '');
    const origin = `${proto}://${host}`;
    const code = `javascript:(function(){window.open('${origin}/save?url='+encodeURIComponent(location.href),'zettlecast','width=440,height=260')})();`;
    return NextResponse.json({ bookmarklet: code });
}
