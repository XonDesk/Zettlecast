import { NextResponse, type NextRequest } from 'next/server';
import { SESSION_COOKIE, validSession } from '@/lib/auth';

// Every page and API call needs a session (the web UI can be opened from other devices on the
// network). API calls arrive as /zc-api/<path> and go on to the API with its token, which
// never reaches the browser.

const OPEN = ['/login', '/auth/login', '/_next/', '/favicon.ico'];

export async function proxy(request: NextRequest) {
    const { pathname, search } = request.nextUrl;
    if (OPEN.some((p) => pathname === p || (p.endsWith('/') && pathname.startsWith(p)))) {
        return NextResponse.next();
    }
    const signedIn = await validSession(request.cookies.get(SESSION_COOKIE)?.value);

    if (pathname === '/zc-api' || pathname.startsWith('/zc-api/')) {
        if (!signedIn) return NextResponse.json({ detail: 'Not signed in' }, { status: 401 });
        const target = new URL(pathname.slice('/zc-api'.length) || '/', process.env.ZC_API_URL || 'http://127.0.0.1:8000');
        target.search = search;
        target.searchParams.set('token', process.env.ZC_API_TOKEN ?? '');
        return NextResponse.rewrite(target);
    }

    if (!signedIn) {
        const login = new URL('/login', request.url);
        if (pathname !== '/') login.searchParams.set('next', pathname + search);
        return NextResponse.redirect(login);
    }
    return NextResponse.next();
}

export const config = {
    matcher: '/:path*',
};
