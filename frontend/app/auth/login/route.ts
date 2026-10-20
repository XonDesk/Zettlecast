import { NextResponse, type NextRequest } from 'next/server';
import { SESSION_COOKIE, SESSION_DAYS, checkPassword, createSession } from '@/lib/auth';

// Failed sign-ins slow down, then lock for a while (one person, one home network: a simple
// global counter is enough)
const LOCK_AFTER = 10;
const LOCK_MS = 10 * 60_000;
let failures = 0;
let lockedUntil = 0;

/**
 * Redirect to a path on this site. Relative, because the dev server listens on 0.0.0.0 and
 * request.url carries that, not the address the phone used.
 */
function redirectTo(path: string): NextResponse {
    return new NextResponse(null, { status: 303, headers: { Location: path } });
}

/** Only same-site paths after signing in (never another site) */
function safeNext(value: FormDataEntryValue | null): string {
    const next = typeof value === 'string' ? value : '/';
    return next.startsWith('/') && !next.startsWith('//') ? next : '/';
}

export async function POST(request: NextRequest) {
    const form = await request.formData();
    const next = safeNext(form.get('next'));
    const back = (error: string) => {
        const query = new URLSearchParams({ error, ...(next !== '/' ? { next } : {}) });
        return redirectTo(`/login?${query}`);
    };

    if (Date.now() < lockedUntil) return back('locked');
    const ok = await checkPassword(String(form.get('user') ?? ''), String(form.get('password') ?? ''));
    if (!ok) {
        failures += 1;
        if (failures >= LOCK_AFTER) {
            lockedUntil = Date.now() + LOCK_MS;
            failures = 0;
        }
        await new Promise((r) => setTimeout(r, 1000));
        return back('wrong');
    }
    failures = 0;
    const response = redirectTo(next);
    response.cookies.set(SESSION_COOKIE, await createSession(String(form.get('user'))), {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        maxAge: SESSION_DAYS * 86_400,
    });
    return response;
}
