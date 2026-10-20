/**
 * Login for the web UI (server side only: the request gate in proxy.ts and the /auth routes).
 *
 * The browser never sees the API token: it calls /zc-api/..., and the gate checks the session
 * cookie and forwards the call to the API (127.0.0.1 only) with the token added. Credentials and
 * secrets come from frontend/.env.local (no NEXT_PUBLIC_ prefix, so they stay on the server):
 *   ZC_LOGIN_USER, ZC_LOGIN_HASH (pbkdf2_sha256:iterations:salt:hash, base64url; ":" because
 *   .env files expand "$"), ZC_SESSION_SECRET, ZC_API_TOKEN, ZC_API_URL
 */

export const SESSION_COOKIE = 'zc_session';
export const SESSION_DAYS = 365;

const encoder = new TextEncoder();

function b64(bytes: ArrayBuffer | Uint8Array): string {
    return Buffer.from(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)).toString('base64url');
}

function fromB64(text: string): Uint8Array {
    return new Uint8Array(Buffer.from(text, 'base64url'));
}

/** Constant-time comparison of two strings */
function equal(a: string, b: string): boolean {
    if (a.length !== b.length) return false;
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return diff === 0;
}

async function hmac(value: string): Promise<string> {
    const secret = process.env.ZC_SESSION_SECRET;
    if (!secret) throw new Error('ZC_SESSION_SECRET is not set');
    const key = await crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    return b64(await crypto.subtle.sign('HMAC', key, encoder.encode(value)));
}

/** Cookie value: user.expiry.signature */
export async function createSession(user: string): Promise<string> {
    const expires = Date.now() + SESSION_DAYS * 86_400_000;
    const value = `${b64(encoder.encode(user))}.${expires}`;
    return `${value}.${await hmac(value)}`;
}

export async function validSession(cookie: string | undefined): Promise<boolean> {
    if (!cookie) return false;
    const parts = cookie.split('.');
    if (parts.length !== 3) return false;
    const [user, expires, signature] = parts;
    if (!(Number(expires) > Date.now())) return false;
    try {
        return equal(signature, await hmac(`${user}.${expires}`));
    } catch {
        return false;
    }
}

export async function checkPassword(user: string, password: string): Promise<boolean> {
    const wantUser = process.env.ZC_LOGIN_USER ?? '';
    const [scheme, iterations, salt, hash] = (process.env.ZC_LOGIN_HASH ?? '').split(':');
    if (!wantUser || scheme !== 'pbkdf2_sha256') return false;
    const key = await crypto.subtle.importKey('raw', encoder.encode(password), 'PBKDF2', false, ['deriveBits']);
    const bits = await crypto.subtle.deriveBits(
        { name: 'PBKDF2', hash: 'SHA-256', salt: fromB64(salt) as BufferSource, iterations: Number(iterations) },
        key,
        fromB64(hash).length * 8,
    );
    // user names are compared ignoring case (phones capitalise the first letter)
    const userOk = equal(user.trim().toLowerCase(), wantUser.toLowerCase());
    return equal(b64(bits), hash) && userOk;
}
