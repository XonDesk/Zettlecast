import { NextResponse } from 'next/server';
import { SESSION_COOKIE } from '@/lib/auth';

export async function POST() {
    // relative: request.url carries the dev server's 0.0.0.0, not the address used
    const response = new NextResponse(null, { status: 303, headers: { Location: '/login' } });
    response.cookies.delete(SESSION_COOKIE);
    return response;
}
