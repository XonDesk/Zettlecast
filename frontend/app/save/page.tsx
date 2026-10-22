import SaveUrl from './SaveUrl';

// Opened by the "Save to Zettlecast" bookmarklet with ?url=<the page>: saving goes through the
// signed-in web UI like any other call, so it works from every device and the bookmarklet holds
// no token (signed out, the sign-in page comes first and returns here).

export default async function SavePage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
    const { url } = await searchParams;
    return (
        <main className="login">
            <div className="login-card">
                <h1>Zettlecast</h1>
                <SaveUrl url={url ?? ''} />
            </div>
        </main>
    );
}
