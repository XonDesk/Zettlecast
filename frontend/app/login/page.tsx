// Sign-in form: a plain HTML post to /auth/login (works before any script has loaded)

const ERRORS: Record<string, string> = {
    wrong: 'Wrong user name or password.',
    locked: 'Too many tries. Wait ten minutes and try again.',
};

export default async function LoginPage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
    const { error, next } = await searchParams;
    return (
        <main className="login">
            <form className="login-card" method="post" action="/auth/login">
                <h1>Zettlecast</h1>
                {error && <p className="login-error" role="alert">{ERRORS[error] ?? 'Sign-in failed.'}</p>}
                <label>
                    User name
                    <input className="input" name="user" autoComplete="username" autoCapitalize="none"
                           autoCorrect="off" spellCheck={false} required autoFocus />
                </label>
                <label>
                    Password
                    <input className="input" name="password" type="password" autoComplete="current-password" required />
                </label>
                <input type="hidden" name="next" value={next ?? '/'} />
                <button className="btn btn-primary" type="submit">Sign in</button>
            </form>
        </main>
    );
}
