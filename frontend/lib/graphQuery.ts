/**
 * Query language of the graph view's filter and colour groups.
 *
 *   heat acclimation          title, topics, people or organisations contain the words
 *   tag:hydration             a topic contains "hydration"
 *   person:"Stephanie Howe"   a person (quotes for spaces)
 *   org:CTS  source:KoopCast  type:audio  community:3  year:2023  year:2021-2023
 *   -tag:nutrition            leading "-" negates a term
 *
 * Every term must match.
 */

export interface QueryNode {
    label: string;
    sourceType: string;
    community: number | null;
    tags: string[];
    people: string[];
    organisations: string[];
    source: string;
    date: string | null;
}

export const QUERY_FIELDS = ['tag', 'person', 'org', 'source', 'type', 'community', 'year'] as const;
export type QueryField = (typeof QUERY_FIELDS)[number];

interface Term {
    field: QueryField | null;
    value: string;
    negate: boolean;
}

const TERM = /(-?)(?:(\w+):)?(?:"([^"]*)"|(\S+))/g;

export function parseQuery(query: string): Term[] {
    const terms: Term[] = [];
    for (const m of query.matchAll(TERM)) {
        const field = m[2]?.toLowerCase();
        const value = (m[3] ?? m[4] ?? '').toLowerCase().trim();
        if (!value) continue;
        if (field && (QUERY_FIELDS as readonly string[]).includes(field)) {
            terms.push({ field: field as QueryField, value, negate: m[1] === '-' });
        } else {
            // unknown "x:" prefix: search for the whole text
            terms.push({ field: null, value: (field ? `${field}:` : '') + value, negate: m[1] === '-' });
        }
    }
    return terms;
}

const has = (list: string[], value: string) => list.some((v) => v.toLowerCase().includes(value));

function matchTerm(node: QueryNode, term: Term): boolean {
    switch (term.field) {
        case 'tag':
            return has(node.tags, term.value);
        case 'person':
            return has(node.people, term.value);
        case 'org':
            return has(node.organisations, term.value);
        case 'source':
            return node.source.toLowerCase().includes(term.value);
        case 'type':
            return node.sourceType.toLowerCase() === term.value;
        case 'community':
            return String(node.community) === term.value;
        case 'year': {
            const year = Number(node.date?.slice(0, 4));
            const [from, to] = term.value.split('-').map(Number);
            return Number.isFinite(year) && year >= from && year <= (Number.isFinite(to) ? to : from);
        }
        default:
            return (
                node.label.toLowerCase().includes(term.value) ||
                has(node.tags, term.value) ||
                has(node.people, term.value) ||
                has(node.organisations, term.value)
            );
    }
}

/** A predicate for the query; null for an empty query (matches everything). */
export function compileQuery(query: string): ((node: QueryNode) => boolean) | null {
    const terms = parseQuery(query);
    if (terms.length === 0) return null;
    return (node) => terms.every((t) => matchTerm(node, t) !== t.negate);
}

/** Query term for a facet value, quoted when it has spaces */
export function facetTerm(field: QueryField, value: string): string {
    return /\s/.test(value) ? `${field}:"${value}"` : `${field}:${value}`;
}
