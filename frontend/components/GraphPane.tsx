'use client';

import { useMemo, useState } from 'react';
import { facetTerm, type QueryField } from '@/lib/graphQuery';

export type ColorBy = 'community' | 'source' | 'year' | 'type';

export interface ColorGroup {
    query: string;
    color: string;
}

export interface GraphSettings {
    filter: {
        query: string;
        hiddenTypes: string[];
        kinds: { similar: boolean; duplicate: boolean; link: boolean };
        minWeight: number;
        orphans: boolean;
        yearFrom: number | null;
        yearTo: number | null;
        hiddenCommunities: number[];
        local: boolean; // only the selected note's neighbourhood
        depth: number;
    };
    groups: {
        colorBy: ColorBy;
        custom: ColorGroup[];
        communityColors: Record<number, string>;
    };
    display: {
        nodeSize: number;
        linkThickness: number;
        textFade: number; // labels shown from this rendered node size
        communityLabels: boolean;
        contours: boolean; // density contours per colour group, as in Embedding Atlas
        contourBandwidth: number; // their smoothing, in layout units
    };
    forces: {
        center: number; // pull towards the middle
        repel: number; // every note pushes the others away
        link: number; // links pull their notes together
        distance: number; // length a link settles at
        communityPull: number; // notes pulled towards their community's centre; links across weaken
    };
}

export const DEFAULT_SETTINGS: GraphSettings = {
    filter: {
        query: '',
        hiddenTypes: [],
        kinds: { similar: true, duplicate: true, link: true },
        minWeight: 0,
        orphans: true,
        yearFrom: null,
        yearTo: null,
        hiddenCommunities: [],
        local: false,
        depth: 1,
    },
    groups: { colorBy: 'community', custom: [], communityColors: {} },
    display: { nodeSize: 1, linkThickness: 1, textFade: 14, communityLabels: true, contours: false, contourBandwidth: 20 },
    // tuned on the KoopCast graph: no overlaps, every community clear of its neighbours
    forces: { center: 0.04, repel: 120, link: 1, distance: 40, communityPull: 0.3 },
};

/** Saved settings merged over the defaults, so settings added later get their default */
export function mergeSettings(saved: Partial<GraphSettings> | null): GraphSettings {
    if (!saved) return DEFAULT_SETTINGS;
    return {
        filter: { ...DEFAULT_SETTINGS.filter, ...saved.filter,
                  kinds: { ...DEFAULT_SETTINGS.filter.kinds, ...saved.filter?.kinds } },
        groups: { ...DEFAULT_SETTINGS.groups, ...saved.groups },
        display: { ...DEFAULT_SETTINGS.display, ...saved.display },
        forces: { ...DEFAULT_SETTINGS.forces, ...saved.forces },
    };
}

export interface CommunityInsight {
    id: number;
    label: string;
    size: number;
    color: string;
    topics: [string, number][];
    people: [string, number][];
    organisations: [string, number][];
    from: string | null;
    to: string | null;
    hours: number;
}

export interface Facets {
    tag: string[];
    person: string[];
    org: string[];
    source: string[];
}

interface Props {
    open: boolean;
    onClose: () => void;
    settings: GraphSettings;
    onChange: (next: GraphSettings) => void;
    types: { type: string; label: string; count: number }[];
    years: number[];
    communities: CommunityInsight[];
    facets: Facets;
    visible: { nodes: number; links: number };
    hasSelection: boolean;
    focusCommunity: number | null;
    onFocusCommunity: (id: number | null) => void;
    onAnimate: () => void;
    legend: React.ReactNode;
}

// Offered for new colour groups, in order (the community palette's first eight)
const GROUP_COLORS = ['#4e9af1', '#f0a33a', '#e0607e', '#3fb950', '#a78bfa', '#2dd4bf', '#e3c84b', '#ef6b5b'];

function Slider({ label, value, min, max, step, onChange, format }: {
    label: string; value: number; min: number; max: number; step: number;
    onChange: (v: number) => void; format?: (v: number) => string;
}) {
    return (
        <label className="gp-slider">
            <span className="gp-slider-row">
                <span>{label}</span>
                <span className="text-muted">{format ? format(value) : value}</span>
            </span>
            <input type="range" min={min} max={max} step={step} value={value}
                   onChange={(e) => onChange(Number(e.target.value))} />
        </label>
    );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
    return (
        <label className="gp-toggle">
            <span>{label}</span>
            <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        </label>
    );
}

/** Facet values matching the last word typed, offered as query terms */
function suggestions(query: string, facets: Facets): { term: string; field: QueryField; value: string }[] {
    const last = query.match(/(-?)(?:(\w+):)?"?([^"\s]*)$/);
    const text = (last?.[3] ?? '').toLowerCase();
    if (text.length < 2) return [];
    const wanted = last?.[2]?.toLowerCase();
    const out: { term: string; field: QueryField; value: string }[] = [];
    for (const field of ['person', 'tag', 'org', 'source'] as const) {
        if (wanted && wanted !== field) continue;
        for (const value of facets[field]) {
            if (value.toLowerCase().includes(text)) out.push({ term: facetTerm(field, value), field, value });
            if (out.length >= 8) return out;
        }
    }
    return out;
}

function ChipList({ items, field, onPick }: {
    items: [string, number][]; field: QueryField; onPick: (term: string) => void;
}) {
    if (items.length === 0) return <span className="text-muted">none stand out</span>;
    return (
        <span className="gp-chips">
            {items.map(([value, n]) => (
                <button key={value} className="gp-chip" onClick={() => onPick(facetTerm(field, value))}
                        title={`Filter to ${value}`}>
                    {value} <span className="text-muted">{n}</span>
                </button>
            ))}
        </span>
    );
}

export default function GraphPane(props: Props) {
    const { settings, onChange } = props;
    const { filter, groups, display, forces } = settings;
    const [expanded, setExpanded] = useState<number | null>(null);
    const [newGroup, setNewGroup] = useState('');
    const offered = useMemo(() => suggestions(filter.query, props.facets), [filter.query, props.facets]);

    const set = <K extends keyof GraphSettings>(key: K, value: Partial<GraphSettings[K]>) =>
        onChange({ ...settings, [key]: { ...settings[key], ...value } });

    const addTerm = (term: string) => {
        const base = filter.query.replace(/(-?)(?:(\w+):)?"?([^"\s]*)$/, '').trim();
        set('filter', { query: `${base} ${term}`.trim() });
    };
    const pickChip = (term: string) => {
        set('filter', { query: filter.query.includes(term) ? filter.query : `${filter.query} ${term}`.trim() });
    };
    const toggleCommunity = (id: number) => {
        const hidden = filter.hiddenCommunities.includes(id)
            ? filter.hiddenCommunities.filter((c) => c !== id)
            : [...filter.hiddenCommunities, id];
        set('filter', { hiddenCommunities: hidden });
    };

    return (
        <aside className={`graph-pane ${props.open ? 'open' : ''}`} aria-hidden={!props.open}>
            <div className="gp-header">
                <span className="gp-count">
                    {props.visible.nodes} notes · {props.visible.links} links
                </span>
                <button className="gp-icon" onClick={() => onChange(DEFAULT_SETTINGS)} title="Restore default settings">
                    ↺
                </button>
                <button className="gp-icon" onClick={props.onClose} title="Close">✕</button>
            </div>

            <details open className="gp-section">
                <summary>Filters</summary>
                <div className="gp-query">
                    <input
                        className="input"
                        value={filter.query}
                        onChange={(e) => set('filter', { query: e.target.value })}
                        placeholder='Search: heat  person:"Stephanie Howe"  -tag:nutrition'
                        spellCheck={false}
                    />
                    {offered.length > 0 && (
                        <ul className="gp-suggest">
                            {offered.map((s) => (
                                <li key={s.term}>
                                    <button onClick={() => addTerm(s.term)}>
                                        <span className="gp-suggest-field">{s.field}</span>{s.value}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
                <div className="gp-hint">Fields: tag: person: org: source: type: community: year:2021-2023 · “-” excludes</div>
                <div className="gp-checks">
                    {props.types.map((t) => (
                        <label key={t.type} className="gp-check">
                            <input
                                type="checkbox"
                                checked={!filter.hiddenTypes.includes(t.type)}
                                onChange={(e) => set('filter', {
                                    hiddenTypes: e.target.checked
                                        ? filter.hiddenTypes.filter((x) => x !== t.type)
                                        : [...filter.hiddenTypes, t.type],
                                })}
                            />
                            {t.label} <span className="text-muted">{t.count}</span>
                        </label>
                    ))}
                </div>
                {props.years.length > 1 && (
                    <div className="gp-years">
                        <span>Years</span>
                        <select className="input" value={filter.yearFrom ?? ''}
                                onChange={(e) => set('filter', { yearFrom: e.target.value ? Number(e.target.value) : null })}>
                            <option value="">from start</option>
                            {props.years.map((y) => <option key={y} value={y}>{y}</option>)}
                        </select>
                        <select className="input" value={filter.yearTo ?? ''}
                                onChange={(e) => set('filter', { yearTo: e.target.value ? Number(e.target.value) : null })}>
                            <option value="">to now</option>
                            {props.years.map((y) => <option key={y} value={y}>{y}</option>)}
                        </select>
                    </div>
                )}
                <div className="gp-checks">
                    {(['similar', 'duplicate', 'link'] as const).map((k) => (
                        <label key={k} className="gp-check">
                            <input type="checkbox" checked={filter.kinds[k]}
                                   onChange={(e) => set('filter', { kinds: { ...filter.kinds, [k]: e.target.checked } })} />
                            {k === 'link' ? 'wikilinks' : k === 'similar' ? 'similar' : 'duplicates'}
                        </label>
                    ))}
                </div>
                <Slider label="Minimum link strength" value={filter.minWeight} min={0} max={0.6} step={0.01}
                        onChange={(v) => set('filter', { minWeight: v })} format={(v) => v.toFixed(2)} />
                <Toggle label="Notes without links" checked={filter.orphans} onChange={(v) => set('filter', { orphans: v })} />
                <Toggle label="Only the selected note's neighbourhood" checked={filter.local}
                        onChange={(v) => set('filter', { local: v })} />
                {filter.local && (
                    <Slider label="Depth" value={filter.depth} min={1} max={4} step={1}
                            onChange={(v) => set('filter', { depth: v })} />
                )}
                {filter.local && !props.hasSelection && <div className="gp-hint">Click a note to centre on it.</div>}
            </details>

            <details open className="gp-section">
                <summary>Communities</summary>
                {Object.keys(groups.communityColors).length > 0 && (
                    <div className="gp-row-buttons">
                        <button className="btn btn-secondary" onClick={() => set('groups', { communityColors: {} })}
                                title="Drop the colours picked here: back to the automatic ones">
                            Reset colours
                        </button>
                    </div>
                )}
                <ul className="gp-communities">
                    {props.communities.map((c) => {
                        const hidden = filter.hiddenCommunities.includes(c.id);
                        return (
                            <li key={c.id} className={hidden ? 'hidden' : ''}>
                                <div className="gp-community-row">
                                    <input
                                        type="color"
                                        className="gp-swatch"
                                        value={c.color}
                                        title="Colour"
                                        onChange={(e) => set('groups', {
                                            communityColors: { ...groups.communityColors, [c.id]: e.target.value },
                                        })}
                                    />
                                    <button className="gp-community-name" aria-expanded={expanded === c.id}
                                            onClick={() => setExpanded(expanded === c.id ? null : c.id)}>
                                        {c.label || `Community ${c.id}`}
                                    </button>
                                    <span className="text-muted">{c.size}</span>
                                    <button className="gp-icon" aria-pressed={props.focusCommunity === c.id}
                                            title="Highlight" onClick={() => props.onFocusCommunity(props.focusCommunity === c.id ? null : c.id)}>
                                        ◎
                                    </button>
                                    <button className="gp-icon" title={hidden ? 'Show' : 'Hide'} onClick={() => toggleCommunity(c.id)}>
                                        {hidden ? '○' : '●'}
                                    </button>
                                </div>
                                {expanded === c.id && (
                                    <div className="gp-community-detail">
                                        <div className="text-muted">
                                            {c.from && c.to ? `${c.from.slice(0, 7)} – ${c.to.slice(0, 7)}` : ''}
                                            {c.hours ? ` · ${c.hours.toFixed(0)} h of audio` : ''}
                                        </div>
                                        <div><span className="gp-label">Topics</span><ChipList items={c.topics} field="tag" onPick={pickChip} /></div>
                                        <div><span className="gp-label">People</span><ChipList items={c.people} field="person" onPick={pickChip} /></div>
                                        <div><span className="gp-label">Organisations</span><ChipList items={c.organisations} field="org" onPick={pickChip} /></div>
                                        <div className="gp-row-buttons">
                                            <button className="btn btn-secondary" onClick={() => set('filter', {
                                                hiddenCommunities: props.communities.filter((o) => o.id !== c.id).map((o) => o.id),
                                            })}>Only this</button>
                                            <button className="btn btn-secondary" onClick={() => set('filter', { hiddenCommunities: [] })}>
                                                Show all
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </li>
                        );
                    })}
                </ul>
            </details>

            <details open className="gp-section">
                <summary>Groups</summary>
                <label className="gp-select">
                    <span>Colour by</span>
                    <select className="input" value={groups.colorBy}
                            onChange={(e) => set('groups', { colorBy: e.target.value as ColorBy })}>
                        <option value="community">community</option>
                        <option value="year">year</option>
                        <option value="source">source</option>
                        <option value="type">type</option>
                    </select>
                </label>
                <div className="gp-hint">Groups colour matching notes over that; the first match wins.</div>
                <ul className="gp-groups">
                    {groups.custom.map((g, i) => (
                        <li key={i}>
                            <input type="color" className="gp-swatch" value={g.color}
                                   onChange={(e) => set('groups', {
                                       custom: groups.custom.map((x, j) => (j === i ? { ...x, color: e.target.value } : x)),
                                   })} />
                            <input className="input" value={g.query}
                                   onChange={(e) => set('groups', {
                                       custom: groups.custom.map((x, j) => (j === i ? { ...x, query: e.target.value } : x)),
                                   })} />
                            <button className="gp-icon" title="Remove"
                                    onClick={() => set('groups', { custom: groups.custom.filter((_, j) => j !== i) })}>✕</button>
                        </li>
                    ))}
                </ul>
                <form className="gp-new-group" onSubmit={(e) => {
                    e.preventDefault();
                    if (!newGroup.trim()) return;
                    const color = GROUP_COLORS[groups.custom.length % GROUP_COLORS.length];
                    set('groups', { custom: [...groups.custom, { query: newGroup.trim(), color }] });
                    setNewGroup('');
                }}>
                    <input className="input" value={newGroup} onChange={(e) => setNewGroup(e.target.value)}
                           placeholder='New group: person:"Nick Tiller"' />
                    <button className="btn btn-secondary" type="submit">Add</button>
                </form>
            </details>

            <details className="gp-section">
                <summary>Display</summary>
                <Slider label="Node size" value={display.nodeSize} min={0.3} max={3} step={0.1}
                        onChange={(v) => set('display', { nodeSize: v })} format={(v) => `${v.toFixed(1)}×`} />
                <Slider label="Link thickness" value={display.linkThickness} min={0} max={4} step={0.1}
                        onChange={(v) => set('display', { linkThickness: v })} format={(v) => `${v.toFixed(1)}×`} />
                <Slider label="Titles from node size" value={display.textFade} min={0} max={40} step={1}
                        onChange={(v) => set('display', { textFade: v })} format={(v) => (v === 0 ? 'all' : `${v}px`)} />
                <Toggle label="Community names" checked={display.communityLabels}
                        onChange={(v) => set('display', { communityLabels: v })} />
                <Toggle label="Density contours" checked={display.contours}
                        onChange={(v) => set('display', { contours: v })} />
                {display.contours && (
                    <Slider label="Contour smoothing" value={display.contourBandwidth} min={5} max={60} step={1}
                            onChange={(v) => set('display', { contourBandwidth: v })} />
                )}
                {props.legend}
            </details>

            <details className="gp-section">
                <summary>Forces</summary>
                <Slider label="Centre force" value={forces.center} min={0} max={0.3} step={0.005}
                        onChange={(v) => set('forces', { center: v })} format={(v) => v.toFixed(3)} />
                <Slider label="Repel force" value={forces.repel} min={0} max={400} step={5}
                        onChange={(v) => set('forces', { repel: v })} />
                <Slider label="Link force" value={forces.link} min={0} max={2} step={0.05}
                        onChange={(v) => set('forces', { link: v })} format={(v) => v.toFixed(2)} />
                <Slider label="Link distance" value={forces.distance} min={10} max={200} step={5}
                        onChange={(v) => set('forces', { distance: v })} />
                <Slider label="Community pull" value={forces.communityPull} min={0} max={1} step={0.05}
                        onChange={(v) => set('forces', { communityPull: v })} format={(v) => v.toFixed(2)} />
                <div className="gp-row-buttons">
                    <button className="btn btn-secondary" onClick={props.onAnimate} title="Lay the graph out again from scratch">
                        Animate
                    </button>
                </div>
            </details>
        </aside>
    );
}
