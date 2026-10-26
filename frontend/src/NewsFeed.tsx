import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, ArrowUpRight, CheckCircle2, Clock, Newspaper, Search, SlidersHorizontal } from "lucide-react";

/** Provider-neutral metadata. Content is full article text, never a truncated API preview. */
export interface FeedStory {
  id: string;
  title: string;
  source: string;
  section: string;
  description: string;
  content?: string;
  url?: string;
  imageUrl?: string;
  publishedAt?: string;
  collection?: string;
}

interface Props {
  stories: FeedStory[];
  selectedId: string | null;
  onOpen: (story: FeedStory) => void;
  collectionLabel?: string;
  compact?: boolean;
}

function externalUrl(value?: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch { return undefined; }
}

function StoryImage({ url }: { url: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return <img src={url} alt="" loading="lazy" onError={() => setFailed(true)}
    className="newsfeed-story-image aspect-video w-full object-cover rounded-t-xl" />;
}

function StoryCard({ story, selected, onOpen, compact }: {
  story: FeedStory; selected: boolean; onOpen: Props["onOpen"]; compact?: boolean;
}) {
  const sourceUrl = externalUrl(story.url);
  const imageUrl = externalUrl(story.imageUrl);
  const date = story.publishedAt ? new Date(story.publishedAt) : null;
  const validDate = date && !Number.isNaN(date.getTime());
  const minutes = story.content?.trim()
    ? Math.max(1, Math.ceil(story.content.trim().split(/\s+/).length / 220)) : null;
  return (
    <article aria-labelledby={`feed-headline-${story.id}`}
      className={`newsfeed-story-card flex flex-col rounded-xl border bg-white dark:bg-slate-900 transition-colors ${
        selected ? "border-indigo-400 dark:border-indigo-500 ring-1 ring-indigo-200 dark:ring-indigo-900"
          : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-600"
      }`}>
      {imageUrl && <StoryImage key={imageUrl} url={imageUrl} />}
      <div className={`newsfeed-story-body flex flex-col flex-1 min-w-0 ${compact ? "gap-2 p-3" : "gap-3 p-5"}`}>
        <div className="newsfeed-topic flex items-start justify-between gap-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">{story.section}</p>
          {selected && <span className="inline-flex shrink-0 items-center gap-1 text-[10px] font-medium text-indigo-600 dark:text-indigo-400">
            <CheckCircle2 className="h-3 w-3" />In editor
          </span>}
        </div>
        <h3 id={`feed-headline-${story.id}`} className={`newsfeed-headline font-serif font-semibold leading-snug tracking-tight text-slate-900 dark:text-slate-100 ${compact ? "text-sm line-clamp-2" : "text-xl"}`}>
          <button type="button" title={story.title} onClick={() => onOpen(story)} className="text-left hover:text-indigo-700 dark:hover:text-indigo-300 transition-colors cursor-pointer">{story.title}</button>
        </h3>
        <div className={`newsfeed-meta flex items-center gap-x-2 gap-y-1 text-slate-500 dark:text-slate-400 ${compact ? "min-w-0 text-[11px]" : "flex-wrap text-xs"}`}>
          <span title={story.source} className="min-w-0 truncate font-medium text-slate-700 dark:text-slate-300">{story.source}</span>
          {validDate && <><span aria-hidden="true">·</span><time dateTime={story.publishedAt}>
            {date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
          </time></>}
          {compact && minutes && <span className="ml-auto shrink-0 inline-flex items-center gap-1 text-[10px]"><Clock className="h-3 w-3" />{minutes} min</span>}
        </div>
        <p className={`newsfeed-description leading-relaxed text-slate-600 dark:text-slate-400 ${compact ? "text-xs line-clamp-2" : "text-sm line-clamp-3"}`}>{story.description.replace(/\s+/g, " ").trim()}</p>
        {!compact && <div className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 text-[10px] text-slate-400 dark:text-slate-500">
          {story.collection && <span>{story.collection}</span>}
          {minutes && <span className="inline-flex items-center gap-1"><Clock className="h-3 w-3" />{minutes} min read</span>}
        </div>}
        <div className={`newsfeed-actions mt-auto flex items-center justify-between border-t border-slate-100 dark:border-slate-800 ${compact ? "pt-2" : "pt-3"}`}>
          <button type="button" id={`sample-btn-${story.id}`} onClick={() => onOpen(story)} aria-label={`Open article: ${story.title}`}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 cursor-pointer">
            Open article<ArrowRight className="h-3.5 w-3.5" />
          </button>
          {sourceUrl && <a href={sourceUrl} target="_blank" rel="noopener noreferrer" aria-label={`View source: ${story.title}`}
            className="inline-flex items-center gap-1 text-xs text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
            Source<ArrowUpRight className="h-3 w-3" />
          </a>}
        </div>
      </div>
    </article>
  );
}

export default function NewsFeed({ stories, selectedId, onOpen, collectionLabel = "Newsfeed", compact = false }: Props) {
  const [query, setQuery] = useState("");
  const [section, setSection] = useState("");
  const [source, setSource] = useState("");
  const [limit, setLimit] = useState(6);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const filterButtonRef = useRef<HTMLButtonElement>(null);
  const sections = useMemo(() => [...new Set(stories.map((story) => story.section))].sort(), [stories]);
  const sources = useMemo(() => [...new Set(stories.map((story) => story.source))].sort(), [stories]);
  const filtered = useMemo(() => {
    const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return stories.filter((story) => {
      const searchable = [story.title, story.source, story.section, story.description, story.content, story.collection]
        .filter(Boolean).join(" ").toLowerCase();
      return (!section || story.section === section) && (!source || story.source === source)
        && words.every((word) => searchable.includes(word));
    });
  }, [stories, query, section, source]);
  useEffect(() => {
    const list = listRef.current;
    if (!compact || !list) return;
    const wheel = (event: WheelEvent) => {
      if (list.scrollWidth > list.clientWidth && Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
        event.preventDefault();
        list.scrollLeft += event.deltaY;
      }
    };
    list.addEventListener("wheel", wheel, { passive: false });
    return () => list.removeEventListener("wheel", wheel);
  }, [compact, filtered.length]);
  useEffect(() => {
    if (compact && listRef.current) {
      listRef.current.scrollLeft = 0;
      listRef.current.scrollTop = 0;
    }
  }, [compact, query, section, source]);
  const reset = () => { setQuery(""); setSection(""); setSource(""); setLimit(6); };
  return (
    <section aria-labelledby="newsfeed-heading" className={compact ? "compact-newsfeed" : "flex flex-col gap-5"}>
      <div className={compact ? "flex shrink-0 items-center gap-2" : "flex flex-wrap items-end justify-between gap-4"}>
        {compact ? <h2 id="newsfeed-heading" className="flex shrink-0 items-center gap-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300">
          <Newspaper className="h-3.5 w-3.5" />{collectionLabel}
        </h2> : <div className="flex flex-col gap-2">
          <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest font-semibold text-indigo-600 dark:text-indigo-400">
            <Newspaper className="h-3.5 w-3.5" />{collectionLabel}
          </span>
          <h2 id="newsfeed-heading" className="font-serif text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">Explore the stories</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">Open a story in the editor to read, review, and analyze how it is framed.</p>
        </div>}
        <div className={compact ? "relative flex-1 min-w-0" : "relative w-full sm:w-72"}>
          <Search className={`pointer-events-none absolute left-2 text-slate-400 ${compact ? "top-2 h-3.5 w-3.5" : "top-3 h-4 w-4"}`} />
          <input type="search" aria-label="Search stories" value={query} placeholder="Search headlines, sources, keywords..."
            onChange={(event) => { setQuery(event.target.value); setLimit(6); }}
            className={`w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 pl-7 pr-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 ${compact ? "py-1.5" : "py-2.5"}`} />
        </div>
        {compact && <button ref={filterButtonRef} type="button" aria-label="Feed filters" aria-expanded={filtersOpen}
          aria-controls="newsfeed-filters" onClick={() => setFiltersOpen((open) => !open)}
          className={`shrink-0 rounded-lg border p-1.5 ${section || source ? "border-indigo-400 text-indigo-600 dark:text-indigo-400" : "border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400"}`}>
          <SlidersHorizontal className="h-4 w-4" />
        </button>}
      </div>
      {compact ? filtersOpen && <div id="newsfeed-filters" role="group" aria-label="Feed filter options"
        onKeyDown={(event) => { if (event.key === "Escape") { setFiltersOpen(false); filterButtonRef.current?.focus(); } }}
        className="absolute top-10 left-0 right-0 z-20 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-3 shadow-lg flex flex-col gap-3 text-xs">
        <label className="flex flex-col gap-1">Section<select aria-label="Filter by section" value={section}
          onChange={(event) => { setSection(event.target.value); setLimit(6); }}
          className="rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1.5">
          <option value="">All stories</option>{sections.map((name) => <option key={name}>{name}</option>)}
        </select></label>
        <label className="flex flex-col gap-1">Source<select aria-label="Filter by source" value={source}
          onChange={(event) => { setSource(event.target.value); setLimit(6); }}
          className="rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-1.5">
          <option value="">All sources</option>{sources.map((name) => <option key={name}>{name}</option>)}
        </select></label>
        <button type="button" onClick={reset} className="self-start font-semibold text-indigo-600 dark:text-indigo-400">Reset filters</button>
      </div> : <div className="flex flex-wrap items-center justify-between gap-3 border-y border-slate-200 dark:border-slate-800 py-3">
        <div role="group" aria-label="Filter by section" className="flex flex-wrap items-center gap-1">
          {["", ...sections].map((option) => <button key={option || "all"} type="button" aria-pressed={section === option}
            onClick={() => { setSection(option); setLimit(6); }}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
              section === option ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
            }`}>{option || "All stories"}</button>)}
        </div>
        <select aria-label="Filter by source" value={source} onChange={(event) => { setSource(event.target.value); setLimit(6); }}
          className="max-w-full sm:max-w-[240px] rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-xs text-slate-600 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500">
          <option value="">All sources</option>
          {sources.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </div>}
      <p aria-live="polite" className="shrink-0 text-[11px] text-slate-400 dark:text-slate-500">
        {compact ? `${filtered.length} ${filtered.length === 1 ? "story" : "stories"}`
          : `Showing ${Math.min(limit, filtered.length)} of ${filtered.length} ${filtered.length === 1 ? "story" : "stories"}`}
      </p>
      {filtered.length ? (
        <div ref={listRef} role={compact ? "region" : undefined} aria-label={compact ? "Scrollable news stories" : undefined}
          tabIndex={compact ? 0 : undefined} className={compact ? "newsfeed-story-list" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"}>
          {(compact ? filtered : filtered.slice(0, limit)).map((story) => <StoryCard key={story.id} story={story}
            selected={selectedId === story.id} onOpen={onOpen} compact={compact} />)}
        </div>
      ) : (
        <div className={`rounded-xl border border-dashed border-slate-300 dark:border-slate-700 text-center ${compact ? "flex-1 min-h-0 overflow-auto py-3" : "py-10"}`}>
          <p className="text-sm text-slate-500 dark:text-slate-400">No stories match your filters.</p>
          <button type="button" onClick={reset} className="mt-3 text-xs font-semibold text-indigo-600 dark:text-indigo-400 cursor-pointer">Clear filters</button>
        </div>
      )}
      {!compact && filtered.length > limit && <button type="button" onClick={() => setLimit((count) => count + 6)}
        className="self-center rounded-lg border border-slate-200 dark:border-slate-700 px-5 py-2.5 text-xs font-semibold text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer">Show more stories</button>}
    </section>
  );
}
