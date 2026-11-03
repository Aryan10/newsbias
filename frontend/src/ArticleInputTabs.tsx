import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject, ReactNode } from "react";
import { AlertTriangle, RefreshCw, Sparkles } from "lucide-react";

export interface ImportedArticle {
  title: string;
  text: string;
  source: {
    type: "raw" | "url" | "upload";
    url: string | null;
    filename: string | null;
    author: string | null;
    publication_date: string | null;
  };
  extraction: { method: string; notices: string[] };
}

type InputMode = "article" | "raw" | "url" | "upload";
const MODES: InputMode[] = ["article", "raw", "url", "upload"];
const LABELS = { article: "Article", raw: "Raw", url: "URL", upload: "Upload" };
const FIELD_CLASS = "w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-3.5 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500";

interface Props {
  children: ReactNode;
  articleRevision: MutableRefObject<number>;
  selectedPresetRevision: number;
  onImport: (article: ImportedArticle) => void;
}

export default function ArticleInputTabs({
  children, articleRevision, selectedPresetRevision, onImport,
}: Props) {
  const [mode, setMode] = useState<InputMode>("article");
  const [raw, setRaw] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRevision = useRef(0);
  const controller = useRef<AbortController | null>(null);

  const cancelExtraction = useCallback(() => {
    requestRevision.current += 1;
    controller.current?.abort();
    controller.current = null;
    setIsExtracting(false);
    setError(null);
  }, []);

  useEffect(() => {
    cancelExtraction();
    setMode("article");
  }, [selectedPresetRevision, cancelExtraction]);

  useEffect(() => () => controller.current?.abort(), []);

  const changeMode = (next: InputMode) => {
    if (next === mode) return;
    cancelExtraction();
    setMode(next);
  };

  const extract = async () => {
    if (mode === "article") return;
    const requestId = ++requestRevision.current;
    const currentArticle = articleRevision.current;
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setIsExtracting(true);
    setError(null);
    try {
      let body: BodyInit;
      let headers: HeadersInit | undefined;
      let endpoint: string;
      if (mode === "upload") {
        if (!file) throw new Error("Choose a text file first.");
        if (!file.name.toLowerCase().endsWith(".txt")) {
          throw new Error("Upload currently supports .txt files. More formats will be added.");
        }
        if (file.size > 1_000_000) throw new Error("Uploads must be 1 MB or smaller.");
        const form = new FormData();
        form.append("file", file);
        body = form;
        endpoint = "file";
      } else {
        if (mode === "raw" && raw.length > 120_000) {
          throw new Error("Raw input must be 120,000 characters or fewer. Paste an article-sized excerpt.");
        }
        body = JSON.stringify(mode === "raw" ? { text: raw } : { url: url.trim() });
        headers = { "Content-Type": "application/json" };
        endpoint = mode;
      }
      const response = await fetch(`/bias/import/${endpoint}`, {
        method: "POST", headers, body, signal: abort.signal,
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof result.detail === "string"
          ? result.detail : "Could not import this article. Check the input and try again.");
      }
      if (requestId !== requestRevision.current || currentArticle !== articleRevision.current) return;
      if (typeof result.title !== "string" || typeof result.text !== "string" || !result.text.trim()
        || !result.source || !Array.isArray(result.extraction?.notices)) {
        throw new Error("The extractor returned an incomplete article. Try again.");
      }
      onImport(result as ImportedArticle);
      setMode("article");
    } catch (failure) {
      if (requestId === requestRevision.current && currentArticle === articleRevision.current
        && !abort.signal.aborted) {
        setError(failure instanceof Error ? failure.message : "Article extraction failed.");
      }
    } finally {
      if (requestId === requestRevision.current) {
        setIsExtracting(false);
        controller.current = null;
      }
    }
  };

  const canExtract = mode === "raw" ? Boolean(raw.trim())
    : mode === "url" ? Boolean(url.trim()) : Boolean(file);

  return (
    <>
      <div role="tablist" aria-label="Article input options"
        className="flex gap-1 border-b border-slate-200 dark:border-slate-800 pb-3">
        {MODES.map((option, index) => (
          <button key={option} type="button" role="tab" id={`input-tab-${option}`}
            aria-selected={mode === option} aria-controls={`input-panel-${option}`}
            tabIndex={mode === option ? 0 : -1}
            onClick={() => changeMode(option)}
            onKeyDown={(event) => {
              let next = index;
              if (event.key === "ArrowRight") next = (index + 1) % MODES.length;
              else if (event.key === "ArrowLeft") next = (index + MODES.length - 1) % MODES.length;
              else if (event.key === "Home") next = 0;
              else if (event.key === "End") next = MODES.length - 1;
              else return;
              event.preventDefault();
              changeMode(MODES[next]);
              document.getElementById(`input-tab-${MODES[next]}`)?.focus();
            }}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
              mode === option
                ? "bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300"
                : "text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
            }`}>
            {LABELS[option]}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`input-panel-${mode}`} aria-labelledby={`input-tab-${mode}`}
        aria-busy={isExtracting} className="flex flex-col gap-4">
        {mode === "article" ? children : (
          <>
            {mode === "raw" && (
              <div className="import-text-input flex flex-col gap-2">
                <label htmlFor="raw-page-input" className="text-sm font-semibold">Copied website text</label>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Copy a news page with Ctrl+A / Ctrl+C and paste it here. The extraction agent
                  separates the article from navigation, ads, and related stories.
                </p>
                <textarea id="raw-page-input" rows={12} value={raw}
                  placeholder="Paste the full copied news page here..."
                  onChange={(event) => { cancelExtraction(); setRaw(event.target.value); }}
                  className={`${FIELD_CLASS} font-serif leading-relaxed min-h-[220px]`} />
              </div>
            )}
            {mode === "url" && (
              <div className="flex flex-col gap-2">
                <label htmlFor="news-url-input" className="text-sm font-semibold">News article URL</label>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Import a public HTTPS news page. If the site cannot be read, use its copied text in Raw.
                </p>
                <input id="news-url-input" type="url" value={url}
                  placeholder="https://example.com/news/article"
                  onChange={(event) => { cancelExtraction(); setUrl(event.target.value); }}
                  className={FIELD_CLASS} />
              </div>
            )}
            {mode === "upload" && (
              <div className="flex flex-col gap-2">
                <label htmlFor="article-file-input" className="text-sm font-semibold">Upload a text document</label>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Currently supports .txt files (UTF-8 or UTF-16), up to 1 MB.
                  More document and image formats will be added later.
                </p>
                <input id="article-file-input" type="file" accept=".txt,text/plain"
                  onChange={(event) => { cancelExtraction(); setFile(event.target.files?.[0] ?? null); }}
                  className={`${FIELD_CLASS} file:mr-4 file:rounded-lg file:border-0 file:px-3 file:py-2 file:cursor-pointer`} />
                {file && <p className="text-xs text-slate-500 dark:text-slate-400">Selected: {file.name}</p>}
              </div>
            )}
            <div className="import-actions sticky bottom-0 z-10 bg-white dark:bg-slate-900 flex flex-wrap items-center justify-between gap-3">
              <button type="button" onClick={extract} disabled={!canExtract || isExtracting}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ml-auto">
                {isExtracting ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {isExtracting ? "Extracting Article..." : "Extract Article"}
              </button>
            </div>
            {error && <div role="alert" className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
              <AlertTriangle className="h-4 w-4 shrink-0" />{error}
            </div>}
          </>
        )}
      </div>
    </>
  );
}
