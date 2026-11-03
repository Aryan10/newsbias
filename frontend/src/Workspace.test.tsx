import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";
import NewsFeed from "./NewsFeed";
import type { FeedStory } from "./NewsFeed";

const analysis = {
  input: { title: "", text: "The council announced a vote." }, sentences: [],
  specialist_sentence: { status: "skipped", label: "NOT EVALUATED", bias_ratio: null,
    biased_sentence_count: null, total_sentences: 1, flagged_sentences: [] },
  specialist_political: { status: "skipped", predicted_label: "NOT EVALUATED", confidence: null, scores: {} },
  isolated_llm: { political_leaning: { label: "center" }, linguistic_bias: { is_biased: false } },
  arbiter: { final_verdict: { is_biased: false, overall_leaning: "center" } },
};

const originalScroll = Object.getOwnPropertyDescriptor(Element.prototype, "scrollIntoView");
afterEach(() => {
  cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks();
  if (originalScroll) Object.defineProperty(Element.prototype, "scrollIntoView", originalScroll);
  else delete Element.prototype.scrollIntoView;
});

it("uses the compact feed with all stories in its independently scrollable region", () => {
  const stories: FeedStory[] = Array.from({ length: 10 }, (_, index) => ({
    id: `${index}`, title: `Story ${index}`, source: "News Wire", section: "World", description: "Excerpt.",
  }));
  const { container } = render(<NewsFeed compact stories={stories} selectedId={null} onOpen={vi.fn()} />);
  expect(container.querySelector(".compact-newsfeed")).toBeTruthy();
  expect(screen.getAllByRole("article")).toHaveLength(10);
  expect(screen.getByRole("region", { name: "Scrollable news stories" }).getAttribute("tabindex")).toBe("0");
  expect(screen.queryByRole("button", { name: "Show more stories" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Feed filters" }));
  expect(screen.getByRole("combobox", { name: "Filter by section" })).toBeTruthy();
  fireEvent.keyDown(screen.getByRole("combobox", { name: "Filter by section" }), { key: "Escape" });
  expect(screen.queryByRole("combobox", { name: "Filter by section" })).toBeNull();
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "Feed filters" }));
});

it("focuses and scrolls to results only after the accepted analysis completes", async () => {
  const scrolled = vi.fn();
  Object.defineProperty(Element.prototype, "scrollIntoView", { configurable: true, value: scrolled });
  let finish: (result: unknown) => void;
  vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => url === "/bias/samples"
    ? Promise.resolve({ ok: true, json: async () => [] })
    : new Promise((resolve) => { finish = resolve; })));
  render(<App />);
  fireEvent.change(screen.getByPlaceholderText("Paste article text here..."), {
    target: { value: "The council announced a vote." },
  });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Bias" }));
  expect(scrolled).not.toHaveBeenCalled();
  await act(async () => { finish({ ok: true, json: async () => analysis }); });
  await waitFor(() => expect(scrolled).toHaveBeenCalled());
  expect((scrolled.mock.contexts[0] as HTMLElement).id).toBe("analysis-results");
  expect(document.activeElement).toBe(screen.getByRole("region", { name: "Article analysis results" }));
});

it("does not scroll to stale results after the article is edited during analysis", async () => {
  const scrolled = vi.fn();
  Object.defineProperty(Element.prototype, "scrollIntoView", { configurable: true, value: scrolled });
  let finish: (result: unknown) => void;
  vi.stubGlobal("fetch", vi.fn().mockImplementation((url) => url === "/bias/samples"
    ? Promise.resolve({ ok: true, json: async () => [] })
    : new Promise((resolve) => { finish = resolve; })));
  render(<App />);
  fireEvent.change(screen.getByPlaceholderText("Paste article text here..."), {
    target: { value: "The council announced a vote." },
  });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Bias" }));
  fireEvent.change(screen.getByPlaceholderText("Paste article text here..."), { target: { value: "A different article." } });
  await act(async () => { finish({ ok: true, json: async () => analysis }); });
  expect(scrolled).not.toHaveBeenCalled();
  expect(screen.queryByRole("region", { name: "Article analysis results" })).toBeNull();
});
