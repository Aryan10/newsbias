import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import NewsFeed from "./NewsFeed";
import type { FeedStory } from "./NewsFeed";
import App from "./App";

const stories: FeedStory[] = Array.from({ length: 8 }, (_, index) => ({
  id: `story-${index}`, title: `Headline ${index}`, source: index % 2 ? "City News" : "Public Wire",
  section: index % 2 ? "World" : "Business", description: `Original story excerpt ${index}.`,
  content: `Full article body ${index}.`, url: `https://news.example/story-${index}`,
  collection: "Curated examples",
}));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("Newsfeed", () => {
  it("shows readable story cards and reveals more stories without changing the article", () => {
    const onOpen = vi.fn();
    render(<NewsFeed stories={stories} selectedId="story-0" onOpen={onOpen} />);
    expect(screen.getAllByRole("article")).toHaveLength(6);
    expect(screen.getByText("In editor")).toBeTruthy();
    expect(screen.getByText("Original story excerpt 0.")).toBeTruthy();
    expect(screen.getByText("Showing 6 of 8 stories")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show more stories" }));
    expect(screen.getAllByRole("article")).toHaveLength(8);
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("combines section, source, and keyword filters and supports clearing them", () => {
    render(<NewsFeed stories={stories} selectedId={null} onOpen={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "World" }));
    expect(screen.getAllByRole("article")).toHaveLength(4);
    fireEvent.change(screen.getByRole("combobox", { name: "Filter by source" }), { target: { value: "City News" } });
    fireEvent.change(screen.getByRole("searchbox", { name: "Search stories" }), { target: { value: "headline 3" } });
    expect(screen.getAllByRole("article")).toHaveLength(1);
    expect(screen.getByRole("heading", { name: "Headline 3" })).toBeTruthy();
    fireEvent.change(screen.getByRole("searchbox", { name: "Search stories" }), { target: { value: "no matching words" } });
    expect(screen.queryAllByRole("article")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getAllByRole("article")).toHaveLength(6);
  });

  it("opens the selected story and keeps source links independent", () => {
    const onOpen = vi.fn();
    render(<NewsFeed stories={stories} selectedId={null} onOpen={onOpen} />);
    const source = screen.getByRole("link", { name: "View source: Headline 0" });
    expect(source.getAttribute("href")).toBe("https://news.example/story-0");
    fireEvent.click(source);
    expect(onOpen).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Open article: Headline 0" }));
    expect(onOpen).toHaveBeenCalledExactlyOnceWith(stories[0]);
  });

  it("only displays actual date/image metadata and hides broken images", () => {
    const { container } = render(<NewsFeed stories={[{
      ...stories[0], publishedAt: "2026-10-30T09:00:00Z", imageUrl: "https://news.example/photo.jpg",
    }, { ...stories[1], publishedAt: "invalid" }]} selectedId={null} onOpen={vi.fn()} />);
    expect(container.querySelectorAll("time")).toHaveLength(1);
    expect(container.querySelector("time").getAttribute("datetime")).toBe("2026-10-30T09:00:00Z");
    fireEvent.error(container.querySelector("img"));
    expect(container.querySelector("img")).toBeNull();
  });

  it("does not use unsafe source URL schemes", () => {
    render(<NewsFeed stories={[{ ...stories[0], url: "javascript:alert(1)" }]} selectedId={null} onOpen={vi.fn()} />);
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("loads a feed story into Article without automatically analyzing it", async () => {
    const sample = { id: "provider-independent", title: "A new article", text: "Full original article body.",
      dataset: "Example corpus", category: "Example", source_name: "Public Wire", source_url: "https://news.example/article" };
    const fetchMock = vi.fn().mockImplementation(async (url: string) => {
      if (url === "/bias/samples") return { ok: true, json: async () => [sample] };
      return { ok: true, json: async () => ({ stories: [] }) };
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Curated" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Open article: A new article" })).toBeTruthy());
    fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
    fireEvent.click(screen.getByRole("button", { name: "Open article: A new article" }));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Article" }).getAttribute("aria-selected")).toBe("true"));
    expect(screen.getByDisplayValue("A new article")).toBeTruthy();
    expect(screen.getByDisplayValue("Full original article body.")).toBeTruthy();
  });

  it("supports switching between Curated and Live collections and shows refresh button", () => {
    const onSelectCollection = vi.fn();
    const onRefresh = vi.fn();
    const { rerender } = render(
      <NewsFeed
        stories={stories}
        selectedId={null}
        onOpen={vi.fn()}
        collection="curated"
        onSelectCollection={onSelectCollection}
        onRefresh={onRefresh}
      />
    );
    expect(screen.getByRole("tab", { name: "Curated" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Live" }).getAttribute("aria-selected")).toBe("false");
    expect(screen.queryByRole("button", { name: "Refresh live news" })).toBeNull();

    fireEvent.click(screen.getByRole("tab", { name: "Live" }));
    expect(onSelectCollection).toHaveBeenCalledWith("live");

    rerender(
      <NewsFeed
        stories={stories}
        selectedId={null}
        onOpen={vi.fn()}
        collection="live"
        onSelectCollection={onSelectCollection}
        onRefresh={onRefresh}
      />
    );
    expect(screen.getByRole("tab", { name: "Live" }).getAttribute("aria-selected")).toBe("true");
    const refreshBtn = screen.getByRole("button", { name: "Refresh live news" });
    expect(refreshBtn).toBeTruthy();
    fireEvent.click(refreshBtn);
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("shows loading state for empty live feed", () => {
    render(
      <NewsFeed
        stories={[]}
        selectedId={null}
        onOpen={vi.fn()}
        collection="live"
        onSelectCollection={vi.fn()}
        isLoading={true}
      />
    );
    expect(screen.getByText("Fetching live news stories...")).toBeTruthy();
  });

  it("opening a live feed story calls /bias/import/url with article URL rather than copying preview", async () => {
    const curatedSample = {
      id: "curated-1",
      title: "Curated Sample Title",
      text: "Curated article body.",
      dataset: "Curated Dataset",
      category: "General",
      source_name: "Curated Wire",
      source_url: "https://news.example/curated-1",
    };

    const liveStory: FeedStory = {
      id: "rapidapi:abc12345",
      title: "Live Breaking News Headline",
      source: "The Hindu",
      section: "National",
      description: "Truncated preview snippet ...",
      content: undefined,
      url: "https://thehindu.com/article/live-123",
      publishedAt: "2026-10-30T10:00:00Z",
      collection: "live",
    };

    const extractedArticle = {
      title: "Extracted Full Headline",
      text: "This is the full extracted article content retrieved by Trafilatura from the real URL.",
      source: {
        type: "url",
        url: "https://thehindu.com/article/live-123",
        author: "Correspondent",
        publication_date: "2026-10-30",
      },
      extraction: {
        method: "trafilatura",
        notices: ["Extracted 3 paragraphs."],
      },
    };

    const fetchMock = vi.fn().mockImplementation(async (url: string) => {
      if (url === "/bias/samples") {
        return { ok: true, json: async () => [curatedSample] };
      }
      if (url.startsWith("/bias/live-news")) {
        return {
          ok: true,
          json: async () => ({
            stories: [liveStory],
            providers: { rapidapi: { status: "ok", count: 1 } },
            cached: false,
          }),
        };
      }
      if (url === "/bias/import/url") {
        return { ok: true, json: async () => extractedArticle };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    // Initial load defaults to Live collection
    await waitFor(() => expect(screen.getByRole("button", { name: "Open article: Live Breaking News Headline" })).toBeTruthy());

    // Click "Open article" for the live story
    fireEvent.click(screen.getByRole("button", { name: "Open article: Live Breaking News Headline" }));

    // Verify /bias/import/url was invoked with the exact article URL
    await waitFor(() => {
      const importCalls = fetchMock.mock.calls.filter((call) => call[0] === "/bias/import/url");
      expect(importCalls).toHaveLength(1);
      const postBody = JSON.parse(importCalls[0][1].body);
      expect(postBody.url).toBe("https://thehindu.com/article/live-123");
    });

    // Verify the Article tab receives the full extracted text, NOT the truncated preview
    await waitFor(() => {
      expect(screen.getByDisplayValue("Extracted Full Headline")).toBeTruthy();
      expect(screen.getByDisplayValue("This is the full extracted article content retrieved by Trafilatura from the real URL.")).toBeTruthy();
    });
    expect(screen.queryByDisplayValue("Truncated preview snippet ...")).toBeNull();
  });
});
