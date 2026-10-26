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
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => [sample] });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Open article: A new article" })).toBeTruthy());
    fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
    fireEvent.click(screen.getByRole("button", { name: "Open article: A new article" }));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Article" }).getAttribute("aria-selected")).toBe("true"));
    expect(screen.getByDisplayValue("A new article")).toBeTruthy();
    expect(screen.getByDisplayValue("Full original article body.")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe("/bias/samples");
  });
});
