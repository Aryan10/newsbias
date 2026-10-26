import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ArticleInputTabs from "./ArticleInputTabs";
import type { ImportedArticle } from "./ArticleInputTabs";

const imported: ImportedArticle = {
  title: "Original headline",
  text: "Original paragraph.\n\nSecond paragraph.",
  source: { type: "raw", url: null, filename: null, author: null, publication_date: null },
  extraction: { method: "raw+agent", notices: [] },
};
const response = (body = imported, ok = true) => ({ ok, json: async () => body });

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function mount() {
  const articleRevision = { current: 0 };
  const onImport = vi.fn();
  const props = { articleRevision, onImport, selectedPresetRevision: 0 };
  const view = render(<ArticleInputTabs {...props}><p>Current article draft</p></ArticleInputTabs>);
  return { articleRevision, onImport, props, view };
}

describe("Article input tabs", () => {
  it("preserves independent drafts and returns a raw import to Article for review", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response());
    vi.stubGlobal("fetch", fetchMock);
    const { onImport } = mount();
    expect(screen.getByRole("tab", { name: "Article" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
    fireEvent.change(screen.getByLabelText("Copied website text"), { target: { value: "Copied page" } });
    fireEvent.click(screen.getByRole("tab", { name: "URL" }));
    fireEvent.change(screen.getByLabelText("News article URL"), { target: { value: "https://news.example/story" } });
    fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
    expect((screen.getByLabelText("Copied website text") as HTMLTextAreaElement).value).toBe("Copied page");
    fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
    await waitFor(() => expect(onImport).toHaveBeenCalledWith(imported));
    expect(fetchMock.mock.calls[0][0]).toBe("/bias/import/raw");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ text: "Copied page" });
    expect(screen.getByRole("tab", { name: "Article" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText("Current article draft")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "URL" }));
    expect((screen.getByLabelText("News article URL") as HTMLInputElement).value).toBe("https://news.example/story");
  });

  it("sends URL drafts to the URL importer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response());
    vi.stubGlobal("fetch", fetchMock);
    const { onImport } = mount();
    fireEvent.click(screen.getByRole("tab", { name: "URL" }));
    fireEvent.change(screen.getByLabelText("News article URL"), { target: { value: "https://news.example/story" } });
    fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
    await waitFor(() => expect(onImport).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0][0]).toBe("/bias/import/url");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ url: "https://news.example/story" });
  });

  it("retains the selected upload across tabs and sends multipart data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response());
    vi.stubGlobal("fetch", fetchMock);
    const { onImport } = mount();
    const file = new File(["Original article"], "article.txt", { type: "text/plain" });
    fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
    fireEvent.change(screen.getByLabelText("Upload a text document"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("tab", { name: "Article" }));
    fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
    expect(screen.getByText("Selected: article.txt")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
    await waitFor(() => expect(onImport).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0][0]).toBe("/bias/import/file");
    expect(fetchMock.mock.calls[0][1].body.get("file").name).toBe("article.txt");
    expect(fetchMock.mock.calls[0][1].headers).toBeUndefined();
  });

  it.each(["edited draft", "switched tab", "selected preset", "edited article"])(
    "ignores late extraction results after an %s", async (action) => {
      let resolve: (value: ReturnType<typeof response>) => void;
      const fetchMock = vi.fn().mockReturnValue(new Promise((done) => { resolve = done; }));
      vi.stubGlobal("fetch", fetchMock);
      const { articleRevision, onImport, props, view } = mount();
      fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
      fireEvent.change(screen.getByLabelText("Copied website text"), { target: { value: "Old page" } });
      fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
      if (action === "edited draft") {
        fireEvent.change(screen.getByLabelText("Copied website text"), { target: { value: "New page" } });
      } else if (action === "switched tab") {
        fireEvent.click(screen.getByRole("tab", { name: "URL" }));
      } else if (action === "selected preset") {
        view.rerender(<ArticleInputTabs {...props} selectedPresetRevision={1}><p>New preset</p></ArticleInputTabs>);
      } else articleRevision.current += 1;
      await act(async () => { resolve(response()); });
      expect(onImport).not.toHaveBeenCalled();
    },
  );

  it("keeps the current article when extraction fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false, json: async () => ({ detail: "No article was found." }),
    }));
    const { onImport } = mount();
    fireEvent.click(screen.getByRole("tab", { name: "Raw" }));
    fireEvent.change(screen.getByLabelText("Copied website text"), { target: { value: "Menu only" } });
    fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("No article was found."));
    expect(onImport).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("tab", { name: "Article" }));
    expect(screen.getByText("Current article draft")).toBeTruthy();
  });

  it("rejects unsupported files before uploading", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    mount();
    fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
    fireEvent.change(screen.getByLabelText("Upload a text document"), {
      target: { files: [new File(["PDF"], "article.pdf")] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Extract Article" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(".txt files"));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("supports keyboard tab navigation", () => {
    mount();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Article" }), { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Raw" }).getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Raw" }));
  });
});
