import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows skipped specialists without fake scores and preserves original-language article text", async () => {
  const text = "தமிழ்நாடு அரசு புதிய திட்டத்தை அறிவித்துள்ளது.\n\nஅதிகாரிகள் விவரங்களை வெளியிட்டனர்.";
  const analysis = {
    input: { text, title: "" }, language: { code: "ta", name: "Tamil", status: "detected", confidence: 1 },
    analysis_method: "llm_only",
    sentences: [{ text, score: null, is_biased: null, label: "NOT EVALUATED" }],
    specialist_sentence: { status: "skipped", reason: "English-only sentence checkpoint.", bias_ratio: null,
      biased_sentence_count: null, total_sentences: 2, label: "NOT EVALUATED", flagged_sentences: [] },
    specialist_political: { status: "skipped", reason: "English-only political checkpoint.", confidence: null,
      predicted_label: "NOT EVALUATED", scores: {} },
    isolated_llm: { political_leaning: { label: "center" }, linguistic_bias: { is_biased: false },
      reasoning: "Original-language LLM reading." },
    arbiter: { final_verdict: { is_biased: false, overall_leaning: "center" },
      why_biased_explanation: "LLM evidence supports this verdict." },
  };
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url) => ({
    ok: true, json: async () => url === "/bias/samples" ? [] : analysis,
  })));
  const { container } = render(<App />);
  fireEvent.change(screen.getByPlaceholderText("Paste article text here..."), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Bias" }));
  await waitFor(() => expect(screen.getByText("LLM-only analysis")).toBeTruthy());
  expect(screen.getByText("Tamil (ta)")).toBeTruthy();
  expect(screen.getAllByText("Not evaluated")).toHaveLength(2);
  expect(screen.queryByText("0%")).toBeNull();
  expect(screen.queryByText(/% conf/)).toBeNull();
  expect(screen.getByText("LLM Editorial Judgment")).toBeTruthy();
  expect(container.querySelector(".whitespace-pre-wrap")?.textContent).toBe(text);
  expect(screen.queryByText("Sentence-by-Sentence Spin Annotation")).toBeNull();
});
