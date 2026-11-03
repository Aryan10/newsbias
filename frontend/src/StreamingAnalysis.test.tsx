import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("handles pure out-of-order streaming chunks where political arrives before sentence and isolated LLM", async () => {
  const text = "Breaking news headline. This is a very testable article sentence.";

  // Mock SSE stream reader that pushes political_specialist FIRST, then isolated_llm, then sentence_specialist, then complete
  const chunks = [
    `data: ${JSON.stringify({
      stage: "political_specialist",
      specialist_political: {
        status: "available",
        reason: null,
        method: "in_house",
        model_name: "newsbias_article",
        predicted_label: "RIGHT",
        confidence: 0.88,
        scores: { LEFT: 0.05, CENTER: 0.07, RIGHT: 0.88 },
      },
    })}\n\n`,
    `data: ${JSON.stringify({
      stage: "isolated_llm",
      isolated_llm: {
        political_leaning: { label: "center", confidence: 0.9 },
        linguistic_bias: { label: "non-biased", is_biased: false, bias_score: 0.1 },
        loaded_phrases: [],
        reasoning: "Zero-shot baseline reading in isolation.",
      },
    })}\n\n`,
    `data: ${JSON.stringify({
      stage: "sentence_specialist",
      sentences: [
        { text: "Breaking news headline.", is_biased: false, label: "NON-BIASED", score: 0.05 },
        { text: "This is a very testable article sentence.", is_biased: true, label: "BIASED", score: 0.85 },
      ],
      specialist_sentence: {
        status: "available",
        reason: null,
        label: "BIASED",
        biased_sentence_count: 1,
        total_sentences: 2,
        bias_ratio: 0.5,
        flagged_sentences: [
          { text: "This is a very testable article sentence.", is_biased: true, label: "BIASED", score: 0.85 },
        ],
      },
      input: { title: "", text },
      language: { code: "en", name: "English", status: "detected", confidence: 1 },
      analysis_method: "specialists+llm",
    })}\n\n`,
    `data: ${JSON.stringify({
      stage: "complete",
      data: {
        analysis_version: 2,
        language: { code: "en", name: "English", status: "detected", confidence: 1 },
        analysis_method: "specialists+llm",
        input: { title: "", text },
        sentences: [
          { text: "Breaking news headline.", is_biased: false, label: "NON-BIASED", score: 0.05 },
          { text: "This is a very testable article sentence.", is_biased: true, label: "BIASED", score: 0.85 },
        ],
        specialist_sentence: {
          status: "available",
          reason: null,
          label: "BIASED",
          biased_sentence_count: 1,
          total_sentences: 2,
          bias_ratio: 0.5,
          flagged_sentences: [],
        },
        specialist_political: {
          status: "available",
          reason: null,
          method: "in_house",
          model_name: "newsbias_article",
          predicted_label: "RIGHT",
          confidence: 0.88,
          scores: { LEFT: 0.05, CENTER: 0.07, RIGHT: 0.88 },
        },
        isolated_llm: {
          political_leaning: { label: "center", confidence: 0.9 },
          linguistic_bias: { label: "non-biased", is_biased: false, bias_score: 0.1 },
          loaded_phrases: [],
          reasoning: "Zero-shot baseline reading in isolation.",
        },
        arbiter: {
          final_verdict: {
            is_biased: true,
            overall_leaning: "right",
            confidence: 0.85,
            primary_bias_type: "linguistic",
          },
          why_biased_heading: "Why is it biased?",
          why_biased_explanation: "One sentence carries noticeable spin.",
          key_findings: ["Sentence 2 flagged."],
        },
      },
    })}\n\n`,
  ];

  let chunkIndex = 0;
  const mockReader = {
    read: vi.fn().mockImplementation(async () => {
      if (chunkIndex < chunks.length) {
        const textChunk = chunks[chunkIndex++];
        const encoded = new TextEncoder().encode(textChunk);
        return { done: false, value: encoded };
      }
      return { done: true, value: undefined };
    }),
  };

  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(async (url) => {
      if (url === "/bias/samples") {
        return { ok: true, json: async () => [] };
      }
      return {
        ok: true,
        body: { getReader: () => mockReader },
      };
    })
  );

  render(<App />);
  fireEvent.change(screen.getByPlaceholderText("Paste article text here..."), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Bias" }));

  // Verify full stream completes and dynamic heading is shown
  await waitFor(() => {
    expect(screen.getByText("Why is it biased?")).toBeTruthy();
  });

  // Verify all 3 specialist results rendered accurately
  expect(screen.getByText("RIGHT")).toBeTruthy();
  expect(screen.getByText("88% conf")).toBeTruthy();
  expect(screen.getByText("50%")).toBeTruthy();
  expect(screen.getByText("Zero-shot baseline reading in isolation.")).toBeTruthy();
});
