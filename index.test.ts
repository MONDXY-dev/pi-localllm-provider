import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import {
  formatModelLine,
  mergeModels,
  modelIdsChanged,
  modelsHeading,
  normalizeBaseUrl,
  registerServer,
  type LLMServer,
} from "./index.ts";

describe("normalizeBaseUrl", () => {
  it("appends /v1 when missing", () => {
    expect(normalizeBaseUrl("http://localhost:8000")).toBe("http://localhost:8000/v1");
  });

  it("leaves an existing /v1 suffix alone", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1")).toBe("http://localhost:8000/v1");
  });

  it("strips trailing slashes before checking the suffix", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1/")).toBe("http://localhost:8000/v1");
    expect(normalizeBaseUrl("http://localhost:8000/")).toBe("http://localhost:8000/v1");
  });

  it("defaults to http:// for a bare host:port with no scheme", () => {
    expect(normalizeBaseUrl("localhost:11434")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("192.168.1.50:8000")).toBe("http://192.168.1.50:8000/v1");
  });

  it("preserves an explicit https:// scheme", () => {
    expect(normalizeBaseUrl("https://my.server.com")).toBe("https://my.server.com/v1");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeBaseUrl("  localhost:11434  ")).toBe("http://localhost:11434/v1");
  });
});

describe("formatModelLine", () => {
  it("formats context window and max tokens in k, with no capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "some-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • some-model  (ctx 64k, max 8k)");
  });

  it("appends reasoning and vision tags when present", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "vision-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: true,
        input: ["text", "image"],
      }),
    ).toBe("  • vision-model  (ctx 64k, max 8k, reasoning, vision)");
  });

  it("shows sub-1024 windows without a k suffix", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "tiny",
        contextWindow: 512,
        maxTokens: 256,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • tiny  (ctx 512, max 256)");
  });

  it("prefixes a checkmark when loaded is true", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: true,
      }),
    ).toBe("  • ✓ m  (ctx 4k, max 2k)");
  });

  it("prefixes a hollow circle when loaded is false", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: false,
      }),
    ).toBe("  • ○ m  (ctx 4k, max 2k)");
  });

  it("omits the loaded prefix entirely when loaded is unknown", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • m  (ctx 4k, max 2k)");
  });

  it("shows size and quantization when present, in order before capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: true,
        input: ["text", "image"],
        sizeBytes: 4912898304,
        quantization: "Q4_K_M",
      }),
    ).toBe("  • m  (ctx 4k, max 2k, 4.6G, Q4_K_M, reasoning, vision)");
  });
});

describe("modelIdsChanged", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("is false when the same single model is refreshed with new metadata", () => {
    expect(modelIdsChanged([baseModel], [{ ...baseModel, contextWindow: 8192 }])).toBe(false);
  });

  it("is false when the same set of models comes back in a different order", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a, b], [b, a])).toBe(false);
  });

  it("is true when the model count changes", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a], [a, b])).toBe(true);
  });

  it("is true when a same-count refresh swaps in a different model id", () => {
    const a = { ...baseModel, id: "a" };
    const c = { ...baseModel, id: "c" };
    expect(modelIdsChanged([a], [c])).toBe(true);
  });

  it("is false for two empty lists", () => {
    expect(modelIdsChanged([], [])).toBe(false);
  });
});

describe("modelsHeading", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("adds the loaded-state legend when at least one model reports it", () => {
    expect(modelsHeading([{ ...baseModel, loaded: true }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
    expect(modelsHeading([{ ...baseModel }, { ...baseModel, loaded: false }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
  });

  it("omits the legend when no model reports loaded state", () => {
    expect(modelsHeading([baseModel])).toBe("Models:");
    expect(modelsHeading([])).toBe("Models:");
  });
});

describe("mergeModels", () => {
  const stored = {
    id: "deepseek-v4.1-flash",
    name: "deepseek-v4.1-flash",
    contextWindow: 200000,
    maxTokens: 32768,
    reasoning: true,
    input: ["text", "image"] as ("text" | "image")[],
    compat: { supportsReasoningEffort: true, supportsDeveloperRole: true },
    thinkingLevelMap: { off: "none", high: "medium" },
    samplingParams: { temperature: 0.6 },
  };
  // What a bare portal card detects: 32k default, no capabilities.
  const bare = {
    id: "deepseek-v4.1-flash",
    name: "deepseek-v4.1-flash",
    contextWindow: 32768,
    maxTokens: 8192,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("keeps stored metadata when detection reports only defaults", () => {
    const [m] = mergeModels([stored], [bare]);
    expect(m.contextWindow).toBe(200000);
    expect(m.maxTokens).toBe(32768);
    expect(m.reasoning).toBe(true);
    expect(m.input).toEqual(["text", "image"]);
    expect(m.compat).toEqual(stored.compat);
    expect(m.thinkingLevelMap).toEqual(stored.thinkingLevelMap);
    expect(m.samplingParams).toEqual(stored.samplingParams);
  });

  it("lets strictly better detection win", () => {
    const [m] = mergeModels([stored], [{ ...bare, contextWindow: 1000000, maxTokens: 131072 }]);
    expect(m.contextWindow).toBe(1000000);
    expect(m.maxTokens).toBe(131072);
  });

  it("prefers a freshly detected alias over a stored id-as-name", () => {
    const [m] = mergeModels(
      [{ ...stored, name: "stored-id" }],
      [{ ...bare, name: "Qwen Alias" }],
    );
    expect(m.name).toBe("Qwen Alias");
  });

  it("keeps detection's own compat evidence over a stored override", () => {
    const detected = { ...bare, compat: { supportsReasoningEffort: false, supportsDeveloperRole: false } };
    const [m] = mergeModels([stored], [detected]);
    expect(m.compat).toEqual(detected.compat);
  });

  it("passes unknown models through untouched", () => {
    const [m] = mergeModels([], [bare]);
    expect(m).toEqual(bare);
  });
});

describe("normalizeBaseUrl with an explicit protocol", () => {
  it("keeps the bare root for Anthropic, whose client appends /v1/messages", () => {
    expect(normalizeBaseUrl("https://api.example.com", "anthropic")).toBe("https://api.example.com");
    expect(normalizeBaseUrl("https://api.example.com/v1", "anthropic")).toBe("https://api.example.com");
    expect(normalizeBaseUrl("https://api.example.com/v1/", "anthropic")).toBe("https://api.example.com");
  });

  it("still appends /v1 for every OpenAI-compatible backend", () => {
    expect(normalizeBaseUrl("https://api.example.com", "vllm")).toBe("https://api.example.com/v1");
    expect(normalizeBaseUrl("https://api.example.com", undefined)).toBe("https://api.example.com/v1");
  });
});

describe("registerServer", () => {
  function capture(server: LLMServer) {
    const registered: { id: string; config: Record<string, unknown> }[] = [];
    const pi = {
      registerProvider: (id: string, config: Record<string, unknown>) => {
        registered.push({ id, config });
      },
      unregisterProvider: () => {},
    } as unknown as ExtensionAPI;
    registerServer(pi, server);
    return registered[0];
  }

  const model = {
    id: "hk-5-5",
    name: "HK-5.5",
    contextWindow: 200000,
    maxTokens: 64000,
    reasoning: true,
    input: ["text", "image"] as ("text" | "image")[],
  };

  it("registers Anthropic servers against /v1/messages with a Bearer key", () => {
    const { config } = capture({
      id: "abc123",
      name: "kui1 HK",
      baseUrl: "https://api.example.com",
      apiKey: "sk-test",
      apiType: "anthropic",
      models: [model],
    });
    expect(config.api).toBe("anthropic-messages");
    expect(config.authHeader).toBe(true);
    expect(config.headers).toEqual({ "anthropic-version": "2023-06-01" });
  });

  it("leaves OpenAI-compatible servers on chat completions without auth extras", () => {
    const { config } = capture({
      id: "abc123",
      name: "My vLLM",
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      apiType: "vllm",
      models: [model],
    });
    expect(config.api).toBe("openai-completions");
    expect(config.authHeader).toBeUndefined();
    expect(config.headers).toBeUndefined();
  });
});
