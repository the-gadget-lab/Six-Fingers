import type { ScoreRequest, ScoreResponse } from "./lib/types";

const hasOffscreen = typeof chrome.offscreen !== "undefined";

class OffscreenHost {
  private creating: Promise<void> | null = null;

  async ensure(): Promise<void> {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    });
    if (contexts.length > 0) return;
    if (!this.creating) {
      this.creating = chrome.offscreen
        .createDocument({
          url: "offscreen.html",
          reasons: [chrome.offscreen.Reason.DOM_PARSER],
          justification: "Run local ONNX inference on image pixels",
        })
        .finally(() => (this.creating = null));
    }
    return this.creating;
  }
}

interface Scorer {
  score(url: string): Promise<ScoreResponse>;
}

class OffscreenScorer implements Scorer {
  private host = new OffscreenHost();

  async score(url: string): Promise<ScoreResponse> {
    await this.host.ensure();
    const resp = (await chrome.runtime.sendMessage({
      target: "offscreen",
      kind: "score",
      url,
    })) as ScoreResponse;
    return resp ?? { ok: false, error: "no response" };
  }
}

class InlineScorer implements Scorer {
  private enginePromise = import("./lib/engine").then((m) => new m.InferenceEngine());

  async score(url: string): Promise<ScoreResponse> {
    try {
      const engine = await this.enginePromise;
      return { ok: true, prob: await engine.score(url) };
    } catch (e) {
      return { ok: false, error: (e as Error).message };
    }
  }
}

class ScoreRouter {
  private cache = new Map<string, ScoreResponse>();

  constructor(private scorer: Scorer) {}

  async score(url: string): Promise<ScoreResponse> {
    const hit = this.cache.get(url);
    if (hit) return hit;
    const resp = await this.scorer.score(url);
    if (resp?.ok) {
      if (this.cache.size > 2000) this.cache.clear();
      this.cache.set(url, resp);
    }
    return resp;
  }
}

const router = new ScoreRouter(hasOffscreen ? new OffscreenScorer() : new InlineScorer());

chrome.runtime.onMessage.addListener(
  (msg: ScoreRequest & { target?: string }, _sender, sendResponse: (r: ScoreResponse) => void) => {
    if (msg.target !== "background" || msg.kind !== "score") return false;
    router
      .score(msg.url)
      .then(sendResponse)
      .catch((e: Error) => sendResponse({ ok: false, error: e.message }));
    return true;
  },
);
