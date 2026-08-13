import * as ort from "onnxruntime-web/webgpu";
import type { ModelSpec, ScoreRequest, ScoreResponse } from "./types";

class Preprocessor {
  constructor(private spec: ModelSpec) {}

  async toTensor(blob: Blob): Promise<ort.Tensor> {
    const { inputSize, resizeSize, mean, std } = this.spec;
    const bmp = await createImageBitmap(blob);
    const scale = resizeSize / Math.min(bmp.width, bmp.height);
    const w = Math.max(inputSize, Math.round(bmp.width * scale));
    const h = Math.max(inputSize, Math.round(bmp.height * scale));
    const canvas = new OffscreenCanvas(inputSize, inputSize);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, (inputSize - w) / 2, (inputSize - h) / 2, w, h);
    bmp.close();

    const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
    const n = inputSize * inputSize;
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        out[c * n + i] = (data[i * 4 + c] / 255 - mean[c]) / std[c];
      }
    }
    return new ort.Tensor("float32", out, [1, 3, inputSize, inputSize]);
  }
}

export class InferenceEngine {
  private session: Promise<ort.InferenceSession> | null = null;
  private preprocessor: Preprocessor | null = null;

  private async init(): Promise<ort.InferenceSession> {
    if (!this.session) {
      ort.env.wasm.wasmPaths = chrome.runtime.getURL("ort/");
      ort.env.wasm.numThreads = 1;
      this.session = (async () => {
        const specUrl = chrome.runtime.getURL("model/model.json");
        const spec = (await (await fetch(specUrl)).json()) as ModelSpec;
        this.preprocessor = new Preprocessor(spec);
        const load = async (file: string, providers: string[]) => {
          const url = chrome.runtime.getURL(`model/${file}`);
          const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
          return ort.InferenceSession.create(bytes, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
          });
        };
        if (spec.fp16File && (await this.webgpuUsable())) {
          try {
            const s = await load(spec.fp16File, ["webgpu"]);
            console.debug("[slopdetect] backend: webgpu fp16");
            return s;
          } catch (e) {
            console.debug("[slopdetect] webgpu init failed, falling back to wasm:", e);
          }
        }
        const s = await load(spec.file, ["wasm"]);
        console.debug("[slopdetect] backend: wasm int8");
        return s;
      })();
    }
    return this.session;
  }

  private async webgpuUsable(): Promise<boolean> {
    try {
      const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      return Boolean(gpu && (await gpu.requestAdapter()));
    } catch {
      return false;
    }
  }

  async score(url: string): Promise<number> {
    const session = await this.init();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${resp.status}`);
    const tensor = await this.preprocessor!.toTensor(await resp.blob());
    const feeds = { [session.inputNames[0]]: tensor };
    const output = await session.run(feeds);
    const logit = (output[session.outputNames[0]].data as Float32Array)[0];
    return 1 / (1 + Math.exp(-logit));
  }
}

export function listenForScoreRequests(engine: InferenceEngine, target: string): void {
  const inflight = new Map<string, Promise<number>>();
  chrome.runtime.onMessage.addListener(
    (msg: ScoreRequest & { target?: string }, _sender, sendResponse: (r: ScoreResponse) => void) => {
      if (msg.target !== target || msg.kind !== "score") return false;
      let p = inflight.get(msg.url);
      if (!p) {
        p = engine.score(msg.url).finally(() => inflight.delete(msg.url));
        inflight.set(msg.url, p);
      }
      p.then((prob) => sendResponse({ ok: true, prob }))
        .catch((e: Error) => sendResponse({ ok: false, error: e.message }));
      return true;
    },
  );
}
