import { SettingsStore, type Settings } from "./lib/settings";
import { MIN_IMAGE_DIM, type ScoreResponse } from "./lib/types";

class TaskQueue {
  private running = 0;
  private queue: (() => void)[] = [];

  constructor(private concurrency: number) {}

  run<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      const start = () => {
        this.running++;
        fn()
          .then(resolve, reject)
          .finally(() => {
            this.running--;
            this.queue.shift()?.();
          });
      };
      this.running < this.concurrency ? start() : this.queue.push(start);
    });
  }
}

class Badge {
  private el: HTMLElement;

  constructor(private img: HTMLImageElement, prob: number, threshold: number) {
    this.el = document.createElement("div");
    this.el.className = "slopdetect-badge";
    const pct = Math.round(prob * 100);
    const isAi = prob >= threshold;
    this.el.textContent = isAi ? `AI ${pct}%` : `${pct}%`;
    this.el.dataset.verdict = isAi ? "ai" : "real";
    document.documentElement.appendChild(this.el);
    this.place();
  }

  place(): void {
    const r = this.img.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) {
      this.el.style.display = "none";
      return;
    }
    this.el.style.display = "block";
    this.el.style.top = `${window.scrollY + r.top + 4}px`;
    this.el.style.left = `${window.scrollX + r.left + 4}px`;
  }

  remove(): void {
    this.el.remove();
  }
}

class BadgeLayer {
  private badges = new Map<HTMLImageElement, Badge>();

  constructor() {
    const reposition = () => {
      for (const [img, badge] of this.badges) {
        img.isConnected ? badge.place() : this.drop(img);
      }
    };
    let scheduled = false;
    const onScrollResize = () => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        reposition();
      });
    };
    addEventListener("scroll", onScrollResize, { passive: true, capture: true });
    addEventListener("resize", onScrollResize, { passive: true });
  }

  show(img: HTMLImageElement, prob: number, threshold: number, badgeAll: boolean): void {
    this.drop(img);
    if (!badgeAll && prob < threshold) return;
    this.badges.set(img, new Badge(img, prob, threshold));
  }

  drop(img: HTMLImageElement): void {
    this.badges.get(img)?.remove();
    this.badges.delete(img);
  }

  clear(): void {
    for (const img of [...this.badges.keys()]) this.drop(img);
  }
}

class ImageScanner {
  private seen = new WeakSet<HTMLImageElement>();
  private queue = new TaskQueue(4);

  constructor(private layer: BadgeLayer, private settings: Settings) {}

  start(): void {
    document.querySelectorAll("img").forEach((img) => this.watch(img));
    new MutationObserver((muts) => {
      for (const m of muts) {
        for (const node of m.addedNodes) {
          if (node instanceof HTMLImageElement) this.watch(node);
          else if (node instanceof Element) {
            node.querySelectorAll("img").forEach((img) => this.watch(img));
          }
        }
      }
    }).observe(document.documentElement, { childList: true, subtree: true });
  }

  private watch(img: HTMLImageElement): void {
    if (this.seen.has(img)) return;
    this.seen.add(img);
    void this.analyze(img);
  }

  private async analyze(img: HTMLImageElement): Promise<void> {
    if (!img.complete) {
      await new Promise((res) => {
        img.addEventListener("load", res, { once: true });
        img.addEventListener("error", res, { once: true });
      });
    }
    const url = img.currentSrc || img.src;
    if (!url || !/^(https?|data|blob):/.test(url)) return;
    if (img.naturalWidth < MIN_IMAGE_DIM || img.naturalHeight < MIN_IMAGE_DIM) return;

    const resp = await this.queue.run(
      () =>
        chrome.runtime.sendMessage({
          target: "background",
          kind: "score",
          url,
        }) as Promise<ScoreResponse>,
    );
    if (resp?.ok && resp.prob !== undefined) {
      img.dataset.slopdetectProb = resp.prob.toFixed(4);
      this.layer.show(img, resp.prob, this.settings.threshold, this.settings.badgeAll);
    } else {
      img.dataset.slopdetectError = resp?.error ?? "no response";
      console.debug("[slopdetect] scoring failed:", url, resp?.error);
    }
  }
}

async function main(): Promise<void> {
  const store = new SettingsStore();
  const settings = await store.get();
  if (!settings.enabled) return;
  const layer = new BadgeLayer();
  const scanner = new ImageScanner(layer, settings);
  store.onChange((s) => {
    Object.assign(settings, s);
    if (!s.enabled) layer.clear();
  });
  scanner.start();
}

void main();
