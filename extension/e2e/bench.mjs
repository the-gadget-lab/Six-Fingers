import { launch } from "puppeteer-core";
import { createServer } from "node:http";
import { readFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const DIST = resolve(import.meta.dirname, "../dist");
const BENCH = process.env.BENCH_DIR || resolve(import.meta.dirname, "../../data/bench");
const THRESHOLD = 0.65;

function findChrome() {
  const base = resolve(import.meta.dirname, "../.chrome/chrome");
  const dir = readdirSync(base).find((d) => d.startsWith("linux-"));
  return join(base, dir, "chrome-linux64", "chrome");
}

function collect() {
  const items = [];
  for (const label of ["real", "fake"]) {
    for (const source of readdirSync(join(BENCH, label))) {
      for (const f of readdirSync(join(BENCH, label, source))) {
        items.push({ label, source, path: join(BENCH, label, source, f) });
      }
    }
  }
  return items;
}

function serve(items) {
  const tags = items
    .map((it, i) => `<img id="im${i}" src="/img${i}.jpg" width="300" loading="eager">`)
    .join("\n");
  const server = createServer((req, res) => {
    const m = req.url.match(/^\/img(\d+)\.jpg$/);
    if (req.url === "/") {
      res.setHeader("content-type", "text/html");
      res.end(`<!doctype html><body>${tags}</body>`);
    } else if (m) {
      res.setHeader("content-type", "image/jpeg");
      res.end(readFileSync(items[+m[1]].path));
    } else {
      res.statusCode = 404;
      res.end();
    }
  });
  return new Promise((ok) => server.listen(0, () => ok([server, server.address().port])));
}

const items = collect();
console.log(`${items.length} bench images`);
const [server, port] = await serve(items);
const browser = await launch({
  executablePath: process.env.CHROME_BIN || findChrome(),
  headless: false,
  args: [
    "--headless=new",
    `--disable-extensions-except=${DIST}`,
    `--load-extension=${DIST}`,
    "--no-sandbox",
  ],
  protocolTimeout: 1800000,
});

const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 120000 });

const t0 = Date.now();
let done = 0;
while (done < items.length && Date.now() - t0 < 1500000) {
  await new Promise((r) => setTimeout(r, 5000));
  done = await page.$$eval(
    "img",
    (els) => els.filter((e) => e.dataset.slopdetectProb || e.dataset.slopdetectError).length,
  );
  process.stdout.write(`\r${done}/${items.length} scored (${((Date.now() - t0) / 1000) | 0}s)`);
}
console.log();

const results = await page.$$eval("img", (els) =>
  els.map((e) => ({ prob: e.dataset.slopdetectProb, err: e.dataset.slopdetectError ?? null })),
);
await browser.close();
server.close();

const per = new Map();
let tp = 0, tn = 0, fp = 0, fn = 0, errs = 0;
results.forEach((r, i) => {
  const it = items[i];
  if (r.prob === undefined) {
    errs++;
    return;
  }
  const isAi = +r.prob >= THRESHOLD;
  const correct = it.label === "fake" ? isAi : !isAi;
  if (it.label === "fake") isAi ? tp++ : fn++;
  else isAi ? fp++ : tn++;
  const s = per.get(it.source) ?? { n: 0, ok: 0 };
  s.n++;
  s.ok += correct ? 1 : 0;
  per.set(it.source, s);
});
const tpr = tp / (tp + fn);
const tnr = tn / (tn + fp);
console.log(`TPR ${tpr.toFixed(4)}  TNR ${tnr.toFixed(4)}  balanced_acc@0.65 ${((tpr + tnr) / 2).toFixed(4)}  errors ${errs}`);
for (const [src, s] of [...per.entries()].sort()) {
  console.log(`  ${src}: ${(s.ok / s.n).toFixed(3)} (${s.n})`);
}
