import { launch } from "puppeteer-core";
import { createServer } from "node:http";
import { readFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const DIST = resolve(import.meta.dirname, "../dist");
const CHROME = process.env.CHROME_BIN || findChrome();

function findChrome() {
  const base = resolve(import.meta.dirname, "../.chrome/chrome");
  const dir = readdirSync(base).find((d) => d.startsWith("linux-"));
  return join(base, dir, "chrome-linux64", "chrome");
}

function pickImages() {
  const root = resolve(import.meta.dirname, "../../data/cf_eval");
  const pick = (dir, n) =>
    readdirSync(dir)
      .filter((f) => /\.(jpg|jpeg|png|webp)$/i.test(f))
      .slice(0, n)
      .map((f) => join(dir, f));
  return {
    real: pick(join(root, "real/real_coco"), 2),
    fake: pick(join(root, "fake/Dalle3"), 2),
  };
}

function serve(images) {
  const files = new Map();
  const tags = [];
  [...images.real.map((p) => ["real", p]), ...images.fake.map((p) => ["fake", p])].forEach(
    ([cls, p], i) => {
      const name = `/img${i}.${p.split(".").pop()}`;
      files.set(name, p);
      tags.push(`<img id="im${i}" data-cls="${cls}" src="${name}" width="400">`);
    },
  );
  const server = createServer((req, res) => {
    if (req.url === "/") {
      res.setHeader("content-type", "text/html");
      res.end(`<!doctype html><body>${tags.join("\n")}</body>`);
    } else if (files.has(req.url)) {
      res.end(readFileSync(files.get(req.url)));
    } else {
      res.statusCode = 404;
      res.end();
    }
  });
  return new Promise((ok) => server.listen(0, () => ok([server, server.address().port])));
}

const [server, port] = await serve(pickImages());
const browser = await launch({
  executablePath: CHROME,
  headless: false,
  args: [
    "--headless=new",
    `--disable-extensions-except=${DIST}`,
    `--load-extension=${DIST}`,
    "--no-sandbox",
  ],
});

const page = await browser.newPage();
page.on("console", (m) => console.log(`[page ${m.type()}]`, m.text()));
page.on("response", (r) => r.status() >= 400 && console.log("[404]", r.url()));
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle0" });
console.log(
  "imgs:",
  await page.$$eval("img", (els) =>
    els.map((e) => ({ id: e.id, cls: e.dataset.cls, w: e.naturalWidth, ok: e.complete })),
  ),
);

try {
  await page.waitForSelector(".slopdetect-badge", { timeout: 60000 });
  await new Promise((r) => setTimeout(r, 8000));
  const badges = await page.$$eval(".slopdetect-badge", (els) =>
    els.map((e) => ({ text: e.textContent, verdict: e.dataset.verdict })),
  );
  console.log("badges:", JSON.stringify(badges));
  if (badges.length !== 4) throw new Error(`expected 4 badges, got ${badges.length}`);
  console.log("SMOKE_OK");
} finally {
  await browser.close();
  server.close();
}
