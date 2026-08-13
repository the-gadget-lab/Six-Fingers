import { launch } from "puppeteer-core";
import { readdirSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { createServer } from "node:http";

const DIST = resolve(import.meta.dirname, "../dist");

function findChrome() {
  const base = resolve(import.meta.dirname, "../.chrome/chrome");
  const dir = readdirSync(base).find((d) => d.startsWith("linux-"));
  return join(base, dir, "chrome-linux64", "chrome");
}

const imgDir = resolve(import.meta.dirname, "../../data/cf_eval/fake/Dalle3");
const img = join(imgDir, readdirSync(imgDir)[0]);
const server = createServer((req, res) => {
  if (req.url === "/") {
    res.setHeader("content-type", "text/html");
    res.end('<img src="/a.png" width="400">');
  } else {
    res.end(readFileSync(img));
  }
});
await new Promise((ok) => server.listen(0, ok));

const browser = await launch({
  executablePath: findChrome(),
  headless: false,
  args: [
    "--headless=new",
    `--disable-extensions-except=${DIST}`,
    `--load-extension=${DIST}`,
    "--no-sandbox",
    ...(process.env.EXTRA_CHROME_ARGS?.split(" ") ?? []),
  ],
});

browser.on("targetcreated", async (t) => {
  if (t.url().includes("offscreen")) {
    const p = await t.page().catch(() => null);
    if (p) p.on("console", (m) => console.log("[offscreen]", m.type(), m.text()));
  }
});

const page = await browser.newPage();
console.log(
  "page gpu adapter:",
  await page.evaluate(async () => Boolean(navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null)))),
);
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.waitForSelector(".slopdetect-badge", { timeout: 90000 });
await new Promise((r) => setTimeout(r, 1500));
await browser.close();
server.close();
