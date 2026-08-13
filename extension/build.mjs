import { build } from "esbuild";
import { createHash } from "node:crypto";
import { cp, mkdir, rm, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";

const MODEL_URL =
  "https://huggingface.co/Loke-60000/slop-detect-vit-s-onnx/resolve/main/model.onnx";
const MODEL_SHA256 = "d431efd677cc124ab68c3f1b20d628b7e9a8362a99803b1878475060b05cff5a";
const MODEL_PATH = "../model/dist/model.onnx";

if (!existsSync(MODEL_PATH)) {
  console.log("fetching model weights (one-time, ~23MB)...");
  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error(`model download failed: ${resp.status}`);
  const bytes = Buffer.from(await resp.arrayBuffer());
  const sha = createHash("sha256").update(bytes).digest("hex");
  if (sha !== MODEL_SHA256) throw new Error(`model hash mismatch: ${sha}`);
  await mkdir("../model/dist", { recursive: true });
  await writeFile(MODEL_PATH, bytes);
  console.log("model verified:", sha.slice(0, 16));
}

const outdir = "dist";
await rm(outdir, { recursive: true, force: true });
await mkdir(outdir, { recursive: true });

await build({
  entryPoints: {
    background: "src/background.ts",
    content: "src/content.ts",
    offscreen: "src/offscreen.ts",
    popup: "src/popup.ts",
  },
  bundle: true,
  format: "iife",
  outdir,
  minify: true,
  logLevel: "info",
});

await cp("public", outdir, { recursive: true });
await cp("../model/dist", `${outdir}/model`, { recursive: true });
await cp("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", `${outdir}/ort/ort-wasm-simd-threaded.wasm`);
await cp("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", `${outdir}/ort/ort-wasm-simd-threaded.mjs`);
console.log("built ->", outdir);

const ffdir = "dist-firefox";
await rm(ffdir, { recursive: true, force: true });
await cp(outdir, ffdir, { recursive: true });
await rm(`${ffdir}/offscreen.html`);
await rm(`${ffdir}/offscreen.js`);
const manifest = JSON.parse(await readFile(`${outdir}/manifest.json`, "utf8"));
manifest.background = { scripts: ["background.js"] };
manifest.permissions = manifest.permissions.filter((p) => p !== "offscreen");
manifest.browser_specific_settings = {
  gecko: { id: "slop-detect@lokman.dev", strict_min_version: "121.0" },
};
await writeFile(`${ffdir}/manifest.json`, JSON.stringify(manifest, null, 2));
console.log("built ->", ffdir);
