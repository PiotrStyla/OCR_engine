import { createWorker, PSM } from 'tesseract.js';
import { readFileSync, readdirSync, mkdirSync, writeFileSync, appendFileSync } from 'node:fs';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { createRequire } from 'node:module';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const require = createRequire(import.meta.url);
const corePackagePath = require.resolve('tesseract.js-core/package.json', {
  paths: [dirname(require.resolve('tesseract.js'))],
});
const [sampleDirectory, outputDirectory] = process.argv.slice(2);
if (!sampleDirectory || !outputDirectory) throw new Error('Usage: node run.mjs SAMPLE_DIRECTORY OUTPUT_DIRECTORY');
const output = resolve(outputDirectory);
mkdirSync(output, { recursive: false });
const hash = data => createHash('sha256').update(data).digest('hex');
const fixture = join(root, 'benchmarks/smoke-v1/manifest.jsonl');
const page = JSON.parse(readFileSync(fixture, 'utf8').trim());
const manifestMode = sampleDirectory.endsWith('.jsonl');
const cases = manifestMode ? readFileSync(sampleDirectory, 'utf8').trim().split('\n').map(line => {
  const row = JSON.parse(line);
  const path = resolve(dirname(resolve(sampleDirectory)), row.image);
  if (hash(readFileSync(path)) !== row.sha256) throw new Error(`Checksum mismatch: ${row.id}`);
  return { id: row.id, kind: 'page', path, reference: row.text };
}) : [{ id: page.id, kind: 'page', path: resolve(dirname(fixture), page.image), reference: page.text }];
for (const file of (manifestMode ? [] : readdirSync(sampleDirectory)).filter(f => f.endsWith('.png')).sort()) {
  cases.push({ id: `line-${file.slice(0, -4)}`, kind: 'line', path: resolve(sampleDirectory, file),
    reference: readFileSync(join(sampleDirectory, file.replace(/\.png$/, '.txt')), 'utf8').trim() });
}
for (const item of cases) item.sha256 = hash(readFileSync(item.path));
writeFileSync(join(output, 'manifest.json'), JSON.stringify(cases, null, 2));
const cachePath = join(here, 'cache');
mkdirSync(cachePath, { recursive: true });
const start = performance.now();
// Rejected page jobs are recorded below; avoid the worker's uncaught rethrow.
const worker = await createWorker('pol+eng', 1, { cachePath, errorHandler: () => {} });
const startupSeconds = (performance.now() - start) / 1000;
let completedPages = 0;
let errorPages = 0;
try {
  for (const item of cases) {
    const psm = item.kind === 'line' ? PSM.SINGLE_LINE : PSM.AUTO;
    await worker.setParameters({ tessedit_pageseg_mode: psm });
    const began = performance.now();
    let result;
    try {
      const { data } = await worker.recognize(item.path);
      const empty = !data.text.trim();
      result = { ...item, path: undefined, status: empty ? 'error' : 'ok', text: data.text,
        error_type: empty ? 'EmptyRecognition' : undefined,
        confidence: empty ? null : data.confidence, raw_engine_confidence: data.confidence,
        psm, elapsed_seconds: (performance.now() - began) / 1000 };
    } catch (error) {
      result = { ...item, path: undefined, status: 'error', text: '', error_type: error?.name || 'RecognitionError',
        psm, elapsed_seconds: (performance.now() - began) / 1000 };
    }
    appendFileSync(join(output, 'predictions.jsonl'), JSON.stringify(result) + '\n');
    completedPages += 1;
    if (result.status === 'error') errorPages += 1;
    console.log(`${item.id}: ${result.status} (${result.elapsed_seconds.toFixed(2)}s)`);
  }
} finally {
  await worker.terminate();
}
const weights = Object.fromEntries(readdirSync(cachePath).filter(f => f.endsWith('.traineddata'))
  .map(f => [f, hash(readFileSync(join(cachePath, f)))]));
writeFileSync(join(output, 'run.json'), JSON.stringify({ engine: 'tesseract.js',
  package: JSON.parse(readFileSync(join(here, 'node_modules/tesseract.js/package.json'))).version,
  core_package: JSON.parse(readFileSync(corePackagePath)).version,
  runner_sha256: hash(readFileSync(fileURLToPath(import.meta.url))),
  pages_expected: cases.length, pages_completed: completedPages, pages_error: errorPages,
  preprocessing: 'original image bytes; no external resizing or enhancement',
  lock_sha256: hash(readFileSync(join(here, 'pnpm-lock.yaml'))), languages: 'pol+eng', oem: 1,
  workers: 1, startup_seconds: startupSeconds, model_hashes: weights,
  node: process.version, cpu: os.cpus()[0].model, platform: process.platform,
  time_utc: new Date().toISOString(), scope: manifestMode ? 'Public scan diagnostic; not SOTA benchmark' : '1 synthetic page + 12 oracle line crops; not SOTA benchmark',
  input_manifest_sha256: manifestMode ? hash(readFileSync(sampleDirectory)) : null
}, null, 2));
