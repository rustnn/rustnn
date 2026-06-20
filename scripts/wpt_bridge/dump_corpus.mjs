#!/usr/bin/env node
/**
 * Load all WPT WebNN graph conformance tests and print one JSON corpus to stdout.
 *
 * Usage:
 *   node dump_corpus.mjs [--wpt-dir PATH]
 *
 * Output: { "wptDir": "...", "cases": [ { "fileName", "operation", "name", "graph", "tolerance"? }, ... ] }
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadWptConformanceFile } from './load-wpt-file.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../..');

function normalizeValue(v) {
  if (typeof v === 'number' && !Number.isFinite(v)) {
    if (Number.isNaN(v)) return 'NaN';
    return v > 0 ? 'Infinity' : '-Infinity';
  }
  if (typeof v === 'bigint') return v.toString();
  if (Array.isArray(v)) return v.map(normalizeValue);
  if (v && typeof v === 'object') {
    const out = {};
    for (const [k, val] of Object.entries(v)) out[k] = normalizeValue(val);
    return out;
  }
  return v;
}

function parseArgs(argv) {
  const opts = {
    wptDir: process.env.WPT_DIR ?? path.join(repoRoot, '.cache', 'wpt')
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--wpt-dir') opts.wptDir = argv[++i];
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  return opts;
}

function operationFromFileName(fileName) {
  const stem = fileName.split('.')[0];
  return stem.replace(/-/g, '_');
}

function isGraphConformanceFile(sourceText) {
  return sourceText.includes('webnn_conformance_test');
}

function resolveToleranceForGraph(loaded, graph) {
  if (typeof loaded.resolveTolerance !== 'function' || !graph) {
    return null;
  }
  try {
    const info = loaded.resolveTolerance(graph, {});
    if (!info || typeof info.value !== 'number') {
      return null;
    }
    if (info.metricType !== 'ULP' && info.metricType !== 'ATOL') {
      return null;
    }
    return { metricType: info.metricType, value: info.value };
  } catch {
    return null;
  }
}

function discoverConformanceFiles(wptDir) {
  const base = path.join(wptDir, 'webnn', 'conformance_tests');
  if (!existsSync(base)) {
    throw new Error(
      `WPT conformance_tests not found at ${base}. Run: node scripts/fetch_wpt.mjs`
    );
  }
  return readdirSync(base)
    .filter((name) => name.endsWith('.https.any.js'))
    .map((name) => path.join(base, name))
    .sort();
}

const opts = parseArgs(process.argv);
const utilsPath = path.join(opts.wptDir, 'webnn', 'resources', 'utils.js');
if (!existsSync(utilsPath)) {
  console.error(`WPT utils.js not found at ${utilsPath}`);
  console.error('Run: node scripts/fetch_wpt.mjs');
  process.exit(2);
}

/** @type {Array<{ fileName: string, operation: string, name: string, graph: any, tolerance?: object }>} */
const cases = [];
const fileErrors = [];

for (const jsPath of discoverConformanceFiles(opts.wptDir)) {
  const fileName = path.basename(jsPath);
  let sourceText;
  try {
    sourceText = readFileSync(jsPath, 'utf8');
  } catch (err) {
    fileErrors.push({ fileName, error: err.message });
    continue;
  }
  if (!isGraphConformanceFile(sourceText)) {
    continue;
  }

  try {
    const loaded = loadWptConformanceFile(sourceText, fileName, { utilsPath });
    const operation = operationFromFileName(fileName);
    for (const test of loaded.tests) {
      const graph = normalizeValue(test.graph);
      const entry = {
        fileName,
        operation,
        name: test.name,
        graph
      };
      const tolerance = resolveToleranceForGraph(loaded, test.graph);
      if (tolerance) {
        entry.tolerance = tolerance;
      }
      cases.push(entry);
    }
  } catch (err) {
    if (String(err.message).includes('No webnn_conformance_test')) {
      continue;
    }
    fileErrors.push({ fileName, error: err.message });
  }
}

process.stdout.write(
  JSON.stringify({
    wptDir: opts.wptDir,
    caseCount: cases.length,
    cases,
    fileErrors
  })
);
