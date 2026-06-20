#!/usr/bin/env node
/**
 * Load a WPT conformance .https.any.js file and print JSON to stdout.
 *
 * Usage:
 *   node dump_tests.mjs PATH/TO/add.https.any.js [--utils PATH/TO/utils.js]
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';

import { loadWptConformanceFile } from './load-wpt-file.mjs';

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
  const opts = { jsPath: null, utilsPath: null };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--utils') opts.utilsPath = argv[++i];
    else if (!opts.jsPath) opts.jsPath = arg;
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!opts.jsPath) {
    console.error('Usage: node dump_tests.mjs PATH/TO/test.https.any.js [--utils PATH/TO/utils.js]');
    process.exit(2);
  }
  return opts;
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

const opts = parseArgs(process.argv);
const sourceText = readFileSync(opts.jsPath, 'utf8');
const fileName = path.basename(opts.jsPath);

const loaded = loadWptConformanceFile(sourceText, fileName, {
  utilsPath: opts.utilsPath ?? undefined
});

const payload = {
  fileName,
  tests: loaded.tests.map((test) => {
    const entry = normalizeValue(test);
    const tolerance = resolveToleranceForGraph(loaded, test.graph);
    if (tolerance) {
      entry.tolerance = tolerance;
    }
    return entry;
  })
};

process.stdout.write(JSON.stringify(payload));
