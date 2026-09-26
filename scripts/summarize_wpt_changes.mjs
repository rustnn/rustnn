#!/usr/bin/env node
/**
 * Generate a markdown PR body summarizing WPT expected-failures changes.
 *
 * Run in the CI aggregate job AFTER backend patches have been applied to the
 * working tree. Stages the changes so `git diff --cached` also sees newly-added
 * (untracked) list files produced by `git apply`.
 *
 * Model: every backend tracks conformance with {backend}_expected_failures.txt.
 * A list entry removed by a sync is a trial that now passes; an entry added is a
 * trial that now fails.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const CONFORMANCE_DIR = 'tests/wpt_conformance';
const WPT_REVISION = 'WPT_REVISION';

function git(args) {
  return execFileSync('git', args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
}

// {backend}::{operation}::{sanitized} -> {sanitized}
function txtName(line) {
  return line.split('::').pop();
}

// Parse a unified diff into added/removed sanitized names.
function txtDiff(file) {
  const diff = git(['diff', '--cached', 'HEAD', '--', file]);
  const added = new Set();
  const removed = new Set();
  for (const raw of diff.split('\n')) {
    if (raw.startsWith('+++') || raw.startsWith('---')) continue;
    const isAdded = raw.startsWith('+');
    if (!isAdded && !raw.startsWith('-')) continue;
    // Comment and blank lines carry no trial id (the Rust parser skips them too).
    const text = raw.slice(1).trim();
    if (!text || text.startsWith('#')) continue;
    const name = txtName(text);
    if (!name) continue;
    (isAdded ? added : removed).add(name);
  }
  return { added, removed };
}

// Stage expected-failure changes so `git diff --cached` also sees newly-added
// (untracked) list files produced by `git apply`.
git(['add', '-A', '--', CONFORMANCE_DIR]);

const nameStatus = git([
  'diff',
  '--cached',
  'HEAD',
  '--name-status',
  '--',
  CONFORMANCE_DIR,
]);
const entries = nameStatus ? nameStatus.split('\n').filter(Boolean) : [];

// Per-backend aggregation keyed by backend name.
const newFailures = new Map();
const newPasses = new Map();

function setName(map, backend, name) {
  if (!map.has(backend)) map.set(backend, new Set());
  map.get(backend).add(name);
}
// Pass totals of the sync runs, as recorded by the jobs in PASS_TOTALS
// ("<backend>: <n> passed, <n> skipped, <n> failed" per line).
function passTotals() {
  const totals = [];
  const notes = [];
  for (const line of (process.env.PASS_TOTALS ?? '').split('\n')) {
    const text = line.trim();
    if (!text) continue;
    const m = text.match(/^(\w+): (\d+) passed, (\d+) skipped, (\d+) failed$/);
    if (m) {
      totals.push({ backend: m[1], passed: m[2], skipped: m[3], failed: m[4] });
    } else {
      notes.push(text);
    }
  }
  totals.sort((a, b) => a.backend.localeCompare(b.backend));
  return { totals, notes };
}

for (const line of entries) {
  const parts = line.split('\t');
  const file = parts[parts.length - 1];
  if (!file || !file.includes('_expected_failures.txt')) continue;

  const backend = path.basename(file).replace(/_expected_failures\.txt$/, '');
  const { added, removed } = txtDiff(file);
  for (const name of added) setName(newFailures, backend, name);
  for (const name of removed) setName(newPasses, backend, name);
}

// GitHub runners are UTC; format DD-MM-YYYY.
function syncDate() {
  const d = new Date();
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  return `${dd}-${mm}-${d.getUTCFullYear()}`;
}

// Pinned WPT revision bump (old SHA -> new SHA), when it changed.
function wptRevisionChange() {
  let oldSha = '';
  try {
    oldSha = git(['show', `HEAD:${WPT_REVISION}`]);
  } catch {
    /* not committed yet */
  }
  let newSha = '';
  try {
    newSha = readFileSync(WPT_REVISION, 'utf8').trim();
  } catch {
    /* missing */
  }
  return oldSha !== newSha ? { oldSha, newSha } : null;
}

const out = [];
out.push('# WPT expected-failures sync');
out.push('');
out.push(
  `Updates the per-backend expected-failure lists to match the pinned WPT corpus (${syncDate()}).`
);
out.push('');
out.push(
  'Lists are rebuilt from a fresh run, so entries removed here are trials that now pass and ' +
    'entries added are trials that now fail. Trials that pass without being listed leave no ' +
    'artefact, so a stale entry is only ever visible as a removal in this diff.'
);
out.push('');

const pinChange = wptRevisionChange();
if (pinChange) {
  out.push('## WPT revision');
  out.push('');
  out.push(`${pinChange.oldSha} -> ${pinChange.newSha}`);
  out.push('');
}

const allBackends = new Set([...newFailures.keys(), ...newPasses.keys()]);

if (allBackends.size > 0) {
  out.push('## Summary');
  out.push('');
  out.push('| Backend | New failures | New passes |');
  out.push('|---------|--------------|------------|');
  for (const backend of [...allBackends].sort()) {
    const f = newFailures.get(backend)?.size ?? 0;
    const p = newPasses.get(backend)?.size ?? 0;
    out.push(`| ${backend} | ${f} | ${p} |`);
  }
  out.push('');
}

// Totals of the runs themselves, not of the diff: the Summary table above
// already accounts for the entries this sync adds or removes.
const { totals, notes } = passTotals();
if (totals.length > 0 || notes.length > 0) {
  out.push('## Stats');
  out.push('');
  if (totals.length > 0) {
    out.push('| Backend | Passed | Skipped | Failed |');
    out.push('|---------|--------|---------|--------|');
    for (const t of totals) {
      out.push(`| ${t.backend} | ${t.passed} | ${t.skipped} | ${t.failed} |`);
    }
    out.push('');
  }
  if (notes.length > 0) {
    out.push(...notes);
    out.push('');
  }
}

function listTransitions(title, backendSets) {
  const items = [];
  for (const backend of [...backendSets.keys()].sort()) {
    for (const name of [...backendSets.get(backend)].sort()) {
      items.push(`- \`${backend}\` ${name}`);
    }
  }
  if (items.length === 0) return;
  out.push(`## ${title}`);
  out.push('');
  out.push(...items);
  out.push('');
}

listTransitions('New Passes', newPasses);
listTransitions('New Failures', newFailures);

console.log(out.join('\n'));
