#!/usr/bin/env node
/**
 * Shallow sparse-clone or update the Web Platform Tests repository for WebNN conformance.
 *
 * The checked-out revision is pinned by the `WPT_REVISION` file at the repo root,
 * so CI always tests against a known-good WPT commit. The scheduled sync workflow
 * updates the pin (and the expected-failures lists) via a PR.
 *
 * Only fetches `interfaces/` and `webnn/` (not the full ~160k-file WPT tree).
 *
 * Usage: node scripts/fetch_wpt.mjs
 * Env: WPT_DIR (default: .cache/wpt under repo root)
 */
import { existsSync, readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const cacheDir = path.join(repoRoot, '.cache');
const wptDir = process.env.WPT_DIR ?? path.join(cacheDir, 'wpt');
const repo = 'https://github.com/web-platform-tests/wpt.git';
const revisionFile = path.join(repoRoot, 'WPT_REVISION');

// rustnn WPT harness only needs WebNN conformance tests under webnn/; interfaces/ kept for parity with webnnjs.
const SPARSE_CONE_PATHS = ['interfaces', 'webnn'];

function pinnedRevision() {
  if (!existsSync(revisionFile)) {
    throw new Error(`Missing ${revisionFile}; expected a pinned WPT commit hash`);
  }
  const revision = readFileSync(revisionFile, 'utf8').trim();
  if (!/^[0-9a-f]{7,40}$/i.test(revision)) {
    throw new Error(`Invalid WPT revision in ${revisionFile}: "${revision}"`);
  }
  return revision;
}

function run(cmd, args, cwd = repoRoot) {
  if (cmd === 'git') {
    console.log(`> git ${args.join(' ')}`);
  }
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, stdio: 'inherit' });
    p.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} failed with code ${code}`));
    });
  });
}

async function ensureSparseCheckout() {
  await run('git', ['sparse-checkout', 'init', '--cone'], wptDir);
  await run('git', ['sparse-checkout', 'set', ...SPARSE_CONE_PATHS], wptDir);
}

async function fetchPinned(revision) {
  await run(
    'git',
    ['fetch', '--depth', '1', '--filter=blob:none', 'origin', revision],
    wptDir
  );
  await run('git', ['reset', '--hard', revision], wptDir);
}

const revision = pinnedRevision();
await mkdir(cacheDir, { recursive: true });

const hasGitRepo = existsSync(path.join(wptDir, '.git'));

if (!hasGitRepo) {
  console.log(
    `Cloning WPT (sparse: ${SPARSE_CONE_PATHS.join(', ')}) into ${wptDir}...`
  );
  await run('git', [
    'clone',
    '--depth',
    '1',
    '--filter=blob:none',
    '--sparse',
    '--single-branch',
    '--branch',
    'master',
    repo,
    wptDir,
  ]);
  await ensureSparseCheckout();
  console.log(`Checking out pinned WPT revision ${revision}...`);
  await fetchPinned(revision);
} else {
  console.log(`Updating WPT in ${wptDir} to pinned revision ${revision}...`);
  await ensureSparseCheckout();
  await fetchPinned(revision);
}

const conformanceDir = path.join(wptDir, 'webnn', 'conformance_tests');
if (!existsSync(conformanceDir)) {
  console.error(`Missing ${conformanceDir} after fetch`);
  process.exit(1);
}

console.log('WPT ready.');
