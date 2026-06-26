#!/usr/bin/env node
/**
 * Shallow sparse-clone or update the Web Platform Tests repository for WebNN conformance.
 *
 * Only fetches `interfaces/` and `webnn/` (not the full ~160k-file WPT tree).
 * Same approach as webnnjs/scripts/fetch-wpt.mjs.
 *
 * Usage: node scripts/fetch_wpt.mjs
 * Env: WPT_DIR (default: .cache/wpt under repo root)
 */
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const cacheDir = path.join(repoRoot, '.cache');
const wptDir = process.env.WPT_DIR ?? path.join(cacheDir, 'wpt');
const repo = 'https://github.com/web-platform-tests/wpt.git';

// rustnn WPT harness only needs WebNN conformance tests under webnn/; interfaces/ kept for parity with webnnjs.
const SPARSE_CONE_PATHS = ['interfaces', 'webnn'];

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

await mkdir(cacheDir, { recursive: true });

const hasGitRepo = existsSync(path.join(wptDir, '.git'));

if (!hasGitRepo) {
  console.log(`Cloning WPT (sparse: ${SPARSE_CONE_PATHS.join(', ')}) into ${wptDir}...`);
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
} else {
  console.log(`Updating WPT in ${wptDir}...`);
  await run(
    'git',
    ['fetch', '--depth', '1', '--filter=blob:none', 'origin', 'master'],
    wptDir
  );
  await ensureSparseCheckout();
  await run('git', ['reset', '--hard', 'origin/master'], wptDir);
}

const conformanceDir = path.join(wptDir, 'webnn', 'conformance_tests');
if (!existsSync(conformanceDir)) {
  console.error(`Missing ${conformanceDir} after fetch`);
  process.exit(1);
}

console.log('WPT ready.');
