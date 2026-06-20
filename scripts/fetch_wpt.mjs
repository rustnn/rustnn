#!/usr/bin/env node
/**
 * Shallow-clone or update the Web Platform Tests repository for WebNN conformance.
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

function run(cmd, args, cwd = repoRoot) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, stdio: 'inherit' });
    p.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} failed with code ${code}`));
    });
  });
}

await mkdir(cacheDir, { recursive: true });

if (!existsSync(wptDir)) {
  console.log(`Cloning WPT into ${wptDir}...`);
  await run('git', ['clone', '--depth', '1', repo, wptDir]);
} else {
  console.log(`Updating WPT in ${wptDir}...`);
  await run('git', ['fetch', '--depth', '1', 'origin', 'master'], wptDir);
  await run('git', ['reset', '--hard', 'origin/master'], wptDir);
}

const conformanceDir = path.join(wptDir, 'webnn', 'conformance_tests');
if (!existsSync(conformanceDir)) {
  console.error(`Missing ${conformanceDir} after fetch`);
  process.exit(1);
}

console.log('WPT ready.');
