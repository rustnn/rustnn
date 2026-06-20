import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

import { createWptHarnessStubs } from './wpt-tolerance-env.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Evaluate a WPT conformance test file unchanged and capture the arguments passed
 * to webnn_conformance_test(tests, buildAndExecuteGraph, toleranceFunc).
 *
 * @param {string} sourceText
 * @param {string} [sourceName]
 * @param {{ utilsPath?: string }} [options]
 */
export function loadWptConformanceFile(
  sourceText,
  sourceName = 'wpt-test.js',
  options = {}
) {
  const utilsPath =
    options.utilsPath ??
    path.resolve(__dirname, '../../.cache/wpt/webnn/resources/utils.js');

  const registration = {
    tests: null,
    resolveTolerance: null,
    buildAndExecuteGraph: null
  };

  const sandbox = createWptHarnessStubs();

  const captureWebnnConformanceTest = (tests, buildAndExecuteGraph, toleranceFunc) => {
    if (!Array.isArray(tests)) {
      throw new TypeError(
        `${sourceName}: webnn_conformance_test first argument must be a test array`
      );
    }
    registration.tests = tests;
    registration.buildAndExecuteGraph =
      typeof buildAndExecuteGraph === 'function' ? buildAndExecuteGraph : null;
    registration.resolveTolerance =
      typeof toleranceFunc === 'function'
        ? (graphResources, intermediateOperands = {}) =>
            toleranceFunc(graphResources, intermediateOperands)
        : null;
  };

  const context = vm.createContext(sandbox);
  const utilsSource = readFileSync(utilsPath, 'utf8');
  vm.runInContext(utilsSource, context, {
    filename: path.basename(utilsPath),
    timeout: 10_000
  });

  sandbox.webnn_conformance_test = captureWebnnConformanceTest;

  try {
    vm.runInContext(sourceText, context, { filename: sourceName, timeout: 30_000 });
  } catch (err) {
    throw new Error(`Failed to evaluate ${sourceName}: ${err.message}`);
  }

  if (!registration.tests) {
    throw new Error(`No webnn_conformance_test(...) call in ${sourceName}`);
  }

  return {
    tests: registration.tests,
    resolveTolerance: registration.resolveTolerance,
    buildAndExecuteGraph: registration.buildAndExecuteGraph
  };
}
