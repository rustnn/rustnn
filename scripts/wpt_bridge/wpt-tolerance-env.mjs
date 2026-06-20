import { readFileSync } from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

/** Minimal WPT harness stubs so utils.js tolerance helpers can load in Node. */
export function createWptHarnessStubs() {
  const noop = () => {};
  const asyncNoop = async () => {};
  const sandbox = {
    globalThis: null,
    Float32Array,
    Float16Array,
    Int8Array,
    Uint8Array,
    Int16Array,
    Uint16Array,
    Int32Array,
    Uint32Array,
    BigInt64Array,
    BigUint64Array,
    BigInt,
    Number,
    Math,
    Array,
    Object,
    String,
    Error,
    TypeError,
    AssertionError: Error,
    URLSearchParams,
    location: { search: '' },
    navigator: { ml: { createContext: asyncNoop } },
    fetch: asyncNoop,
    test: noop,
    promise_test: noop,
    promise_setup: noop,
    assert_implements: noop,
    assert_equals: noop,
    assert_array_equals: noop,
    assert_less_than_equal: noop,
    assert_array_approx_equals_ulp: noop,
    assert_array_approx_equals: noop
  };
  sandbox.globalThis = sandbox;
  return sandbox;
}

/**
 * Load WPT utils.js tolerance helpers (getPrecisionTolerance, getZeroULPTolerance, …).
 * @param {string} utilsPath
 */
export function createWptToleranceContext(utilsPath) {
  const sandbox = createWptHarnessStubs();
  const context = vm.createContext(sandbox);
  const source = readFileSync(utilsPath, 'utf8');
  vm.runInContext(source, context, {
    filename: path.basename(utilsPath),
    timeout: 10_000
  });
  return { context, sandbox };
}
