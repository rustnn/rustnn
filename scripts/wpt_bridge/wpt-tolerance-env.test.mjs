import assert from 'node:assert/strict';
import {test} from 'node:test';
import vm from 'node:vm';
import {createWptHarnessStubs} from './wpt-tolerance-env.mjs';

test('fixtures can clone shared graphs without changing numeric values or the original', () => {
  const context = vm.createContext(createWptHarnessStubs());
  const result = vm.runInContext(`(() => {
    const original = {
      data: [-0, NaN, Infinity, 1.00048828125 + 2 ** -40],
      tensor: new Float32Array([-0, 2 ** -149]),
      options: {epsilon: 0.125},
    };
    const copy = structuredClone(original);
    copy.options.epsilon = 0.25;
    return {original, copy};
  })()`, context);
  assert.equal(result.original.options.epsilon, 0.125);
  assert.equal(result.copy.options.epsilon, 0.25);
  assert.ok(Object.is(result.copy.data[0], -0));
  assert.ok(Number.isNaN(result.copy.data[1]));
  assert.equal(result.copy.data[2], Infinity);
  assert.equal(result.copy.data[3], 1.00048828125 + 2 ** -40);
  assert.ok(Object.is(result.copy.tensor[0], -0));
  assert.equal(result.copy.tensor[1], 2 ** -149);
  assert.notStrictEqual(result.copy.tensor.buffer, result.original.tensor.buffer);
});
