import assert from 'node:assert/strict';
import {test} from 'node:test';
import {normalizeSourceTolerance} from './source-tolerance.mjs';

const graph = (dataType, data) => ({expectedOutputs: {result: {descriptor: {dataType}, data}}});

for (const [type, values] of [
  ['int4', [-8, 0, 7]], ['uint4', [0, 15]],
  ['int8', [-128, 0, 127]], ['uint8', [0, 255]],
  ['int32', [-2147483648, -16777217, 0, 16777217, 2147483647]],
  ['uint32', [0, 16777217, 4294967295]],
]) {
  for (const value of [undefined, NaN]) {
    test(`${String(value)} ULP budget preserves exact ${type} array comparison`, () => {
      assert.deepEqual(normalizeSourceTolerance({metricType: 'ULP', value}, graph(type, values)),
          {metricType: 'ULP', value: 0});
    });
  }
}

test('finite, infinite and coercible source budgets are unchanged', () => {
  for (const value of [0, 2, -1, 0.5, Infinity, -Infinity, null, '0', 'NaN', false]) {
    const source = {metricType: 'ULP', value};
    assert.strictEqual(normalizeSourceTolerance(source, graph('int32', [1])), source);
  }
});

test('normalize multiple integer outputs without mutating the source budget', () => {
  const input = {expectedOutputs: {
    signed: {descriptor: {dataType: 'int32'}, data: [-16777217]},
    unsigned: {descriptor: {dataType: 'uint32'}, data: [4294967295]},
  }};
  for (const value of [undefined, NaN]) {
    const source = Object.freeze({metricType: 'ULP', value});
    assert.deepEqual(normalizeSourceTolerance(source, input), {metricType: 'ULP', value: 0});
    assert.strictEqual(source.value, value);
  }
});

test('missing callbacks and non-ULP metrics remain invalid', () => {
  for (const source of [null, undefined, {}, {metricType: 'ATOL'}, {metricType: 'other'}, {metricType: 'ATOL', value: NaN}]) {
    assert.strictEqual(normalizeSourceTolerance(source, graph('int32', [1])), source);
  }
});

test('do not invent budgets for floats, BigInt coercion, scalars or malformed expectations', () => {
  const sparse = [1, , 3];
  for (const input of [
    graph('float32', [0.1]), graph('float16', [1]), graph('int64', [1n]),
    graph('uint64', [1n]), graph('int32', 1), graph('int32', [1n]),
    graph('int32', [0.5]), graph('int32', [2147483648]), graph('uint32', [-1]),
    graph('int8', [128]), graph('uint4', [16]), graph('int4', [-9]),
    graph('int32', [NaN]), graph('int32', sparse), graph('unknown', [1]),
    {}, {expectedOutputs: {}},
    {expectedOutputs: {...graph('int32', [1]).expectedOutputs,
      other: {descriptor: {dataType: 'float32'}, data: [1]}}},
  ]) {
    for (const value of [undefined, NaN]) {
      const missing = {metricType: 'ULP', value};
      assert.strictEqual(normalizeSourceTolerance(missing, input), missing);
    }
  }
});
