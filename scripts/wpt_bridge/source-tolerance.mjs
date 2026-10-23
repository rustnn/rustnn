const integerRanges = new Map([
  ['int4', [-8, 7]], ['uint4', [0, 15]],
  ['int8', [-128, 127]], ['uint8', [0, 255]],
  ['int32', [-2147483648, 2147483647]], ['uint32', [0, 4294967295]],
]);

// WPT's ULP comparator accepts equal primitive values before consulting its
// budget. An undefined or NaN budget then rejects every unequal integer pair.
// Encode that exact comparison, not a guessed operator tolerance. Infinity is
// different: it accepts unequal values and must remain unchanged. Restrict this to
// represented Number-valued integer arrays: floating-point rounding, scalar
// wrapping and Number/BigInt coercion have different JavaScript semantics.
export function normalizeSourceTolerance(tolerance, graph) {
  if (tolerance?.metricType !== 'ULP' ||
      (tolerance.value !== undefined && !Number.isNaN(tolerance.value))) {
    return tolerance;
  }
  const outputs = Object.values(graph?.expectedOutputs ?? {});
  if (outputs.length === 0 || !outputs.every(output => {
    const range = integerRanges.get(output?.descriptor?.dataType);
    const values = output?.data;
    if (!range || !Array.isArray(values)) return false;
    for (let index = 0; index < values.length; index++) {
      if (!Object.hasOwn(values, index) || !Number.isSafeInteger(values[index]) ||
          values[index] < range[0] || values[index] > range[1]) return false;
    }
    return true;
  })) {
    return tolerance;
  }
  return {...tolerance, value: 0};
}
