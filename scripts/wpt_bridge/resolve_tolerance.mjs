// Resolve a source callback after the Rust builder has inferred intermediate shapes.
// Do not duplicate WPT's tolerance formulas or infer shapes from expected values.
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {loadWptConformanceFile} from './load-wpt-file.mjs';

const [wptDir, fileName, testName] = process.argv.slice(2);
const jsPath = path.join(wptDir, 'webnn/conformance_tests', fileName);
const loaded = loadWptConformanceFile(readFileSync(jsPath, 'utf8'), fileName, {
  utilsPath: path.join(wptDir, 'webnn/resources/utils.js')
});
const test = loaded.tests.find(test => test.name === testName);
if (!test || !loaded.resolveTolerance) {
  throw new Error(`No upstream tolerance callback for ${fileName}: ${testName}`);
}
const intermediates = JSON.parse(readFileSync(0, 'utf8'));
const tolerance = loaded.resolveTolerance(test.graph, intermediates);
if (!tolerance || !Number.isFinite(tolerance.value)) {
  throw new Error(`Upstream tolerance callback returned no finite budget for ${fileName}: ${testName}`);
}
process.stdout.write(JSON.stringify(tolerance));
