#!/usr/bin/env node
/**
 * Extract WPT Test Data from JavaScript Files
 *
 * This script uses Node.js to properly parse JavaScript test files
 * and extract test case data as JSON.
 *
 * Usage:
 *   node scripts/extract_wpt_tests.js <test-file.js>
 */

const fs = require('fs');
const path = require('path');

function extractTests(filePath) {
    try {
        // Read the JavaScript file
        let content = fs.readFileSync(filePath, 'utf8');

        // Extract the test array name (e.g., "reluTests", "addTests", etc.)
        const arrayMatch = content.match(/const\s+(\w+Tests)\s*=\s*\[/);
        if (!arrayMatch) {
            console.error('No test array found in file');
            process.exit(1);
        }

        const arrayName = arrayMatch[1];

        // Remove the test execution line at the end (webnn_conformance_test(...))
        // This line references utilities that we don't need for data extraction
        content = content.replace(/\nwebnn_conformance_test\([^)]*\);?\s*$/, '');

        // Custom JSON.stringify replacement that handles BigInt
        const originalStringify = JSON.stringify;
        JSON.stringify = function(value) {
            return originalStringify(value, (key, val) =>
                typeof val === 'bigint' ? val.toString() + 'n' : val
            );
        };

        // Wrap the content to capture the test array
        const wrappedContent = `
            'use strict';
            ${content}
            JSON.stringify(${arrayName});
        `;

        // Use eval in a controlled context
        // Note: This is safe here because we're parsing known WPT test files
        const result = eval(wrappedContent);

        // Restore original stringify
        JSON.stringify = originalStringify;

        // Output the JSON
        console.log(result);

    } catch (error) {
        console.error(`Error extracting tests: ${error.message}`);
        process.exit(1);
    }
}

// Main execution
if (process.argv.length < 3) {
    console.error('Usage: node extract_wpt_tests.js <test-file.js>');
    process.exit(1);
}

const testFile = process.argv[2];
if (!fs.existsSync(testFile)) {
    console.error(`File not found: ${testFile}`);
    process.exit(1);
}

extractTests(testFile);
