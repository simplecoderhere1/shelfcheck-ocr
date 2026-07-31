/**
 * One place that decides which ground-truth file the harnesses read.
 *
 * v2 listed roughly half the books physically on these shelves, so every recall
 * and precision figure measured against it was measuring the key. v3 is the
 * rebuilt one (see _gt/RESUME.md). Prefer v3 when it exists, fall back to v2
 * while the rebuild is still in progress, and SAY WHICH on stderr every time —
 * a number quoted without knowing its denominator is how this went wrong the
 * first time.
 *
 *   import { loadGT, GT_FILE } from './gt_load.mjs';
 *
 * Override with  GT=ground_truth_v2.json node <harness>
 */
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));

function pick() {
  if (process.env.GT) return process.env.GT;
  if (existsSync(join(HERE, 'ground_truth_v3.json'))) return 'ground_truth_v3.json';
  return 'ground_truth_v2.json';
}

export const GT_FILE = pick();

export function loadGT(quiet = false) {
  const p = join(HERE, GT_FILE);
  if (!existsSync(p)) {
    console.error(`ground truth not found: ${p}`);
    process.exit(1);
  }
  if (!quiet) {
    const warn = GT_FILE.includes('v2')
      ? '  ** v2 is INCOMPLETE — it lists about half the books; recall is flattered and precision is meaningless **'
      : '';
    console.error(`[gt] ${GT_FILE}${warn}`);
  }
  return JSON.parse(readFileSync(p, 'utf-8'));
}
