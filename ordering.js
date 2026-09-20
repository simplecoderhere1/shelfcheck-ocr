// Deterministic order-checking logic for ShelfCheck.
//
// Extracted verbatim from the pre-rewrite index.html (lines 6380-7809 of the
// commit tagged "pre-vision-rebuild") when the OCR/assembly pipeline around it
// was replaced with a single vision-model read per shelf-segment photo. This
// code compares ALREADY-READ book labels to decide shelving order — it was not
// the source of the app's real-shelf failures (those were in Azure OCR box
// assembly) and is kept because it is validated, hard-won logic: the LIS-based
// "longest correct run" approach plus confidence-gated demotions is what
// enforces the cardinal invariant "never accuse a correctly-shelved book."
//
// Input shape per book (see labelsToBooks in the old pipeline, and reader.js's
// readShelfSegment in the new one): { spine_label, title, title_verified,
// title_conf, confidence, shelfRow, _bbox: [x,y,w,h], _score, _truncated,
// _unclear, _malformed, _ambiguousTail, _noGiven, _gem, _inserted, _src }.
// A clean single-pass vision read will rarely set most of the underscore
// hooks (_malformed, _ambiguousTail, ...) — they were tripped by OCR/assembly
// artifacts a direct read mostly avoids — but the fields are left in place so
// the same safety net still catches the vision model's own low-confidence or
// garbled reads without requiring a second engine.

// See the RED_MIN_SCORE derivation below: 0.57 is the measured knee where a
// real detection ("WEISBERGER Lauren" at 0.59) is promoted to red without
// promoting any confirmed-correct book. That sweep is BLIND to this bar's real
// cost, because order_fixture.json does not carry the low-confidence books the
// bar actually decides. Re-swept 2026-08-23 over all 36 shelves of
// app_fixture.json (`_bar_sweep.mjs`), scoring known misfiles promoted to red
// against must_not_flag books wrongly accused:
//
//   bar     misfiles RED   guard violations   total reds
//   0.60         11               0               15
//   0.57         12               0               16     <- the only new red
//   0.55         12               0               17        IS the true one
//   0.50         12               0               18
//
// 0.57 is free: it promotes exactly one book and that book is a real
// misfile ("WEISBERGER Lauren", read as precisely the right text, genuinely
// out of order, scored 0.59 — a hundredth of a point deciding whether a
// volunteer is sent to it). Below 0.57 the red count keeps climbing while
// detections do not, so those extra flags are unverified and not worth it.
// Do not read this as "confidence works" — this file records repeatedly that
// a misread comes back as confident as a good read. The bar bounds damage;
// it does not identify truth.
export const RED_MIN_SCORE = 0.57;

export const SURNAME_PREFIXES = new Set([
  'de','del','dela','di','du','la','le','les','lo','mac','mc',
  'o','van','von','st','san','santa','el','al','bin','ibn','da','das','dos',
]);

function lettersLower(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9]/g, ''); }

// Multi-word surnames keep a space separator so comparison can file
// word-by-word (library standard): "VON ZIEGESAR" files before "VONNEGUT"
// because the first word "von" ends before "vonnegut" continues.
function buildMultiWordSurname(tokens, start) {
  const words = []; let i = start;
  while (i < tokens.length - 1 && SURNAME_PREFIXES.has(tokens[i].toLowerCase())) {
    words.push(tokens[i].toLowerCase().replace(/[^a-z]/g, '')); i++;
  }
  if (i < tokens.length) words.push(tokens[i].toLowerCase().replace(/[^a-z]/g, ''));
  const s = words.filter(Boolean).join(' ');
  return s || null;
}

// Word-by-word alphabetical compare: a space files before any letter.
export function compareFiling(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] === b[i]) continue;
    if (a[i] === ' ') return -1;
    if (b[i] === ' ') return 1;
    return a[i] < b[i] ? -1 : 1;
  }
  return a.length - b.length;
}

// Are these two names the same name, one of them misread?
//
// Nonfiction protects itself from this already: a Dewey number Azure reports
// as cut off is tied against a longer sibling rather than compared, because
// digits that are not in the photograph cannot decide shelf position. Fiction
// had no equivalent and it is where every remaining false accusation comes
// from — not from bad ordering logic, but from the last letter of a surname.
//
// Libraries shelve an author's books together, so neighbours routinely share
// a surname, and then the ONLY thing separating them is the part of the read
// that is least reliable: WALKER/WALKET/WALKEN/WALKED off four adjacent
// spines, or WAITES "Maryn" against WAITES "Martyn". One substituted letter
// reverses the order and the shelf looks wrong when it is perfectly right.
//
// So: treat a difference that a single OCR slip explains as no evidence at
// all. Two forms count as the same name when the shorter is a prefix of the
// longer (the read stopped early) or one single-character edit turns one into
// the other — a letter read wrong, dropped, or invented ("Maryn"/"Martyn" is
// a dropped t, not a swapped one, and a substitution-only test misses it).
// Minimum lengths keep short scraps like "WAL" from swallowing every surname.
//
// This loses detections — WALKER really shelved among WALTER now passes — and
// that is the intended direction. A missed misfiling leaves a volunteer where
// they were; a false one sends them to the wrong book.
export const NAME_PREFIX_MIN = 4;   // shorter than this, a prefix proves nothing
export const NAME_EDIT_MIN   = 5;   // shorter than this, one letter is most of the word
export function nameSlip(a, b) {
  const norm = s => String(s || '').toLowerCase().replace(/[^a-z]/g, '');
  const x = norm(a), y = norm(b);
  if (!x || !y || x === y) return false;        // equal names compare normally
  const [s, l] = x.length <= y.length ? [x, y] : [y, x];
  if (l.startsWith(s)) return s.length >= NAME_PREFIX_MIN;
  if (s.length < NAME_EDIT_MIN || l.length - s.length > 1) return false;
  // One edit, checked by walking both together and allowing a single skip.
  let i = 0, j = 0, edits = 0;
  while (i < s.length && j < l.length) {
    if (s[i] === l[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (s.length === l.length) { i++; j++; } else { j++; }   // substitute / drop
  }
  return edits + (l.length - j) + (s.length - i) === 1;
}

// A given-name sibling of nameSlip, deliberately more lenient. nameSlip's
// floors (NAME_PREFIX_MIN/NAME_EDIT_MIN) are tuned for SURNAMES, which run
// long enough that a short match is genuinely uninformative. Given names
// are shorter by nature ("Brad", "Meg", "Jim") and just as unreliably
// read, so those same floors reject an obvious one-letter slip ("Brad" /
// "Bred", 4 letters, below NAME_EDIT_MIN) as a real disagreement — which
// then wrongly broke the corroborated-surname guard above and un-excused
// a correctly shelved "TAYLOR Brad" (measured: replay_order regression).
// Used only to decide whether two given names plainly DISAGREE (a strong
// veto), never to decide they match on its own, so leaning permissive
// here is the safe direction — it can only make the app cautious less
// often, never accuse a name it isn't sure about.
export function givenSlip(a, b) {
  const norm = s => String(s || '').toLowerCase().replace(/[^a-z]/g, '');
  const x = norm(a), y = norm(b);
  if (!x || !y || x === y) return true;
  const [s, l] = x.length <= y.length ? [x, y] : [y, x];
  if (l.startsWith(s)) return true;
  if (l.length - s.length > 1) return false;
  let i = 0, j = 0, edits = 0;
  while (i < s.length && j < l.length) {
    if (s[i] === l[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (s.length === l.length) { i++; j++; } else { j++; }
  }
  return edits + (l.length - j) + (s.length - i) <= 1;
}


// Dewey number compare: integer part numerically, then decimal digits
// positionally, shorter decimal first — standard shelf filing order
// (635 < 635.01 < 635.04 < 635.0483 < 635.1).
export function compareDeweyNums(aStr, bStr) {
  const [ai, af = ''] = String(aStr).split('.');
  const [bi, bf = ''] = String(bStr).split('.');
  const d = parseInt(ai, 10) - parseInt(bi, 10);
  if (d !== 0) return d;
  const n = Math.min(af.length, bf.length);
  for (let i = 0; i < n; i++) {
    if (af[i] !== bf[i]) return af.charCodeAt(i) - bf.charCodeAt(i);
  }
  return af.length - bf.length;
}

export function parseSurnameAndFirst(s) {
  if (!s) return null;
  let str = String(s).trim().replace(/\s*\d+\s*$/, '').trim();
  if (!str) return null;
  if (str.includes(',')) {
    const parts = str.split(',').map(p => p.trim()).filter(Boolean);
    const sTokens = parts[0].split(/\s+/).filter(Boolean);
    const surname = buildMultiWordSurname(sTokens, 0) || lettersLower(parts[0]);
    const first   = parts[1] ? lettersLower(parts[1]) : '';
    if (!surname) return null;
    return { surname, first };
  }
  let tokens = str.split(/\s+/).filter(Boolean);
  if (!tokens.length) return null;
  while (tokens.length > 1 && /^(jr|sr|ii|iii|iv|v|esq)\.?$/i.test(tokens[tokens.length - 1])) tokens.pop();
  if (tokens.length === 1) return { surname: lettersLower(tokens[0]), first: '' };
  const t0 = tokens[0], t1 = tokens[1];
  // Surname-first heuristic: all-caps first token followed by a token that
  // is NOT all-caps (mixed case, all-lowercase, or an initial like "s.").
  // Requiring [A-Z][a-z] here missed real stickers like "WACHMAN sam",
  // which then fell through to First-Last parsing → surname "sam".
  // "KURT VONNEGUT" (both all-caps) still takes the last-token path.
  if (/^[A-Z]{2,}$/.test(t0) && !/^[A-Z]{2,}\.?$/.test(t1)) {
    const surname = buildMultiWordSurname(tokens.slice(0, -1), 0) || lettersLower(t0);
    return { surname, first: lettersLower(tokens[tokens.length - 1]) };
  }
  let surnameStart = tokens.length - 1;
  while (surnameStart > 1 && SURNAME_PREFIXES.has(tokens[surnameStart - 1].toLowerCase())) surnameStart--;
  const surname = buildMultiWordSurname(tokens, surnameStart);
  const first   = lettersLower(tokens.slice(0, surnameStart).join(''));
  return { surname: surname || lettersLower(tokens[tokens.length - 1]), first };
}

export function parseDewey(s) {
  if (!s) return null;
  const m = s.match(/^\s*([0-9]+(?:\.[0-9]+)?|[A-Za-z]+)\s*([A-Za-z]+)?/i);
  if (!m) return null;
  const num    = isNaN(parseFloat(m[1])) ? null : parseFloat(m[1]);
  const numStr = num === null ? null : m[1];
  const prefix = m[1].toLowerCase();
  let cut = (m[2] || '').toLowerCase();
  if (num === null) cut = (prefix + cut).replace(/[^a-z]/g, '');
  return { num, numStr, cut, prefix };
}

export function extractVolume(book) {
  if (typeof book.series_volume === 'number' && isFinite(book.series_volume)) return book.series_volume;
  if (book.spine_label) { const m = String(book.spine_label).match(/(\d+)\s*$/); if (m) return parseInt(m[1], 10); }
  return null;
}

export function sortKey(book, section) {
  // A confident on-device read (AR local-only cycle: score ~0.6, below the
  // 0.65 cross-engine bar so tagged confidence:'low') is still a perfectly
  // good SORT key — it just can't earn a red flag on its own. Let it sort so
  // a correctly-ordered on-device read renders GREEN instead of a wall of
  // yellow; the red-vs-yellow gate lives in checkOrderOneShelf's final pass.
  // Genuinely weak reads (score < 0.5, e.g. the 0.4 low-confidence tier) and
  // score-less "uncertain" GT rows (_score undefined) stay unsortable/yellow.
  // ...or when the VISION MODEL read the same label independently.
  //
  // The score is the MINIMUM confidence across a label's words, so one shaky
  // glyph sinks an otherwise perfect read. Measured on the audit corpus, two
  // of the four misfilings the app finds but will not call are exactly that:
  // "WEISBERGER, Lauren" at 0.30 and "TATA, A.J." at 0.23, both
  // character-for-character correct, both left unsortable and shown amber.
  //
  // `_gem` means a second, independent engine read the same shelf and agreed
  // on this label. That is corroboration, not a lowered bar -- the same
  // reasoning as _numCorroborated above, and the general rule holds on any
  // shelf: two engines agreeing beats one engine's confidence number. The
  // red bar (RED_MIN_SCORE) is applied separately, so this lets a book be
  // sorted and judged; it does not by itself make it red.
  if (book.confidence === 'low' && !(book._score >= 0.5) &&
      !book._numCorroborated && !book._gem) return null;
  const vol   = extractVolume(book);
  // Titles must never tie-break unless human/OL-verified: Gemini titles are
  // frequently unverified (verifyTitlesOpenLibrary's timeout/budget-abort
  // path keeps the raw title with title_verified=false), and unverified
  // titles feeding compareKeys produced false "out of order" flags on
  // same-author runs. Both fiction and nonfiction now use the title only when
  // OL-verified; unverified titles resolve to '' and compareKeys treats an
  // empty title on either side as tied (no flag). For same-author fiction the
  // tie-break order is series/volume first (vol, extracted from the spine),
  // then verified title A-Z, so publication/series-ordered runs are not
  // flagged unless a verified title genuinely places a book out of sequence.
  // Titles stay on the label objects for display either way.
  const title = book.title_verified
    ? lettersLower((book.title || '').replace(/^the\s+|^a\s+|^an\s+/i, ''))
    : '';
  function strKey(s) {
    const a = parseSurnameAndFirst(s);
    if (!a) return null;
    let surname = a.surname, first = a.first;
    // Fiction spine labels lead with the surname ("WALLACE, David Foster").
    // With the comma present the parser gets this right. When OCR drops the
    // comma — which is common, it's a tiny mark on a small sticker — the
    // parser falls back to reading the LAST word as the surname, so
    // "VONNEGUT Kurt" sorts under "kurt" and "VON KREISLER Kristin" under
    // "kristin". Those land nowhere near where the book physically sits, and
    // a run of them drags the longest-in-order chain off its true path until
    // a perfectly shelved neighbour is the thing that looks wrong (measured:
    // "VODOLAZKIN" flagged red with an ideal slot of 9 while sitting first
    // and correct). With no comma to go on, trust the shelf's own convention
    // and take the leading word(s) as the surname.
    const raw = String(s).trim();
    if (!raw.includes(',')) {
      const words = raw.split(/\s+/).filter(w => /[A-Za-z]/.test(w));
      if (words.length > 1) {
        const norm = t => t.toLowerCase().replace(/[^a-z ]/g, '').replace(/\s+/g, ' ').trim();
        // "All but the last word" is only right for a two-word label. On a
        // three-word one it puts a given name into the surname: "WALKER Karen
        // Thomp" filed under "walker karen", which sits nowhere near the
        // WALKER run it belongs to and made the book look misshelved.
        //
        // The sticker itself says where the split is, in the same convention
        // labelPlausible already relies on: the surname line is CAPITALISED
        // and the given name is not. Take the leading run of capitalised words
        // as the surname. When that run is the whole label there is no given
        // name visible to separate, so fall back to the old rule rather than
        // leaving the given name empty.
        const isCaps = w => /[A-Za-z]/.test(w) && w === w.toUpperCase();
        let k = 0;
        while (k < words.length && isCaps(words[k])) k++;
        if (k === 0 || k === words.length) k = words.length - 1;
        const lead = norm(words.slice(0, k).join(' '));
        // The given name is joined WITHOUT a space, because that is what the
        // comma path produces ("shirleyrussak"). Keeping the space made the
        // same name in two spellings compare unequal — compareFiling sorts a
        // space before any letter — so "WACHTEL, Shirley Russak" and "WACHTEL
        // Shirley Russak", one book each, accused each other. The surname does
        // keep its spaces: the comma path spaces those too ("von kreisler").
        if (lead) { surname = lead; first = norm(words.slice(k).join('')); }
      }
    }
    return { kind: 'str', val: surname, first, vol, title };
  }
  if (section === 'fiction')    return book.spine_label ? strKey(book.spine_label) : null;
  if (section === 'nonfiction') {
    const d = parseDewey(book.spine_label);
    // A number read with NO cutter (d.cut === '', not null — parseDewey
    // always returns a string) sorts before every real cutter under
    // localeCompare, a fabricated tie-break with no basis in the sticker.
    // Treat it exactly like any other unparseable label (null key -> the
    // existing 'nokey'/unclear path in checkOrderOneShelf): it is counted
    // and shown as unreadable, but never sorted and never red. This is what
    // makes it safe for the assembly stage to stop DELETING these labels
    // (the reason they used to vanish rather than show as "couldn't read")
    // without opening a new false-positive path.
    return d && d.cut ? { kind: 'dewey', val: d, vol, title, trunc: !!book._truncated } : null;
  }
  const d = parseDewey(book.spine_label);
  if (d && d.num !== null) return { kind: 'dewey', val: d, vol, title };
  return book.spine_label ? strKey(book.spine_label) : null;
}

// Set fresh per shelf in checkOrderOneShelf, read by compareKeys below.
// Counts how many entries on THIS shelf carry each exact surname string —
// see the corroboration check where it's used.
let CURRENT_SURNAME_COUNTS = null;
// Where each fully-read author name ("murray jamess") sits on the current
// shelf. Built beside CURRENT_SURNAME_COUNTS; read only by the sameauthor
// pass, to tell a repeated real name from a one-off garbled one.
let CURRENT_AUTHOR_POS = null;

export function compareKeys(a, b) {
  if (a.kind !== b.kind) return a.kind === 'dewey' ? -1 : 1;
  let cmp;
  if (a.kind === 'str') {
    // Undecidable rather than out of order — see nameSlip(). But not when
    // BOTH of two things hold: the surname is independently CORROBORATED
    // (read as its own complete, exact string more than once on this
    // shelf), AND the given names plainly disagree. Corroboration alone
    // is not enough to break the tie — a real same-author run legitimately
    // repeats its correct surname while ALSO containing a genuinely
    // garbled copy of it ("WALKER" x2 alongside a misread "WALKEN"), and
    // requiring only corroboration flagged that innocent run (measured:
    // name_slip_test regression). Given names alone aren't enough either —
    // they're read even less reliably than surnames and a bare truncation
    // ("Eud" for "Eudora") would misfire (see the TRIED AND REVERTED note
    // below in checkOrderOneShelf). Together they're specific: "CARR,
    // Robyn" x11 next to "CARREY, Jim" has a corroborated surname AND an
    // unrelated given name — that pair is two different real people, not
    // one misread (measured live, 2026-08-23).
    const slip = nameSlip(a.val, b.val);
    // Scoped to a bare one-word surname on both sides — see the matching
    // note in checkOrderOneShelf's sameauthor pass. A multi-word garble
    // ("taylor blakem" for "Taylor") looks like a one-word prefix too,
    // once nameSlip strips its internal space, but it is a mangled read
    // of ONE sticker regardless of corroboration or given names.
    const bothSingleWord = a.val.indexOf(' ') === -1 && b.val.indexOf(' ') === -1;
    const corroborated = bothSingleWord && CURRENT_SURNAME_COUNTS && (
      (CURRENT_SURNAME_COUNTS.get(a.val) || 0) >= 2 ||
      (CURRENT_SURNAME_COUNTS.get(b.val) || 0) >= 2);
    const namesDisagree = bothSingleWord && a.first && b.first &&
      a.first !== b.first && !givenSlip(a.first, b.first);
    if (slip && !(corroborated && namesDisagree)) return 0;
    cmp = compareFiling(a.val, b.val);
  } else {
    if (a.val.numStr === null || b.val.numStr === null) {
      cmp = (a.val.numStr === null) === (b.val.numStr === null) ? 0
          : (a.val.numStr === null ? 1 : -1);
    } else {
      const as = String(a.val.numStr), bs = String(b.val.numStr);
      // A bare read ("635") next to a decimal read with the same integer
      // part ("635.04") is more often a truncated read than a really-bare
      // sticker, and bare-sorts-first would then flag the correctly-read
      // neighbor. Treat the pair as fully tied (no cutter tiebreak — the
      // two books' cutters aren't comparable across that uncertainty).
      if (as.includes('.') !== bs.includes('.') &&
          parseInt(as, 10) === parseInt(bs, 10)) return 0;
      // Treating ALL prefix pairs ("635.04" vs "635.0483") as tied was tried
      // and reverted: it suppressed the truncation false positives but also
      // erased genuine detections, because a real 635.04 HEM sitting inside a
      // 635.048x run is itself a prefix pair (TP 5 -> 1). That attempt had to
      // INFER truncation from the strings, so it could not tell a short read
      // from a genuinely short call number.
      //
      // Azure states it outright: a sticker it read as "635." with no digits
      // following is flagged truncated upstream. Only then do we tie a prefix
      // pair — the shorter read is a known-incomplete view of a number whose
      // remaining digits are not in the photograph, so its position relative
      // to a longer sibling is genuinely undecidable. Two COMPLETE reads still
      // compare strictly, which is what preserves the real detections.
      if ((a.trunc || b.trunc) && (as.startsWith(bs) || bs.startsWith(as))) return 0;
      cmp = compareDeweyNums(as, bs);
    }
    if (cmp === 0) cmp = a.val.cut.localeCompare(b.val.cut, undefined, { sensitivity: 'base' });
  }
  if (cmp !== 0) return cmp;
  if (a.kind === 'str') {
    // The surnames matched exactly, so these are two books by one author and
    // the given name decides. It is read from the same sticker and slips the
    // same way ("Maryn" for "Martyn"), so it gets the same protection.
    if (nameSlip(a.first || '', b.first || '')) return 0;
    cmp = compareFiling(a.first || '', b.first || '');
    if (cmp !== 0) return cmp;
  }
  const av = a.vol == null ? Infinity : a.vol, bv = b.vol == null ? Infinity : b.vol;
  if (av !== bv) return av - bv;
  const at = a.title || '', bt = b.title || '';
  if (!at || !bt) return 0;
  return at.localeCompare(bt, undefined, { sensitivity: 'base' });
}

export function longestNonDecreasingSubseq(keys) {
  // Guard the empty case explicitly: below, bestEnd starts at 0 assuming
  // dp[0]/prev[0] exist, but when a shelf's books all have unparseable/
  // low-confidence labels, checkOrderOneShelf calls this with keys=[]. dp
  // and prev are then empty arrays, so prev[0] is undefined rather than
  // -1, and the backtrack loop below (`i !== -1`) never terminates —
  // undefined !== -1 forever — pegging the main thread in an infinite
  // loop. This was the exact cause of the renderer hard-freezing after a
  // Gemini refine landed on a dense multi-row photo: it only takes one
  // shelf row (of any size) whose books are *all* unverifiable/low-
  // confidence to hit this.
  if (!keys.length) return new Set();
  const n = keys.length, dp = new Array(n).fill(1), prev = new Array(n).fill(-1);
  let bestEnd = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (compareKeys(keys[j], keys[i]) <= 0 && dp[j] + 1 > dp[i]) { dp[i] = dp[j] + 1; prev[i] = j; }
    }
    if (dp[i] > dp[bestEnd]) bestEnd = i;
  }
  const set = new Set();
  for (let i = bestEnd; i !== -1; i = prev[i]) set.add(i);
  return set;
}

export function groupByShelfRow(books) {
  const byRow = new Map();
  for (const b of books) {
    const row = (typeof b.shelfRow === 'number' && b.shelfRow >= 0) ? b.shelfRow : 0;
    if (!byRow.has(row)) byRow.set(row, []);
    byRow.get(row).push(b);
  }
  return [...byRow.entries()].sort((a, b) => a[0] - b[0]).map(([, shelf]) => shelf);
}

export function editDistance(a, b) {
  a = a || ''; b = b || '';
  const m = a.length, n = b.length;
  if (!m) return n; if (!n) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = cur;
  }
  return prev[n];
}

export function checkOrderOneShelf(books, section) {
  // A nonfiction label can be read PERFECTLY and still not sort, because the
  // score is the MINIMUM confidence across its words and one shaky glyph in a
  // three-letter cutter drags the whole label under the 0.5 floor. Three books
  // in the corpus carry text identical to ground truth and are shown as "no
  // sortable call number could be read" at 0.08, 0.39 and 0.47.
  //
  // Lowering the floor for everyone was measured and rejected: sorted +2 but
  // WRONG +2, trading honest amber boxes for silent misfilings. So corroborate
  // instead of trusting. Let a sub-floor label sort only when a CONFIDENT
  // label in the same row carries the identical call number, which is the
  // half of the label the sort actually turns on. That leaves the cutter as
  // the only unverified part, and a wrong cutter cannot manufacture a red
  // flag by itself -- the red bar (RED_MIN_SCORE) is applied separately and
  // still excludes these books. The exposure is that a wrong cutter drags the
  // in-order chain and makes a NEIGHBOUR look misplaced, which is exactly
  // what replay_order's must_not_flag guard is there to catch.
  const CORROB_MIN = 0.9;
  if (section === 'nonfiction') {
    const numFor = b => (/^\s*(\d{1,3}(?:\.\d+)?)(?=\s|$)/.exec(String(b.spine_label || '')) || [])[1] || null;
    const solid = new Map();
    for (const b of books) {
      const n = numFor(b);
      if (!n || !(b._score >= CORROB_MIN) || b._truncated) continue;
      const row = b.shelfRow ?? 0;
      if (!solid.has(row)) solid.set(row, new Set());
      solid.get(row).add(n);
    }
    for (const b of books) {
      // Recomputed, not accumulated: checkOrder runs more than once on the
      // same book objects, and a corroboration that held on one pass must not
      // survive into a pass where the row no longer supports it.
      b._numCorroborated = false;
      if (b._score >= 0.5 || b._truncated) continue;
      const n = numFor(b);
      if (n && solid.get(b.shelfRow ?? 0)?.has(n)) b._numCorroborated = true;
    }
  }

  const keys = books.map(b => sortKey(b, section));

  function samePrimary(a, b) {
    if (a.kind !== b.kind) return false;
    return a.kind === 'str' ? a.val === b.val && (a.first || '') === (b.first || '')
                            : a.val.num === b.val.num && a.val.cut === b.val.cut;
  }
  function primaryStr(k) { return k.kind === 'str' ? (k.val || '') : String(k.val.num ?? ''); }

  // Pre-pass: collapse re-reads of the SAME physical sticker. The two engines
  // crop slightly differently, so one label routinely lands as several
  // entries at different truncations ("VONNE" / "VONNEG" / "VONNEGUT") and as
  // leading/trailing fragments of a neighbour ("ARNIM ," sitting beside
  // "VON ARNIM, Elizabeth"). Those fragments sort nowhere near their true
  // position, and the longest-in-order run can then route THROUGH the
  // fragments and around a genuinely correct book — which is exactly how a
  // correctly shelved book gets accused (measured: "VODOLAZKIN" flagged red
  // while sitting in perfect order).
  //
  // A key that is a proper prefix or suffix of a nearby neighbour's key is a
  // truncated re-read of that neighbour, not a second book, so it's dropped
  // from ordering. EQUAL keys are deliberately left alone: a shelf really can
  // hold several books by the same author. Dropping only ever moves a book to
  // "check by hand" — it can never manufacture a flag.
  // Every decision is made against the ORIGINAL keys and applied afterwards.
  // Dropping as we scan would let an already-dropped fragment stop counting
  // as evidence that its neighbour is a fragment too, so a run of
  // progressively truncated reads only ever lost its first member.
  const DUP_WINDOW = 3;   // physical neighbours to compare against
  const dupDrops = new Set();
  for (let i = 0; i < keys.length; i++) {
    if (!keys[i] || keys[i].kind !== 'str') continue;
    const a = String(keys[i].val || '');
    if (!a) continue;
    for (let j = Math.max(0, i - DUP_WINDOW); j <= Math.min(keys.length - 1, i + DUP_WINDOW); j++) {
      if (j === i || !keys[j] || keys[j].kind !== 'str') continue;
      if ((books[j].shelfRow ?? 0) !== (books[i].shelfRow ?? 0)) continue;
      const b = String(keys[j].val || '');
      if (b.length <= a.length) continue;            // only ever drop the SHORTER read
      if (!(b.startsWith(a) || b.endsWith(a))) continue;
      // A shared prefix is NOT evidence on its own. This shelf is sorted
      // alphabetically, so genuinely different authors whose names share a
      // prefix are neighbours BY CONSTRUCTION -- WAITE beside WAITES, WAKE
      // beside WAKEFIELD, TRAN beside TRANTER. Dropping on text alone deleted
      // 145 real books across the corpus, 11% of every shelf, and sent them
      // all to "couldn't read this one".
      //
      // What separates a truncated RE-READ from a real neighbour is physical:
      // two reads of one sticker sit on top of each other, two different
      // books sit side by side. Require the boxes to overlap. With no box to
      // check, keep the old text-only behaviour rather than risk letting a
      // true fragment back into the ordering.
      const bi = books[i]._bbox, bj = books[j]._bbox;
      if (bi && bj) {
        const lo = Math.max(bi[0], bj[0]), hi = Math.min(bi[0] + bi[2], bj[0] + bj[2]);
        const ov = hi - lo;
        if (!(ov > 0 && ov >= 0.5 * Math.min(bi[2], bj[2]))) continue;
      }
      dupDrops.add(i); break;
    }
  }
  for (const i of dupDrops) keys[i] = null;

  // Pre-pass: demote obvious OCR outliers before LIS
  const misreads = [];
  for (let i = 0; i < keys.length; i++) {
    if (!keys[i]) continue;
    let L = i - 1; while (L >= 0 && !keys[L]) L--;
    let R = i + 1; while (R < keys.length && !keys[R]) R++;
    if (L < 0 || R >= keys.length) continue;
    const kL = keys[L], kR = keys[R], kM = keys[i];
    // A different entry sandwiched between two IDENTICAL author+given-name
    // reads is usually an OCR duplicate of that surrounding sticker landing
    // as a mangled reading — but not always: a real book by a different
    // author fits the same shape when it is genuinely misfiled into the
    // middle of someone else's run ("CARREY, Jim" between two "CARR,
    // Robyn" entries). Text shape alone can't tell those apart; physical
    // position can. Only demote when M's box overlaps L's or R's — two
    // reads of one sticker occupy the same spot, two different books sit
    // side by side. With no box to check, keep the old text-only call.
    if (samePrimary(kL, kR) && !samePrimary(kM, kL)) {
      const bm = books[i]._bbox, bl = books[L]._bbox, br = books[R]._bbox;
      const overlaps = (x, y) => {
        if (!x || !y) return true;
        const lo = Math.max(x[0], y[0]), hi = Math.min(x[0] + x[2], y[0] + y[2]);
        return hi - lo > 0 && hi - lo >= 0.5 * Math.min(x[2], y[2]);
      };
      if (overlaps(bm, bl) || overlaps(bm, br)) { misreads.push(i); continue; }
    }
    const vL = primaryStr(kL), vR = primaryStr(kR), vM = primaryStr(kM);
    if (vL && vR && vM && vL[0] === vR[0] &&
        Math.abs(vM.charCodeAt(0) - vL.charCodeAt(0)) >= 3 &&
        Math.abs(vM.charCodeAt(0) - vR.charCodeAt(0)) >= 3) misreads.push(i);
  }
  for (const i of misreads) keys[i] = null;

  CURRENT_SURNAME_COUNTS = new Map();
  // ...and the same census for the FULL author name. A garbled given name is
  // a one-off -- "Eud" happens once on a shelf of Eudoras -- while a real
  // author with several books on the shelf has their whole name read the same
  // way more than once, on physically different spines. That is the one piece
  // of evidence the sameauthor pass below can use to tell the two apart; see
  // the note there. Indexed positionally so a corroborating copy can be
  // required to sit somewhere ELSE on the shelf.
  CURRENT_AUTHOR_POS = new Map();
  for (let ki = 0; ki < keys.length; ki++) {
    const k = keys[ki];
    if (!k || k.kind !== 'str' || !k.val) continue;
    CURRENT_SURNAME_COUNTS.set(k.val, (CURRENT_SURNAME_COUNTS.get(k.val) || 0) + 1);
    if (!k.first) continue;
    const full = k.val + '|' + k.first;
    if (!CURRENT_AUTHOR_POS.has(full)) CURRENT_AUTHOR_POS.set(full, []);
    CURRENT_AUTHOR_POS.get(full).push(ki);
  }

  const vIdx = []; keys.forEach((k, i) => { if (k) vIdx.push(i); });
  // The in-order run is measured over COMPLETE reads only. A truncated read
  // ties every number it is a prefix of (compareKeys treats "616.8" as equal
  // to both "616.8527" and "616.8588"), so leaving it in the run lets it
  // BRIDGE a genuinely-misfiled clean read into a subsequence that only looks
  // non-decreasing because the truncated neighbour ties everything: a
  // "616.8588 GAR" sitting in a run of "616.8527" came back unflagged because
  // a truncated "616.8" beside it tied them together. Building the backbone
  // from complete reads keeps that bridge out; a truncated read that then
  // falls outside the run is never turned red — the _truncated demotion below
  // sends it to "check by hand" instead, which is the honest verdict for a
  // number we could not fully read.
  const backbone  = vIdx.filter(i => !books[i]._truncated);
  const lisLocal  = longestNonDecreasingSubseq(backbone.map(i => keys[i]));
  const lisGlobal = new Set([...lisLocal].map(li => backbone[li]));

  const sortedV = [...vIdx].sort((a, b) => compareKeys(keys[a], keys[b]));
  const idealRank = new Map();
  sortedV.forEach((origIdx, rank) => idealRank.set(origIdx, vIdx[rank] + 1));

  const tiedSlots = new Map();
  let r = 0;
  while (r < sortedV.length) {
    let s = r;
    while (s + 1 < sortedV.length && compareKeys(keys[sortedV[s + 1]], keys[sortedV[r]]) === 0) s++;
    const lo = vIdx[r] + 1, hi = vIdx[s] + 1;
    for (let k = r; k <= s; k++) tiedSlots.set(sortedV[k], [lo, hi]);
    r = s + 1;
  }

  const checked = books.map((book, i) => {
    const pos = i + 1;
    if (!keys[i]) return { ...book, _why: 'nokey', outOfOrder: false, unverifiable: true, possiblyOutOfOrder: false, idealPosition: null, shelfPosition: pos, shelfSize: books.length, tiedGroupSlots: null };
    const ideal    = idealRank.get(i) ?? null;
    const tied     = tiedSlots.get(i) || null;
    const inTied   = tied && pos >= tied[0] && pos <= tied[1];
    return { ...book, outOfOrder: !lisGlobal.has(i) && !inTied, unverifiable: false, possiblyOutOfOrder: false, idealPosition: ideal, shelfPosition: pos, shelfSize: books.length, tiedGroupSlots: tied };
  });

  // trustworthy enough to anchor a correctness claim; used by both the
  // near-twin demotion below and the strong-anchor rescue after it
  const STRONG_ANCHOR = 0.9;
  // Post-pass: sandwich / near-twin demotion
  for (let i = 0; i < checked.length; i++) {
    if (!checked[i].outOfOrder) continue;
    // Skip past null-key (unverifiable) neighbors too, not just flagged
    // ones — a truncated OCR read (VONNE next to VONNEGUT) often sits
    // beside an unparseable read, and stopping there let the near-twin
    // demotion below miss obvious truncation artifacts.
    let L = i - 1; while (L >= 0 && (checked[L].outOfOrder || !keys[L])) L--;
    let R = i + 1; while (R < checked.length && (checked[R].outOfOrder || !keys[R])) R++;
    if (L < 0 || R >= checked.length) continue;
    const kL = keys[L], kR = keys[R], kM = keys[i];
    if (!kL || !kR || !kM) continue;
    // This whole post-pass exists to rescue a SINGLE-engine misread that
    // looks sandwiched between two identical/near-identical real reads
    // (e.g. a garbled duplicate-detection artifact). A book with its OWN
    // independent cross-engine confirmation (score >= 0.9, src 'both')
    // doesn't need rescuing from that same-key-neighbors heuristic — two
    // engines agreeing on ITS text is stronger evidence than "its
    // neighbors happen to share a key", so demoting it anyway just
    // because duplicate copies of the same call number legitimately sit
    // on the shelf was needlessly turning good reads into "please check".
    // NOTE: `_src === 'both'` meant two ENGINES agreed on the text. Gemini
    // is gone, so this is now always false and the rescue never fires. That
    // looks like dead code but is NOT safe to "fix": relaxing it to a plain
    // score >= STRONG_ANCHOR was measured and made things worse on both
    // pipelines (app FP 11 -> 13; geometry FP 8 -> 9 plus a must_not_flag
    // violation). The demotion it disables is doing real work. Left exactly
    // as-is, deliberately.
    const crossConfirmed = checked[i]._score >= STRONG_ANCHOR && !checked[i]._malformed && !checked[i]._truncated;
    // "Strong" means the read is trustworthy, and OCR confidence is only one
    // way to establish that -- a sticker the vision model read the same way is
    // verified by a second, independent engine, which is better evidence than
    // any single-engine number. Hoisted out of the Dewey branch below because
    // the SURNAME branch needs exactly the same test and never had one.
    const isStrong = k => (checked[k]._score ?? 0) >= STRONG_ANCHOR ||
                          (!!checked[k]._gem && !checked[k]._inserted);
    let demote = !crossConfirmed && compareKeys(kL, kR) === 0;
    if (!demote && kL.kind === kR.kind && kM.kind === kL.kind) {
      const vL = kL.kind === 'str' ? kL.val : String(kL.val.num ?? '');
      const vR = kR.kind === 'str' ? kR.val : String(kR.val.num ?? '');
      const vM = kM.kind === 'str' ? kM.val : String(kM.val.num ?? '');
      if (vL && vR && vM && vL[0] === vR[0] &&
          Math.abs(vM.charCodeAt(0) - vL.charCodeAt(0)) >= 3 &&
          Math.abs(vM.charCodeAt(0) - vR.charCodeAt(0)) >= 3) demote = true;
    }
    if (!demote && kM.kind === 'str') {
      const vM = kM.val || '';
      for (const [kN, nIdx] of [[kL, L], [kR, R]]) {
        if (kN.kind !== 'str') continue;
        const vN = kN.val || '';
        const minLen = Math.min(vM.length, vN.length);
        // Two confident reads that are NOT a truncation of one another are
        // two different surnames, not one sticker read twice.
        //
        // Edit distance alone cannot tell those apart, and shelving is what
        // makes that fatal: the surnames closest in spelling are exactly the
        // ones filed next to each other. MURPHY and MURRAY are two edits
        // apart, both read at 0.99, and this rule quietly ate the genuine
        // misfile sitting between them (James S. Murray's "The Stowaway"
        // filed ahead of the three Amita Murrays).
        //
        // The truncation test is what keeps that safe. A real garble on a
        // dense run drops characters off the END -- "WALKE" beside "WALKER",
        // and Azure will hand back a truncated word at 0.9+ perfectly
        // happily, so confidence on its own is NOT enough (measured: without
        // this clause "WALKER Boo" goes red on a photo whose audit lists it
        // as must-not-flag). An internal substitution between two confident
        // reads is a different matter: OCR does not turn RRAY into RPHY.
        // Scoped to the edit-distance test only -- the shared-prefix test
        // below is the TREMBLAY/TREMBES family case and keeps its old
        // behaviour untouched.
        const truncation = vM.startsWith(vN) || vN.startsWith(vM);
        const bothSure = isStrong(i) && isStrong(nIdx) && !truncation;
        if (!bothSure && vM !== vN &&
            ((minLen >= 5 && editDistance(vM, vN) <= 2) ||
             (minLen >= 4 && editDistance(vM, vN) <= 1))) { demote = true; break; }
        // Long shared prefix: several damaged reads of ONE surname. A dense
        // author run returns "TREMBLAY", "TREMBES", "TREMBL", "TREME" for the
        // same five books; the tails diverge too far for edit distance (3+)
        // but a five-character common prefix is not something two different
        // surnames on the same shelf produce by accident. Sorting on the
        // garbled tails made the CORRECT read the odd one out and reddened a
        // book already confirmed correctly shelved. Five is deliberately long:
        // WALSH/WALKER share 3, TATA/TAWADA 2, THOMAS/THORPE 3, TOIBIN/TODD 2
        // — every real misfile in the corpus stays flaggable.
        if (vM !== vN && minLen >= 5) {
          let i = 0; while (i < minLen && vM[i] === vN[i]) i++;
          if (i >= 5) { demote = true; break; }
        }
      }
    }
    // Dewey cutter near-twin: same call number as both neighbors and a
    // cutter one edit away from a neighbor's is a misread of that neighbor's
    // sticker ("635 FRA" beside "635 FRO"), not a misshelved book.
    if (!demote && kM.kind !== 'str' && kL.kind === kM.kind && kR.kind === kM.kind &&
        kM.val.num != null && kM.val.num === kL.val.num && kM.val.num === kR.val.num) {
      const cM = kM.val.cut || '';
      // One edit is NOT evidence of a misread when the cutter is three
      // letters: FRA/FRO and WHA/WHI are one edit apart and ground truth
      // lists both as real, distinct books on the same shelf. Demoting on
      // edit distance alone cost three real misfiles (635 FRO twice,
      // 635 WHA). Require the flagged read to actually look like the weaker
      // one — a genuine misread is not more confident than the sticker it
      // was misread from.
      // `isStrong` (hoisted above) is what "trustworthy" means here.
      //
      // This is what the gate was costing: on 195907424 the live read is
      // PERFECT, all 20 labels matching ground truth including the
      // transposed WHI/WHA, and nothing was flagged solely because WHI came
      // back at 0.82 instead of 0.90. Re-running that exact label list with
      // WHI at 0.90 flags 635 WHA red and its partner yellow, correctly. A
      // 0.08 confidence difference was deciding whether a real misfile was
      // caught.
      const sM = checked[i]._score ?? 0;
      for (const [kN, nIdx] of [[kL, L], [kR, R]]) {
        const cN = kN.val.cut || '';
        if (!cM || !cN || cM === cN || editDistance(cM, cN) > 1) continue;
        const sN = checked[nIdx]._score ?? 0;
        if (isStrong(i) && isStrong(nIdx)) continue;
        if (sM > sN) continue;
        demote = true; break;
      }
    }
    if (demote) {
      checked[i].outOfOrder = false; checked[i].unverifiable = true;
      checked[i].idealPosition = null; checked[i]._why = 'neartwin';
    }
  }

  // Post-pass: a confident, correctly-placed book must not take the fall for a
  // neighbour's garbled read. The LIS can only keep one book from a conflicting
  // pair; when it keeps a WEAK misread and drops the CONFIDENT neighbour, the
  // red lands on the wrong book (measured: a "363.7052 U LOA" sticker read as
  // cutter "ULOA" sorted after "THO" and got the correctly-shelved "363.7052
  // THO" flagged, while ULOA itself — the actual misread — stayed in the run).
  // If a flagged book sits in non-decreasing order between its nearest STRONG,
  // in-order (LIS-backbone) neighbours on BOTH sides, it is genuinely in place,
  // so clear it. This only ever REMOVES a red — it cannot manufacture a false
  // positive; its one cost is possibly missing a real misfile, which the
  // strong-anchor-on-both-sides requirement bounds. A weak neighbour that then
  // stands alone as the out-of-run read falls to the confidence demotion below.
  for (let i = 0; i < checked.length; i++) {
    if (!checked[i].outOfOrder || !keys[i] || !(checked[i]._score >= STRONG_ANCHOR)) continue;
    let L = i - 1;
    while (L >= 0 && !(keys[L] && lisGlobal.has(L) && checked[L]._score >= STRONG_ANCHOR)) L--;
    let R = i + 1;
    while (R < checked.length && !(keys[R] && lisGlobal.has(R) && checked[R]._score >= STRONG_ANCHOR)) R++;
    if (L < 0 || R >= checked.length) continue;
    if (compareKeys(keys[L], keys[i]) <= 0 && compareKeys(keys[i], keys[R]) <= 0) {
      checked[i].outOfOrder = false;   // correctly shelved between confirmed in-order anchors
    }
  }

  // Fiction: never turn a book RED for its order WITHIN a run of its own
  // author. Libraries shelve one author's books together, so a red caused
  // PURELY by first-name / OCR-variant noise among same-surname neighbours
  // is not a misfiling a volunteer would reshelve (measured: a clean
  // "THAYNE RaeAnn" flagged red inside its own Thayne run; "TAYLOR Bred"
  // flagged inside a 15-book Taylor block; the clean-GT suite's "WALKER,
  // Ann Marie"/"WALKER, Boo").
  //
  // An earlier version of this rule demoted on ANY same-surname word within
  // a fixed physical window, regardless of why the book was flagged — that
  // wrongly demoted two REAL detections: "WILLIAMS Hattie" stranded in a
  // WILSON zone (its window just happened to also contain other WILLIAMS
  // entries) and "TOIBIN, Colm" misordered against TODD (three duplicate
  // reads of the same Toibin sticker sat in each other's window and
  // demoted one another). The fix: only demote when the SURNAME ITSELF
  // (compareFiling, the same comparator compareKeys uses before the
  // first-name tiebreak) is in correct order against the nearest
  // LIS-included neighbour on each side. If the surname is fine and only
  // the full key (surname+given name) violates the run, the given name is
  // the whole story — safe to demote. If the surname itself is out of
  // order against a flank (WILLIAMS after WILSON; TOIBIN before TODD),
  // that is a real cross-author stranding and must stay red.
  // nameSlip still guards a misread surname (a flank that is the SAME
  // author read slightly wrong) from being treated as a real cross-author
  // boundary. This only ever removes a red, so it cannot manufacture a
  // false positive.
  if (section === 'fiction') {
    for (let i = 0; i < checked.length; i++) {
      if (!checked[i].outOfOrder || !keys[i] || keys[i].kind !== 'str') continue;
      const sur = keys[i].val || '';
      if (!sur) continue;
      const sameRow = j => (books[j].shelfRow ?? 0) === (books[i].shelfRow ?? 0);
      let L = i - 1; while (L >= 0 && !(keys[L] && lisGlobal.has(L) && sameRow(L))) L--;
      let R = i + 1; while (R < checked.length && !(keys[R] && lisGlobal.has(R) && sameRow(R))) R++;
      const flankOk = (j, dir) => {
        if (j < 0 || j >= checked.length || !keys[j]) return true;   // no flank, vacuously fine
        const v = keys[j].val || '';
        // TRIED AND REVERTED (2026-08-22) — comparing GIVEN names here when
        // the flank is the same author, so that one Walker book out of
        // first-name order inside a run of Walkers could go red. That is the
        // commonest shelving mistake there is, and this rule is the only
        // thing standing between the app and detecting it, so it looked like
        // the obvious win.
        //
        // Measured over 36 shelves: true positives 12 -> 13, false positives
        // 1 -> 6, AND a must_not_flag violation. Every new false positive was
        // a garbled given name accusing a correctly shelved book -- "WELTY
        // Eud" for Eudora, "TILLY, Med" for Meg, "WHITE Kafe" for Kate. A
        // confidence bar does not rescue it: swept 0.85 / 0.75 / 0.65 / 0.0
        // and the false positives are identical at every level, because (as
        // the note in the pairing loop already records) an OCR misread comes
        // back just as confident as a good read. One extra detection is not
        // worth five false accusations and a book already confirmed correct.
        //
        // RETRIED AND REVERTED AGAIN (2026-08-23), three ways, all measured:
        //   a) gated on a second read of the same sticker agreeing -- the
        //      verification this note asks for. It NEVER FIRES: duplicate
        //      reads are merged upstream in the reader, so by the time books
        //      reach checkOrder the overlapping boxes are gone (measured: 0
        //      corroborated books on both test shelves, 0/4 overlapping
        //      pairs). The evidence exists in the READER, not here.
        //   b) blind but tolerant, via givenSlip -- which does absorb all
        //      three garbles above ("eud"/"eudora", "med"/"meg",
        //      "kafe"/"kate" all compare equal). FP 7 -> 9, TWO guard
        //      violations.
        //   c) strictest form: names must be unmistakably different (no
        //      containment either way, edit budget scaled by length). Still
        //      1 guard violation, FP 7 -> 8.
        // Every variant gained ZERO true positives (TP 11 -> 11, and the
        // clean-GT suite 2 -> 2). The books this was meant to catch --
        // "WALKER, Ann Marie" and "WALKER, Boo" -- are not reachable from
        // here at all: that shelf's surnames read WALKE / WALKET / WALKEN /
        // WALKED / "TRI WALKER" / "BOO WALKER" and its given names read
        // "Ann Mario", "Laren nom homesal", "waren". They are demoted by
        // the near-twin rule, never by this one, and no first-name
        // comparison can order text that garbled. It is a READING problem.
        //
        // Do not retry at the ordering layer. The only thing that would
        // change the answer is a given name verified in the READER, before
        // duplicate reads are merged.
        // Same combined guard as compareKeys (corroborated surname AND
        // disagreeing given names — see the comment there for why NEITHER
        // alone is safe: corroboration alone flags an innocent same-author
        // run with one garbled copy of its own correct surname; given
        // names alone misfire on ordinary given-name truncation). "CARR"
        // x11 next to "CARREY, Jim" clears both; "WALKER" x2 next to a
        // misread "WALKEN" (both blank given names) clears neither.
        // The veto below only applies to a bare one-word surname prefix
        // ("carr"/"carrey") — the same job nameSlip is already doing. It
        // must NOT apply when either side is a genuine multi-word garble
        // ("taylor blakem" for a book whose true surname is "Taylor",
        // "thay thatne" for "Thayne"): nameSlip's own space-stripping
        // makes those look like a one-word prefix too, but they're a
        // mangled read of ONE sticker regardless of given names or
        // repetition elsewhere — vetoing them broke two confirmed-correct
        // books (must_not_flag regressions, both measured).
        const bothSingleWord = sur.indexOf(' ') === -1 && v.indexOf(' ') === -1;
        const firstI = keys[i].first || '', firstJ = keys[j].first || '';
        const namesDisagree = bothSingleWord && firstI && firstJ &&
          firstI !== firstJ && !givenSlip(firstI, firstJ);
        const corroborated = bothSingleWord && CURRENT_SURNAME_COUNTS && (
          (CURRENT_SURNAME_COUNTS.get(sur) || 0) >= 2 ||
          (CURRENT_SURNAME_COUNTS.get(v) || 0) >= 2);
        const slipOk = nameSlip(sur, v) && !(corroborated && namesDisagree);
        if (!v || slipOk) return true;                    // same author, misread surname
        // A same-author flank can itself be a badly garbled MULTI-WORD OCR
        // read of the identical sticker ("THAY! THATNE." parsed as surname
        // "thay thatne" for a book whose true surname is "Thayne") that
        // nameSlip's single-name edit-distance test does not recognise.
        // Compare just the leading token on each side with the same prefix
        // tolerance — still same-author evidence, not a licence to ignore a
        // real cross-author boundary (both leading tokens must clear
        // NAME_PREFIX_MIN, same bar nameSlip itself uses).
        const a0 = sur.split(' ')[0] || '', b0 = v.split(' ')[0] || '';
        if (!(corroborated && namesDisagree) &&
            a0.length >= NAME_PREFIX_MIN && b0.length >= NAME_PREFIX_MIN &&
            (a0.startsWith(b0) || b0.startsWith(a0))) return true;
        // FIFTH ATTEMPT (2026-09-02), and the first one with evidence behind
        // it. Everything above stands: a given name may NOT accuse on its own.
        // But read the four failures again -- "WELTY Eud" for Eudora, "TILLY,
        // Med" for Meg, "WHITE Kafe" for Kate, and the WALKER shelf's "Ann
        // Mario" / "waren" -- and they are all the SAME shape. Every one is a
        // garble that occurs ONCE. Attempt (a) already identified the missing
        // ingredient ("a given name verified in the READER") but looked for it
        // in the wrong place: it asked for a second read of the same sticker,
        // which the reader merges away before checkOrder ever sees it.
        //
        // The verification does exist, and it survives that merge, because it
        // is not a second read -- it is a second BOOK. An author with several
        // books on a shelf has their whole name read identically on several
        // physically separate spines; OCR does not garble the same name the
        // same way twice at different points on the shelf. So: compare given
        // names only when BOTH sides' full author names are corroborated by a
        // copy standing elsewhere on this shelf.
        //
        // That is the exact discriminator. On the reported photo "MURRAY,
        // James S." is read four times (0.99 / 0.97 / 0.95 / 0.95) and the
        // flank "MURRAY, Amita" three times, so the stranded copy filed ahead
        // of the Amita run is judged and goes red. "WELTY Eud" is read once,
        // scores no corroboration, and stays demoted exactly as before -- as
        // do all four historical false positives, none of which can reach a
        // count of two. It is the same corroboration test compareKeys already
        // applies to SURNAMES (CURRENT_SURNAME_COUNTS), just extended to the
        // rest of the name.
        //
        // Distance >= 2 so a garble cannot corroborate itself off an adjacent
        // duplicate read, and nameSlip still absorbs an ordinary truncation
        // ("Jame" vs "James"), so this can only ever fire on two names that
        // are both repeated AND unmistakably different.
        const corrob = idx => {
          const k = keys[idx];
          if (!k || k.kind !== 'str' || !k.val || !k.first || !CURRENT_AUTHOR_POS) return false;
          const at = CURRENT_AUTHOR_POS.get(k.val + '|' + k.first) || [];
          return at.some(p => Math.abs(p - idx) >= 2);
        };
        const surCmp = dir < 0 ? compareFiling(v, sur) : compareFiling(sur, v);
        if (surCmp > 0) return false;              // real cross-author stranding
        if (surCmp === 0 && corrob(i) && corrob(j)) {
          const gI = keys[i].first || '', gJ = keys[j].first || '';
          if (gI && gJ && !nameSlip(gI, gJ))
            return dir < 0 ? compareFiling(gJ, gI) <= 0 : compareFiling(gI, gJ) <= 0;
        }
        return true;
      };
      if (flankOk(L, -1) && flankOk(R, 1)) {
        checked[i].outOfOrder = false; checked[i].possiblyOutOfOrder = true;
        checked[i].idealPosition = null; checked[i]._why = 'sameauthor';
      }
    }
  }

  // Fiction: a red flag is a GARBLED FRAGMENT of a read Azure ALSO produced
  // correctly, elsewhere on the same shelf, at higher confidence — not a
  // second, differently-named book. Measured: Azure returns "TRIGIANI" at
  // 0.99+ many times on a shelf, and once returns "TRIC" (0.76) — a genuine
  // standalone OCR word, not a weld of two spines — that sorts far from
  // TRIGIANI's true position and gets flagged. Same for "RIGIA" against
  // "TRIGIANI" and "HIRKELL" against "THIRKELL" (a dropped leading letter).
  // The fragment is short and degraded; the full read is long, confident,
  // and the shelf itself already accepted it into the in-order run.
  //
  // Length-scaled containment, not a flat edit-distance bar, is load-
  // bearing: under a flat "≤1 edit" rule, "TATA" (a REAL misfile, stranded
  // among TAWADAs) is edit-distance 1 from "TAWA" and would be wrongly
  // demoted, silencing a genuine detection. A short fragment (4-5 letters)
  // must be an EXACT substring of the longer read to count; only surnames
  // long enough to be unambiguous (6+) get the 1-edit tolerance, and even
  // then only within a matching-length WINDOW of the longer string, not
  // the whole thing. Demote only, never drop — this only ever removes a
  // red, so it cannot manufacture a false positive.
  if (section === 'fiction') {
    const isFragmentOf = (s, t) => {
      if (t.length < s.length + 1) return false;
      if (s.length <= 5) return t.includes(s);
      for (let start = 0; start <= t.length - s.length + 1; start++) {
        for (const len of [s.length - 1, s.length, s.length + 1]) {
          if (len < 1 || start + len > t.length) continue;
          if (editDistance(s, t.slice(start, start + len)) <= 1) return true;
        }
      }
      return false;
    };
    // Deliberately no "t's confidence must beat s's" comparison. Confidence
    // does not separate a misread from a good one — a documented, measured
    // fact elsewhere in this file (the fusion tie-break comment: "the
    // misreads come back just as confident as the good reads") — and it is
    // measured again here: "RIGIA Adrian" reads at 0.988, higher than every
    // surviving "TRIGIANI, Adriana" anchor on its own shelf (0.984, 0.914 —
    // the one at 0.993 was itself collapsed by the duplicate-read pre-pass
    // above before this rule ever saw it). An absolute floor (STRONG_ANCHOR)
    // is the load-bearing bar, not a relative one.
    for (let i = 0; i < checked.length; i++) {
      if (!checked[i].outOfOrder || !keys[i] || keys[i].kind !== 'str') continue;
      const s = keys[i].val || '';
      if (s.length < 3) continue;
      for (let j = 0; j < checked.length; j++) {
        if (j === i || !keys[j] || keys[j].kind !== 'str' || !lisGlobal.has(j)) continue;
        const t = keys[j].val || '';
        if ((checked[j]._score ?? 0) < STRONG_ANCHOR) continue;
        // A surname collision this close still isn't the same author when
        // the GIVEN names plainly disagree ("Jim" vs "Emmanuel" for
        // "Carrey"/"Carrère" — edit-distance 1 apart as surnames, real
        // people). Only fires as a veto: no first name on either side, or
        // an exact/slip match, is silently fine and falls through to the
        // surname-only fragment test as before.
        const gI = keys[i].first || '', gJ = keys[j].first || '';
        if (gI && gJ && gI !== gJ && !givenSlip(gI, gJ)) continue;
        if (isFragmentOf(s, t)) {
          checked[i].outOfOrder = false; checked[i].possiblyOutOfOrder = true;
          checked[i].idealPosition = null; checked[i]._why = 'fragment';
          break;
        }
      }
    }
    // Companion bar for a short garbled surname with NO surviving sibling
    // read to contain-match against ("TRIC", 0.76 — Azure's own only
    // reading of that spine at the shipped 2400px frame; the cleaner
    // "TRIGIANI" reads elsewhere on the shelf don't share enough of TRIC's
    // letters for isFragmentOf to connect them). Length capped at 4:
    // longer surnames keep their evidence even at moderate confidence, and
    // 4-5 letter surnames that clear 0.85 stay red — this is specifically
    // why isFragmentOf demands an EXACT substring at that length rather
    // than reaching for a bar here too. Verified against the one real
    // 4-letter misfile in this shelf set: "TATA" (Direct Fire, stranded
    // after TAWADA) reads at 0.881 and clears this bar untouched.
    for (const b of checked) {
      if (!b.outOfOrder || !b._src) continue;
      const sur = (b.spine_label || '').split(/[,\s]+/)[0] || '';
      if (sur.length <= 4 && (b._score ?? 0) < 0.85) {
        b.outOfOrder = false; b.possiblyOutOfOrder = true;
        b.idealPosition = null; b._why = 'fragment';
      }
    }
  }

  // ── Unverified-title order pass → possiblyOutOfOrder (yellow) ────────────
  // Within a primary-key tie group, unverified spine-read titles are blanked
  // by sortKey (never feed the red LIS), so two clearly out-of-alphabetical-
  // order titles on tied books go unsurfaced. Surface that as YELLOW
  // ("possibly out of order — not sure"), NEVER red: a title that is not
  // OL-verified is single-source evidence and must never cause outOfOrder
  // (the 0-false-positive misshelving invariant). Verified titles already
  // break the primary key and get red via the LIS above, so they are
  // excluded here.
  //
  // Noise guard: only a FIRST-LETTER disagreement counts as a violation.
  // Sub-first-letter OCR noise (a stray middle character) must not trigger a
  // flag, so titles sharing a first letter are treated as tied for ordering.
  const filingTitle = t =>
    lettersLower(String(t || '').replace(/^the\s+|^a\s+|^an\s+/i, ''));
  const titleCmp = (a, b) => (a[0] === b[0] ? 0 : compareFiling(a, b));
  const titleLNDS = titles => {
    const n = titles.length;
    if (!n) return new Set();
    const dp = new Array(n).fill(1), prev = new Array(n).fill(-1);
    let bestEnd = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        if (titleCmp(titles[j], titles[i]) <= 0 && dp[j] + 1 > dp[i]) { dp[i] = dp[j] + 1; prev[i] = j; }
      }
      if (dp[i] > dp[bestEnd]) bestEnd = i;
    }
    const set = new Set();
    for (let i = bestEnd; i !== -1; i = prev[i]) set.add(i);
    return set;
  };

  // Group verified indices by their tie range (lo is unique per group, since
  // vIdx is strictly increasing). Books in the same tie group share [lo, hi].
  const titleGroups = new Map();
  for (let i = 0; i < checked.length; i++) {
    const tied = tiedSlots.get(i);
    if (!tied) continue;
    const gk = tied[0];
    if (!titleGroups.has(gk)) titleGroups.set(gk, []);
    titleGroups.get(gk).push(i);   // pushed in physical (left→right) order
  }
  for (const members of titleGroups.values()) {
    if (members.length < 2) continue;   // singletons are never "out of order"
    const eligIdx = [], eligTitle = [];
    for (const i of members) {
      const b = checked[i];
      if (b.outOfOrder || b.unverifiable) continue;   // already surfaced otherwise
      if (b.title_verified) continue;                 // verified titles → red path, handled by LIS
      const ft = filingTitle(b.title);
      if (!ft) continue;                              // empty/absent title
      if ((b.title || '').replace(/[^a-z0-9]/gi, '').length < 4) continue;  // min length bar
      if (!(typeof b.title_conf === 'number' && b.title_conf >= 0.6)) continue;  // confidence bar
      eligIdx.push(i); eligTitle.push(ft);
    }
    if (eligIdx.length < 2) continue;
    const keep = titleLNDS(eligTitle);   // longest non-decreasing (by first letter) run
    eligTitle.forEach((_, k) => {
      if (!keep.has(k)) checked[eligIdx[k]].possiblyOutOfOrder = true;
    });
  }

  // ── Red requires cross-engine confirmation → otherwise yellow ────────────
  // A red "misshelved" flag is an accusation and must never rest on a single
  // engine (the sacred 0-false-positive invariant). Only a fused read (src
  // 'both', score >= 0.65) may STAY red. A confident on-device-only read (AR
  // local-only cycle: src 'gemini'/'ocr-only', score ~0.6) that lands out of
  // sequence is downgraded to yellow "possibly out of order" — still boxed
  // and sortable (so its in-order neighbours render green), but the definite
  // red flag waits for the periodic remote fuse to confirm it. Books with no
  // score/src metadata (offline ordering GT) are trusted and keep red, so the
  // ordering suite is byte-for-byte unaffected.
  // Truncated-neighbour demotion (nonfiction). Every reader here clips
  // trailing decimals on tiny stickers: the on-device engine folded them at
  // the spine edge, and the Gemini reader flattens a whole 635.048x block to
  // "635.04" while reading two of its members in full. The books that then
  // look misplaced are the ACCURATELY read ones — measured live as 3 of 5
  // false positives ("635.0483 BRO" and "635.0484 BIG" accused by a row of
  // truncated 635.04 neighbours).
  //
  // The test is the VIOLATION, not the row. A book is only demoted when
  // every in-order neighbour it actually conflicts with is a prefix pair with
  // it (one number is a truncation of the other). A conflict against a
  // neighbour with the SAME number is a cutter violation — real evidence,
  // untouched — which is what the genuine detections are made of
  // ("635.04 SMI, SMI, HEM"). Demoting on the whole row instead erases those
  // too (tried: TP 5 -> 1).
  if (section !== 'fiction') {
    const numOf = k => (k && k.kind !== 'str' && k.val && k.val.num != null)
      ? String(k.val.num) : null;
    const prefixPair = (x, y) =>
      x && y && x !== y && (x.startsWith(y) || y.startsWith(x));
    for (let i = 0; i < checked.length; i++) {
      const b = checked[i];
      if (!b.outOfOrder || !keys[i]) continue;
      const myNum = numOf(keys[i]);
      if (!myNum || !myNum.includes('.')) continue;
      // In-order neighbours this book actually conflicts with.
      const conflicts = [];
      for (const j of lisGlobal) {
        if (!keys[j]) continue;
        const before = j < i, cmp = compareKeys(keys[j], keys[i]);
        if ((before && cmp > 0) || (!before && cmp < 0)) conflicts.push(j);
      }
      if (!conflicts.length) continue;
      if (conflicts.every(j => prefixPair(myNum, numOf(keys[j])))) {
        b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
      }
    }
  }

  // Per-shelf Dewey class census. A lone class one character off from the
  // shelf's dominant class ("535" among 635s, "041.5" among 641.5s) is a
  // correlated glyph misread, not a misshelved book — 5 and 6 differ by one
  // stroke on a worn sticker. This test used to live in the fusion path, so
  // it only ran when two engines were being merged and it required a decimal
  // point to match, which let bare misreads like "535 HOM" through. It
  // belongs here, where every engine's reads pass through equally.
  //
  // Deliberately narrow: the shelf must show the modal class at least four
  // times, this book's class at most once, and the two must differ by exactly
  // one character. A genuinely misshelved book from another class is nearly
  // always more than one character away.
  if (section === 'nonfiction') {
    const cls = b => { const d = parseDewey(b.spine_label);
                       return d && d.numStr ? String(d.numStr).split('.')[0] : null; };
    const cm = new Map();
    for (const b of checked) {
      const c = cls(b);
      if (c && b._score >= 0.65) cm.set(c, (cm.get(c) || 0) + 1);
    }
    let modal = null, modalN = 0;
    for (const [c, n] of cm) if (n > modalN) { modal = c; modalN = n; }
    const off1 = (a, z) => a.length === z.length &&
      a.split('').reduce((n, ch, i) => n + (ch !== z[i] ? 1 : 0), 0) === 1;
    if (modal && modalN >= 4) {
      for (const b of checked) {
        const c = cls(b);
        if (!b.outOfOrder || !c || c === modal) continue;
        if ((cm.get(c) || 0) <= 1 && off1(c, modal)) {
          b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
        }
      }
    }
  }

  // Shelf-consensus truncation. A call number that is a strict MID-DECIMAL
  // prefix of a longer number on the SAME shelf lost its trailing digits --
  // "641.56" beside "641.568", "635.04" beside "635.0483". The spine curves
  // away or the read stops early; either way the shelver did nothing wrong,
  // and sorting on the short value drops the book in the wrong place. Marking
  // it _truncated routes it to yellow through the existing demotion below.
  // Only mid-decimal counts: "635" is a prefix of "635.01" but is a perfectly
  // good whole Dewey number, and treating it as truncated yellows every plain
  // 635 on the shelf and silently kills detection there.
  // Measured on 28 photos: false positives 17 -> 8, true positives unchanged.
  if (section === 'nonfiction') {
    const numTxt = checked.map(b => (/^\s*([\d.]+)/.exec(b.spine_label || '') || [])[1] || '');
    const intOf = m => m.split('.')[0];
    const widths = numTxt.filter(Boolean).map(m => intOf(m).length).sort((a, b) => a - b);
    const modalW = widths[widths.length >> 1] || 3;
    checked.forEach((b, i) => {
      const n = numTxt[i];
      if (!n) return;
      const ip = intOf(n);
      let trunc = n.includes('.') &&
                  numTxt.some((m, j) => j !== i && m.length > n.length && m.startsWith(n));
      // A Dewey class is three digits, so an integer part narrower than the
      // shelf's own modal width is a damaged read, not a real number --
      // "3.7" on a shelf of 363.70xx.
      if (!trunc && ip.length < modalW) trunc = true;
      // No decimal of its own, on a shelf where this same class almost always
      // has one: "363" among 363.7052/363.7063 is the same sticker with its
      // decimals unread. The majority guard keeps a genuine whole number
      // ("635" on a shelf of plain 635s) out of this.
      if (!trunc && !n.includes('.')) {
        const same = numTxt.filter(m => m && intOf(m) === ip);
        const dec  = same.filter(m => m.includes('.')).length;
        if (same.length >= 3 && dec / same.length > 0.6) trunc = true;
      }
      if (trunc) b._truncated = true;
    });
  }

  // Deep-decimal cutter-clash demotion (single vision-model reader).
  // A vision model reading a shelf of, say, 641.5686 books REGULARIZES the
  // genuine outliers (a real 641.56 or 641.5 sticker) up to the dominant
  // 641.568 — confidently, and it does this even on a close-up crop where the
  // true trailing digits are plainly legible (measured 2026-09-17: three
  // stickers a human reads as 641.5 / 641.56 / 641.5686 all came back
  // "641.568"). The flattened outliers then sit as SAME-number books whose
  // cutter is out of alphabetical order, and get flagged red on a distinction
  // the reader cannot actually make — the exact trailing-decimal ambiguity the
  // shelf-consensus truncation rule above is meant to catch, except the reader
  // destroyed the short read it keys on.
  //
  // So demote a red whose number carries deep decimals (>= 2 places) when a
  // book with the IDENTICAL number sits in order on the same row: that is the
  // fingerprint of a flattened deep-decimal shelf, not a real misfile. It is
  // deliberately narrow and leaves the genuine detections alone:
  //   • a real cutter misfile on a shallow shelf ("635 GAR" among plain 635s)
  //     has < 2 decimals, untouched — verified it still flags GAR;
  //   • a real NUMBER inversion ("635.04 HEM" stranded in a 635.048x run)
  //     conflicts with DIFFERENT-numbered neighbours, so no identical-number
  //     book is in order beside it, untouched.
  // Only ever demotes red -> yellow, so it cannot manufacture a false positive;
  // the book is still surfaced for a human to check. Measured on the 10 row-GT
  // photos read by the vision model: false reds 12 -> 6, real catches unchanged.
  if (section === 'nonfiction') {
    const numOf = b => { const d = parseDewey(b.spine_label); return d && d.numStr ? String(d.numStr) : null; };
    const decOf = n => { const i = n.indexOf('.'); return i < 0 ? 0 : n.length - i - 1; };
    for (const b of checked) {
      if (!b.outOfOrder) continue;
      const n = numOf(b);
      if (!n || decOf(n) < 2) continue;
      const sameNumInOrder = checked.some(o => o !== b && numOf(o) === n &&
        (o.shelfRow ?? 0) === (b.shelfRow ?? 0) &&
        !o.outOfOrder && !o.possiblyOutOfOrder && !o.unverifiable);
      if (sameNumInOrder) {
        b.outOfOrder = false; b.possiblyOutOfOrder = true;
        b.idealPosition = null; b._why = 'deepdecimal';
      }
    }
  }

  for (const b of checked) {
    // Red requires BOTH conditions, not either one: the read must be
    // cross-confirmed by both engines AND clear the 0.65 confidence bar.
    // Accepting either alone let a two-engine agreement at 0.6 raise a red
    // flag (measured: "635.04 Col", a misread cutter, accusing a correctly
    // shelved book) — which is precisely the case the fused-confidence rule
    // exists to stop.
    // Post-rewrite (single vision-model reader, no more Azure/fusion): the
    // reasoning that let 'azure' alone stand for "the verified read" in
    // Azure-only mode carries over unchanged to 'gemini' — there is again
    // exactly one engine, so requiring a SECOND one to agree would bar every
    // red flag outright. A book with no _src at all (offline ordering-test
    // fixtures) is trusted as before, so the test suite is unaffected.
    if (b.outOfOrder && b._src &&
        !(b._src === 'gemini' && b._score >= RED_MIN_SCORE)) {
      b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
      b._why = 'lowconf';
    }
    // A truncated read can never be red, whatever the ordering says. We do not
    // know the digits that would decide it, so "out of order" would be an
    // accusation resting on digits nobody read. Yellow says the true thing:
    // this one needs a human's eyes. idealPosition is cleared for the same
    // reason — we cannot name a slot we cannot compute.
    if (b._truncated && b.outOfOrder) {
      b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
      b._why = 'trunc';
    }
    // Same label, twice on one row, far apart, and the other copy sits IN
    // ORDER: the flagged one is a misread wearing a real label's text. The
    // first sticker on this shelf came back "635 COM" at 0.99 where the shelf
    // truly begins "635 BAK", and a genuine "635 COM" stood fifteen books
    // later exactly where it belongs. Confidence cannot separate those -- the
    // misread is the MORE confident of the two.
    // Distance is what makes this safe: real duplicate call numbers sit side
    // by side ("635 COL 635 COL", "635 DAM 635 DAM") and never flag each
    // other, so requiring a gap leaves genuine repeats alone. Cross-ROW
    // repeats are handled separately and geometrically, because a Dewey run
    // legitimately carries the same number onto the next shelf.
    if (section !== 'fiction' && b.outOfOrder) {
      const norm2 = t => String(t || '').toUpperCase().replace(/\s+/g, ' ').trim();
      const me = norm2(b.spine_label), iMe = checked.indexOf(b);
      if (me) for (let j = 0; j < checked.length; j++) {
        const o = checked[j];
        if (o === b || Math.abs(j - iMe) < 3) continue;
        if ((o.shelfRow ?? 0) !== (b.shelfRow ?? 0)) continue;
        if (o.outOfOrder || o.unverifiable || o._malformed) continue;
        if (norm2(o.spine_label) !== me) continue;
        b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
        b._why = 'rowdupe';
        break;
      }
    }
    // A very short fiction surname needs corroboration before it may accuse.
    // Azure returns the tail of a clipped sticker as a whole word -- "VER" at
    // 0.989 where the spine says OLIVER -- and three letters sort nowhere near
    // the real name, so it lands mid-shelf looking misfiled. Confidence cannot
    // catch this; the read IS confident, it is just not the whole name.
    // Corroboration rather than a ban, because real three-letter surnames
    // exist (Lee, Fox, Poe): a genuine one has siblings on an author-run
    // shelf, a clipped fragment has none. Ground truth here contains exactly
    // one three-letter "surname" and it is the filing prefix VON.
    if (section === 'fiction' && b.outOfOrder) {
      const sur = t => String(t || '').trim().toUpperCase()
                           .replace(/[^A-Z ]/g, '').split(/\s+/)[0] || '';
      const mine = sur(b.spine_label);
      if (mine && mine.length <= 3 &&
          !checked.some(o => o !== b && !o._malformed && sur(o.spine_label) === mine)) {
        b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
        b._why = 'shortsurname';
      }
    }
    // A surname read with no given name can be filed under its author but not
    // WITHIN that author: a run of one surname is ordered by given name, and
    // we did not read one. Yellow is the honest verdict.
    if (b._noGiven && b.outOfOrder) {
      // ...but only where the given name is what the ordering turns on, i.e.
      // a neighbour shares this surname. A surname sitting in the wrong place
      // among DIFFERENT surnames ("WALSH" inside a run of WALKERs) is a real
      // misfile and the missing given name is irrelevant to it.
      const sur = t => String(t || '').trim().toUpperCase()
                           .replace(/[^A-Z ]/g, '').split(/\s+/)[0] || '';
      const mine = sur(b.spine_label);
      const twin = mine && checked.some(o => o !== b && !o._malformed &&
                                             sur(o.spine_label) === mine);
      // The `twin` test alone was too coarse: a surname-only read whose
      // SURNAME sorts out of order at the surname level -- a stray "TATA"
      // physically sitting AFTER the whole TAWADA run (TATA < TAWADA) -- has
      // twins (the 7 real Tatas at the start of the shelf) yet its misfiling
      // has nothing to do with a missing given name. Keep it RED when the
      // surname ITSELF is misplaced against the nearest STRONG, in-order
      // (LIS-backbone) neighbour on either side -- the exact surname-level,
      // strong-anchor evidence the sameauthor flank rule already trusts.
      // Cannot red a correctly-shelved book: a correct surname is, by
      // definition, in order against its confirmed in-order flanks.
      const bi = checked.indexOf(b);
      let surnameMisplaced = false;
      if (keys[bi] && keys[bi].kind === 'str') {
        const surI = keys[bi].val || '';
        let L = bi - 1;
        while (L >= 0 && !(keys[L] && keys[L].kind === 'str' && lisGlobal.has(L) &&
                           checked[L]._score >= STRONG_ANCHOR)) L--;
        let R = bi + 1;
        while (R < checked.length && !(keys[R] && keys[R].kind === 'str' && lisGlobal.has(R) &&
                                       checked[R]._score >= STRONG_ANCHOR)) R++;
        if (L >= 0 && (keys[L].val || '') !== surI &&
            compareFiling(keys[L].val || '', surI) > 0) surnameMisplaced = true;
        if (R < checked.length && (keys[R].val || '') !== surI &&
            compareFiling(surI, keys[R].val || '') > 0) surnameMisplaced = true;
      }
      if (twin && !surnameMisplaced) {
        b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
        b._why = 'nogiven';
      }
    }
    // A malformed label can never be red either, for a stronger reason than
    // truncation: there is no single book it reliably refers to.
    // "BOO WAL WALKER, Boo" is two stickers welded together, so flagging it
    // accuses whichever of the two the sort happened to land on.
    if (b._malformed && b.outOfOrder) {
      b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
      b._why = 'malformed';
    }
    // Same bar, for a number that geometry says may have a decimal tail
    // (a tail-shaped word sat right where one would continue) which the
    // shelf could not corroborate. Unlike _truncated this needs no sibling
    // consensus to trip — it is a live, uncorroborated ambiguity on THIS
    // sticker, exactly the "635 COM" case: a short number sorts below its
    // full-length neighbours and reads as a misfiling that is not there.
    if (b._ambiguousTail && b.outOfOrder) {
      b.outOfOrder = false; b.possiblyOutOfOrder = true; b.idealPosition = null;
      b._why = 'tail';
    }
  }

  // Surface the OTHER side of every accusation.
  //
  // "Out of order" is a statement about a PAIR: book A sorts after book B
  // while sitting before it. The app names one of them, and which one it
  // names is decided by whichever choice leaves the longer in-order run --
  // usually right, but genuinely undecidable when the book physically
  // between them was unreadable. Measured: on one shelf "635 FRE" is flagged
  // and "635 FRO" shown GREEN, when ground truth says FRO is the misfile and
  // the book between them (635 FRA, 0.47) could not be read to settle it.
  // The volunteer was being sent to one of two adjacent books with no hint
  // that its neighbour was the other candidate.
  //
  // So mark the conflicting partner YELLOW. It cannot raise a false
  // accusation -- yellow is "check this", not "this is wrong" -- and it turns
  // a silent miss into a book the volunteer actually looks at. Only ever
  // touches a book that is currently clean; never downgrades a red.
  for (let i = 0; i < checked.length; i++) {
    if (!checked[i].outOfOrder || !keys[i]) continue;
    for (const dir of [-1, 1]) {
      let j = i + dir;
      while (j >= 0 && j < checked.length && !keys[j]) j += dir;
      if (j < 0 || j >= checked.length) continue;
      if (checked[j].outOfOrder || checked[j].unverifiable || checked[j].possiblyOutOfOrder) continue;
      if ((books[j].shelfRow ?? 0) !== (books[i].shelfRow ?? 0)) continue;
      // the inversion runs whichever way the pair sits on the shelf
      const bad = dir > 0 ? compareKeys(keys[i], keys[j]) > 0
                          : compareKeys(keys[j], keys[i]) > 0;
      if (!bad) continue;
      // ...but only when it is genuinely undecidable, which the rule above
      // asserts and never tested. A partner whose own key is repeated by
      // UNFLAGGED books elsewhere in the row is confirmed in place by its own
      // copies: there is nothing for a volunteer to decide, and the "the book
      // between them was unreadable" story does not apply. Measured on a
      // MURPHY shelf: MUNAWEERA sitting alone inside a run of twenty MURPHYs
      // is an unambiguous red, and this rule put a yellow on one of six
      // identical "MURPHY, Monica" reads whose five siblings were green.
      if (checked.some((o, k) => k !== j && keys[k] &&
            !o.outOfOrder && !o.unverifiable && !o.possiblyOutOfOrder &&
            (books[k].shelfRow ?? 0) === (books[j].shelfRow ?? 0) &&
            compareKeys(keys[k], keys[j]) === 0)) continue;
      checked[j].possiblyOutOfOrder = true;
      checked[j].idealPosition = null;
      checked[j]._why = 'pair';
    }
  }

  return checked;
}

export function checkOrder(books, section) {
  const shelves = groupByShelfRow(books);
  const checked = shelves.flatMap((shelf, si) =>
    checkOrderOneShelf(shelf, section).map(b => ({ ...b, shelfIndex: si }))
  );
  // Stray-duplicate demotion: a flagged label whose exact text also sits
  // IN order in a different row is almost always the same sticker detected
  // twice (tile overlap / row mis-clustering), not a second misshelved
  // copy — don't accuse on a phantom.
  const dupNorm = t => (t || '').toUpperCase().replace(/\s+/g, ' ').trim();
  for (const b of checked) {
    if (!b.outOfOrder) continue;
    const n = dupNorm(b.spine_label);
    if (!n) continue;
    const twin = checked.find(o => o !== b && !o.outOfOrder && !o.unverifiable &&
                                   o.shelfRow !== b.shelfRow && dupNorm(o.spine_label) === n);
    if (!twin) continue;
    // Matching TEXT on another row is NOT evidence of a phantom. A Dewey run
    // legitimately puts "635 GAR" on two rows, and a long author run spans
    // rows the same way — on the 0607 gardening shelves this rule was eating
    // a real misfile (a stranded 635 GAR whose twin sat in order one row
    // down). A genuine phantom is ONE sticker assigned to two rows, so the
    // two boxes sit at nearly the same x. Require that; with no geometry to
    // check, do not accuse the rule's way out of a real detection.
    // Measured over 28 photos: TP 8 -> 9, FP unchanged at 10, 0 guard hits.
    const bb = b._bbox, tb = twin._bbox;
    if (!bb || !tb) continue;
    const sep = Math.abs((bb[0] + bb[2] / 2) - (tb[0] + tb[2] / 2));
    if (sep > Math.max(bb[2], tb[2]) * 0.75) continue;
    b.outOfOrder = false; b.unverifiable = true; b.idealPosition = null;
    b._why = 'dupe';
  }
  return checked;
}
