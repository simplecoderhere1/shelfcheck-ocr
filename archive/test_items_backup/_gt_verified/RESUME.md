# Ground-truth rebuild — where it stands

`ground_truth_v2.json` lists roughly **half** the books physically on these
shelves. A crop-by-crop audit of the labels it did not account for found them to
be real, legible stickers — 17 of 17 on 195911082, 17 of 17 on 195849222. So:

- **recall** measured against v2 was flattering (denominator too small)
- **precision** against v2 was meaningless (a correct read counted as wrong for
  matching a book the key simply never listed)

Nothing about the app's reading accuracy can be trusted until v3 exists.

## Done

| shelf | result |
|---|---|
| `PXL_20260607_195907424` | 20/20 verified `ok`, 0 missed |
| `PXL_20260607_195853918` | 21 `ok`, 1 `junk`, **3 missed stickers found** (641.5686 DEV / SOU / STI) |

Both agents were careful and specific — the junk call correctly identified cover
art (`100 Recipes That Redefine Outdoor Cooking`) rather than a sticker, and the
second agent noticed something the key needs to record: **row 1 of 195853918
shows page-tops, not spines, so no call number is readable there at all.**

## Remaining (8 shelves)

```
PXL_20260607_195849222      nonfiction  51 labels  2 rows  (PORTRAIT photo)
PXL_20260607_195853918      DONE
PXL_20260607_195858362      nonfiction  53 labels  2 rows
PXL_20260607_195901698.MP   nonfiction  49 labels  2 rows  (has the "635 FAS" false flag)
PXL_20260607_195903035      nonfiction  24 labels  1 row   (has the "635 ICON" junk label)
PXL_20260607_195907424      DONE
PXL_20260607_195909571.MP   nonfiction  27 labels  1 row   (has the "635.0 COM" false flag)
PXL_20260607_195911082      nonfiction  38 labels  2 rows  (worst GT coverage: 17 unaccounted)
PXL_20260607_200004299.MP   fiction     55 labels  5 rows
PXL_20260607_200006014      fiction     49 labels  3 rows  (WALLACE, David Foster misorder)
```

Nine agents were launched and died together on a session usage limit, not on any
fault in the task — one of the two that did complete had already finished by
then. Re-run them when quota allows.

## How to re-run

Artifacts are already built; no need to regenerate unless the app's reads change.

```
node gt_plan.mjs --fixture=live_step1.json   # only if re-reading the shelves
python gt_sheets.py                          # only if the plan changed
```

Spawn one Sonnet agent per remaining shelf. The prompt that worked is below —
substitute the image name, section, and the coverage-segment list (see the files
in this directory, named `<img>_row<N>_seg<M>.jpg`).

Give each agent BOTH artifacts and be explicit that Task B is the important one:

- `<img>_crops.jpg` — numbered tile per read label, sticker ringed red.
  Verdicts: `ok` / `corrected` / `illegible` / `junk`.
- `<img>_row<N>_seg<M>.jpg` — the shelf at readable zoom, every read label boxed
  green and numbered. Task B is to find stickers with **no** green box.

Tell it the collection's conventions, which are measured, not assumed:

- nonfiction tag is **always exactly three letters**; some labels carry a
  trailing year (`641.568 SOU 2019`)
- fiction is **always** `SURNAME, First` with a comma; longest real surname is 9
  characters; only `VON X` multi-word surnames occur
- deep decimals are load-bearing: `635.048` and `635.0484` are different books
- **never guess** a character hidden by blur, spine curvature, or the frame edge
  — mark it `illegible`

Output goes to `verified/<img>.json`:

```json
{
  "image": "...", "section": "nonfiction",
  "labels":  [{"n": 1, "verdict": "ok", "text": "635 WEA"}],
  "missing": [{"row": 0, "betweenN": [7, 8], "text": "635 WHI", "note": "..."}],
  "confidence": "..."
}
```

## Then

```
node gt_merge.mjs --check    # refuses to write while any shelf is unverified
node gt_merge.mjs            # -> ground_truth_v3.json
```

The merge preserves **physical left-to-right order** — do not sort the rows.
`run_ordering_test.mjs` replays each row as a sequence and expects the app to
flag exactly the genuinely misfiled books; a key sorted into filing order would
silently erase every misshelving the test exists to catch.

Afterwards, point the harnesses at v3 (`site_suite.mjs`, `run_ordering_test.mjs`,
`replay_order.mjs`, `azure_recon_bench.mjs`, `audit_extras.mjs` all read
`ground_truth_v2.json` by name) and re-measure. Expect recall to DROP — the
denominator gets bigger and more honest — and precision to RISE sharply.
