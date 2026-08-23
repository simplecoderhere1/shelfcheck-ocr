// The call counter for the Azure free tier, as a Durable Object.
//
// WHY A DURABLE OBJECT AND NOT KV. One shelf photo is not one Azure call — it
// is the full frame plus up to nine band segments plus a zoom sheet, fired in
// PARALLEL. A KV counter is read-modify-write with no locking, so ten
// simultaneous calls all read the same number and all write count+1: the
// budget would be spent roughly ten times faster than it was counted, which
// defeats the whole point of counting. A Durable Object is a single instance
// with an input gate, so the read and the write either side of an `await` are
// not interleaved with another request. The count is exact.
//
// It also costs nothing to keep exact here: this object handles a handful of
// requests per shelf and nothing else.

// Azure AI Vision F0 (free) allowances, from the pricing page.
const MONTH_LIMIT  = 5000;
const MINUTE_LIMIT = 20;

// Stop short of the ceiling. A call can be counted here and then lost on the
// wire, or counted by Azure and not by us if this object is unreachable, so
// the two numbers drift. Overshooting means Azure answers 403 to a volunteer
// standing at a shelf; stopping early means we hand that shelf to Gemini a
// few reads sooner. The second is the cheaper mistake.
const MONTH_BUDGET = 4900;

// Two different numbers, because they answer two different questions.
//
// RESERVE is the WORST a photo can cost: 1 full frame + 3 bands x 3 segments +
// 1 zoom sheet. The app is told to switch engines while at least this much is
// left, so it never starts a shelf it cannot finish — a shelf read with three
// of its calls refused is worse than one read by Gemini from the start.
//
// TYPICAL is what a photo actually costs. Nonfiction bands now ship PACKED
// into one call per band instead of one per segment (encodeBandPack), which
// dropped a nonfiction shelf from 1 full + 4.5 band-segment calls + 1 closer
// look = 6.5 down to 1 + 1.5 packed + 1 = 3.5. Fiction never tiles or bands
// (see the revert note in runOcr), so it costs 1 full + 1 closer look = 2.
// Blended by the corpus mix (17 nonfiction / 19 fiction shelves), that is
// about 2.7 calls/photo. Reporting "photos left" against the worst case
// would understate the real allowance by 4x (445 shelves rather than about
// 1,800) and make the app look far closer to the wall than it is.
const PHOTO_RESERVE = 11;
const PHOTO_TYPICAL = 2.7;

const monthKey = (d = new Date()) =>
  `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`;

// First instant of next UTC month — when Azure's allowance resets and this
// counter goes back to zero.
function monthReset(d = new Date()) {
  return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1)).toISOString();
}

export class Quota {
  constructor(ctx) {
    this.ctx = ctx;
    // The per-minute window lives in memory on purpose. It describes the last
    // 60 seconds, so there is nothing worth persisting: if this object is
    // evicted, the burst it was tracking is over.
    this.recent = [];
  }

  async load() {
    const month = monthKey();
    const rec = (await this.ctx.storage.get('m')) || { month, used: 0 };
    // A new month is a fresh allowance. Detected by comparing the stored
    // stamp rather than by a scheduled reset, so it works even if nobody
    // opened the app on the 1st.
    if (rec.month !== month) return { month, used: 0 };
    return rec;
  }

  status(rec) {
    return {
      month: rec.month,
      used: rec.used,
      limit: MONTH_LIMIT,
      budget: MONTH_BUDGET,
      remaining: Math.max(0, MONTH_BUDGET - rec.used),
      photosLeft: Math.floor(Math.max(0, MONTH_BUDGET - rec.used) / PHOTO_TYPICAL),
      enoughForAPhoto: MONTH_BUDGET - rec.used >= PHOTO_RESERVE,
      reset: monthReset(),
    };
  }

  async fetch(request) {
    const op = new URL(request.url).pathname;
    const rec = await this.load();

    if (op === '/status') return Response.json(this.status(rec));

    // /take — reserve one call, BEFORE it is made. Azure counts a call the
    // moment it arrives, so a request that is sent and then lost still spends
    // the allowance; counting after a successful response would undercount by
    // exactly the calls most likely to be retried.
    if (op === '/take') {
      const now = Date.now();
      this.recent = this.recent.filter(t => now - t < 60000);
      if (this.recent.length >= MINUTE_LIMIT) {
        return Response.json({
          ok: false, scope: 'minute',
          retryAfter: Math.ceil((60000 - (now - this.recent[0])) / 1000),
          ...this.status(rec),
        });
      }
      if (rec.used >= MONTH_BUDGET) {
        return Response.json({ ok: false, scope: 'month', ...this.status(rec) });
      }
      rec.used++;
      this.recent.push(now);
      await this.ctx.storage.put('m', rec);
      return Response.json({ ok: true, ...this.status(rec) });
    }

    // /spent — Azure itself said the allowance is gone (403 out-of-quota).
    // It is the authority, not this counter, so believe it and jump the
    // stored count to the budget instead of waiting to drift up to it.
    if (op === '/spent') {
      rec.used = Math.max(rec.used, MONTH_BUDGET);
      await this.ctx.storage.put('m', rec);
      return Response.json(this.status(rec));
    }

    return new Response('no such op', { status: 404 });
  }
}

export const QUOTA_CONST = {
  MONTH_LIMIT, MINUTE_LIMIT, MONTH_BUDGET, PHOTO_RESERVE, PHOTO_TYPICAL,
};
