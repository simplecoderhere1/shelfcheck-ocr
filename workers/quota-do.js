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

// A photo costs about this many calls: 1 full frame + up to 9 band segments +
// 1 zoom sheet. Below this there is not enough left to read a whole shelf, so
// the app is told to switch engines BEFORE it starts rather than half way
// through — a shelf read with three of its ten calls refused is worse than one
// read by Gemini from the start.
const PHOTO_COST = 12;

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
      photosLeft: Math.floor(Math.max(0, MONTH_BUDGET - rec.used) / PHOTO_COST),
      enoughForAPhoto: MONTH_BUDGET - rec.used >= PHOTO_COST,
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

export const QUOTA_CONST = { MONTH_LIMIT, MINUTE_LIMIT, MONTH_BUDGET, PHOTO_COST };
