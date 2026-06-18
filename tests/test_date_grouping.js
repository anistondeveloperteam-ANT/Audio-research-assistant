/**
 * P3 — sidebar history is grouped by the user's LOCAL calendar day, not UTC.
 *
 * The grouping logic lives inside an IIFE in webapp/static/app.js (not exported), so this test
 * keeps a VERBATIM mirror of toEpochSec/localMidnight/bucketLabel and asserts the boundary
 * behavior. Keep in sync with app.js — the app.js comment points back here.
 *
 * Run: node tests/test_date_grouping.js   (exit 0 = pass, 1 = fail)
 */
"use strict";
const assert = require("node:assert");

// ---- VERBATIM mirror of webapp/static/app.js (renderSessions helpers) -------------------------
function toEpochSec(ts) {
  let n = typeof ts === "number" ? ts : parseFloat(ts);
  if (!isFinite(n)) return NaN;
  if (n > 1e12) n = n / 1000;                 // value looks like milliseconds -> seconds
  return n;
}
function localMidnight(d) { const x = new Date(d.getTime()); x.setHours(0, 0, 0, 0); return x; }
function bucketLabel(ts) {
  const sec = toEpochSec(ts);
  if (!isFinite(sec)) return "Earlier";
  const todayMs = localMidnight(new Date()).getTime();
  const dayMs = localMidnight(new Date(sec * 1000)).getTime();
  const y = localMidnight(new Date(todayMs)); y.setDate(y.getDate() - 1);
  const w = localMidnight(new Date(todayMs)); w.setDate(w.getDate() - 7);
  if (dayMs >= todayMs) return "Today";
  if (dayMs >= y.getTime()) return "Yesterday";
  if (dayMs >= w.getTime()) return "Previous 7 days";
  return "Earlier";
}
// -----------------------------------------------------------------------------------------------

const nowSec = Date.now() / 1000;
let passed = 0;
function check(name, got, want) {
  assert.strictEqual(got, want, `${name}: expected "${want}", got "${got}"`);
  passed += 1;
}

// A conversation created NOW is "Today" (the original bug grouped today's UTC-evening chats as
// "Yesterday" for users behind UTC — the core regression this fix addresses).
check("now -> Today", bucketLabel(nowSec), "Today");

// Just after local midnight today (00:05) is still "Today" in local time even though, for a
// negative-UTC-offset user, that instant is the PREVIOUS UTC day.
const localMidnightTodaySec = localMidnight(new Date()).getTime() / 1000;
check("local 00:05 today -> Today", bucketLabel(localMidnightTodaySec + 5 * 60), "Today");

// Just before local midnight today (23:55) is "Today" too — the symmetric case (this instant is
// the NEXT UTC day for a positive-UTC-offset user).
check("local 23:55 today -> Today", bucketLabel(localMidnightTodaySec + (23 * 60 + 55) * 60), "Today");

// Calendar-day boundaries (use local-noon anchors so a fractional UTC offset can't shift the day).
const localNoonTodaySec = localMidnightTodaySec + 12 * 3600;
check("yesterday noon -> Yesterday", bucketLabel(localNoonTodaySec - 24 * 3600), "Yesterday");
check("3 days ago -> Previous 7 days", bucketLabel(localNoonTodaySec - 3 * 24 * 3600), "Previous 7 days");
check("10 days ago -> Earlier", bucketLabel(localNoonTodaySec - 10 * 24 * 3600), "Earlier");

// Defensive normalization: accidental milliseconds and numeric strings still resolve to Today;
// garbage falls back to "Earlier" (never throws, never mislabels as a real bucket).
check("milliseconds -> Today", bucketLabel(nowSec * 1000), "Today");
check("numeric string -> Today", bucketLabel(String(nowSec)), "Today");
check("NaN -> Earlier", bucketLabel("not-a-date"), "Earlier");
check("null -> Earlier", bucketLabel(null), "Earlier");

console.log(`test_date_grouping.js: ${passed} assertions passed`);
