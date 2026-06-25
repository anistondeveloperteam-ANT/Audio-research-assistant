/**
 * P3 — sidebar history is grouped by the user's LOCAL calendar day (not UTC), and every chat row
 * shows its EXACT date so nothing lands in a vague catch-all.
 *
 * The logic lives inside an IIFE in webapp/static/app.js (not exported), so this test keeps a
 * VERBATIM mirror of toEpochSec/localMidnight/bucketLabel/rowDate and asserts the boundary behavior.
 * Keep in sync with app.js — the app.js comment points back here.
 *
 * Run: node tests/test_date_grouping.js   (exit 0 = pass, 1 = fail)
 */
"use strict";
const assert = require("node:assert");

// ---- VERBATIM mirror of webapp/static/app.js (renderSessions helpers) -------------------------
const MONTHS = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"];
const MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
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
  const d = new Date(sec * 1000);
  const dayMs = localMidnight(d).getTime();
  const y = localMidnight(new Date(todayMs)); y.setDate(y.getDate() - 1);
  const w = localMidnight(new Date(todayMs)); w.setDate(w.getDate() - 7);
  const m = localMidnight(new Date(todayMs)); m.setDate(m.getDate() - 30);
  if (dayMs >= todayMs) return "Today";
  if (dayMs >= y.getTime()) return "Yesterday";
  if (dayMs >= w.getTime()) return "Previous 7 days";
  if (dayMs >= m.getTime()) return "Previous 30 days";
  return MONTHS[d.getMonth()] + " " + d.getFullYear();   // e.g. "May 2026"
}
function rowDate(ts) {
  const sec = toEpochSec(ts);
  if (!isFinite(sec)) return "";
  const d = new Date(sec * 1000);
  if (localMidnight(d).getTime() >= localMidnight(new Date()).getTime()) {
    let h = d.getHours(); const mm = String(d.getMinutes()).padStart(2, "0");
    const ap = h < 12 ? "AM" : "PM"; h = h % 12 || 12;
    return h + ":" + mm + " " + ap;
  }
  return MONTHS_SHORT[d.getMonth()] + " " + d.getDate();
}
// -----------------------------------------------------------------------------------------------

const nowSec = Date.now() / 1000;
let passed = 0;
function check(name, got, want) {
  assert.strictEqual(got, want, `${name}: expected "${want}", got "${got}"`);
  passed += 1;
}
function checkMatch(name, got, re) {
  assert.ok(re.test(got), `${name}: "${got}" did not match ${re}`);
  passed += 1;
}

// ---- bucketLabel: LOCAL-day buckets, newest first --------------------------------------------
// A conversation created NOW is "Today" (the original bug grouped today's UTC-evening chats as
// "Yesterday" for users behind UTC — the core regression this fix addresses).
check("now -> Today", bucketLabel(nowSec), "Today");

// Just after / before local midnight today is still "Today" in LOCAL time (DST/offset-robust).
const localMidnightTodaySec = localMidnight(new Date()).getTime() / 1000;
check("local 00:05 today -> Today", bucketLabel(localMidnightTodaySec + 5 * 60), "Today");
check("local 23:55 today -> Today", bucketLabel(localMidnightTodaySec + (23 * 60 + 55) * 60), "Today");

// Calendar-day boundaries (local-noon anchors so a fractional UTC offset can't shift the day).
const noon = localMidnightTodaySec + 12 * 3600;
check("yesterday -> Yesterday", bucketLabel(noon - 24 * 3600), "Yesterday");
check("2 days ago -> Previous 7 days", bucketLabel(noon - 2 * 24 * 3600), "Previous 7 days");
check("3 days ago -> Previous 7 days", bucketLabel(noon - 3 * 24 * 3600), "Previous 7 days");
check("10 days ago -> Previous 30 days", bucketLabel(noon - 10 * 24 * 3600), "Previous 30 days");
check("25 days ago -> Previous 30 days", bucketLabel(noon - 25 * 24 * 3600), "Previous 30 days");

// Older than 30 days -> "Month YYYY" (no vague catch-all). Compute the expected label the same way.
const old = new Date((noon - 45 * 24 * 3600) * 1000);
check("45 days ago -> Month YYYY", bucketLabel(noon - 45 * 24 * 3600),
      MONTHS[old.getMonth()] + " " + old.getFullYear());

// Defensive normalization: accidental milliseconds and numeric strings still resolve to Today;
// garbage falls back to "Earlier" (never throws, never mislabels as a real bucket).
check("milliseconds -> Today", bucketLabel(nowSec * 1000), "Today");
check("numeric string -> Today", bucketLabel(String(nowSec)), "Today");
check("NaN -> Earlier", bucketLabel("not-a-date"), "Earlier");
check("null -> Earlier", bucketLabel(null), "Earlier");

// ---- rowDate: each chat's exact date on its row ----------------------------------------------
checkMatch("today row -> a time", rowDate(nowSec), /^\d{1,2}:\d{2} (AM|PM)$/);
const d3 = new Date((noon - 3 * 24 * 3600) * 1000);
check("3 days ago row -> 'Mon D'", rowDate(noon - 3 * 24 * 3600),
      MONTHS_SHORT[d3.getMonth()] + " " + d3.getDate());
check("invalid row -> empty", rowDate("not-a-date"), "");

console.log(`test_date_grouping.js: ${passed} assertions passed`);
