/**
 * Apply the audited corrections to the final-project deck.
 *
 * HOW TO RUN
 *   1. Open the presentation → Extensions → Apps Script
 *   2. Delete whatever is in Code.gs, paste this whole file
 *   3. Press Run (▶). First run asks for authorisation — approve it.
 *   4. Read the Execution log. Every line shows how many replacements were made.
 *      A line ending in "-> 0" means the text did NOT match and that fix did not
 *      apply; fix that one by hand rather than assuming it worked.
 *
 * SAFE TO RE-RUN. Every replacement is idempotent: running twice changes nothing
 * the second time, because the "find" text no longer exists after the first pass.
 *
 * TO UNDO EVERYTHING: File → Version history → restore the version from before.
 */

function applyAuditFixes() {
  var deck = SlidesApp.getActivePresentation();
  var log = [];

  // Helper: replace, report the count, and flag misses loudly.
  function fix(label, find, replace, matchCase) {
    var n = deck.replaceAllText(find, replace, matchCase !== false);
    log.push((n === 0 ? '!! MISS  ' : '   ok    ') + label + '  -> ' + n);
    return n;
  }

  // ── SLIDE 14 — A1's speaker cosine is NEGATIVE ──────────────────────────
  // -0.020, not 0.020. A negative correlation is a stronger claim than a small
  // positive one, so the sign matters. Distinctive context avoids hitting the
  // 0.020 that may appear elsewhere.
  fix('S14 cosine sign (prose)',
      'speaker cosine similarity of 0.020',
      'speaker cosine similarity of −0.020');

  // ── SLIDE 14 — MCD caveat lost its en-dash ──────────────────────────────
  fix('S14 MCD range 4-8 dB',
      'published literature benchmarks of 48 dB',
      'published literature benchmarks of 4–8 dB');

  // ── SLIDE 14 — it is the trainer's eval split, not a validation set ──────
  fix('S14 eval loss wording',
      'driving validation loss from 3.825',
      'driving eval loss from 3.825');

  // ── SLIDE 13 — "engineered to fail" overclaims ──────────────────────────
  // You did not sabotage A1; it was a genuine attempt expected to underperform.
  // "How did you engineer it to fail?" has no good answer.
  fix('S13 engineered-to-fail',
      'mathematically engineered to fail to establish a clean performance floor',
      'expected to underperform, establishing a clean performance floor — and it '
      + 'failed more completely than predicted (see the post-mortem)');

  // ── SLIDE 15 — near-silent, with the threshold that makes it checkable ───
  fix('S15 near-silent wording',
      '7 silent, 5 heavily saturated',
      '7 near-silent (rms ≤ 0.008), 5 saturated (peak = 1.000)');

  // ── SLIDE 8 — match slide 9's precision ─────────────────────────────────
  fix('S8 EER precision', '0.67%', '0.668%');

  // ── SLIDE 18 — cross_test.py is listed as your contribution on slide 10 ──
  fix('S18 add cross_test',
      'features_ssl · backend_keras · baseline_cnn',
      'features_ssl · backend_keras · baseline_cnn · cross_test');

  // ── SLIDE 16 — the two rows are NOT scored on the same fakes ────────────
  // SSL: all 36 generated clips. CNN: only the 12 fine-tuned XTTS clips.
  fix('S16 asymmetry disclosure',
      'Real speech score of 0.0009 signals high-confidence false spoof classification.',
      'Real speech score of 0.0009 signals high-confidence false spoof classification. '
      + 'NOT a like-for-like row comparison: the SSL row is scored on all 36 generated '
      + 'clips, the CNN row on only the 12 fine-tuned XTTS clips. The asymmetry runs '
      + 'against the CNN row being flattered — it faced only the hardest fakes, while '
      + 'the SSL row includes 12 A1 clips that are noise and trivially flagged.');

  // ── SLIDE 9 — the external comparison, corrected ────────────────────────
  // VERIFIED against arXiv:2011.01108 (read from the paper's prose):
  //   "the pooled min t-DCF for the baseline is 0.09, whereas the best performing
  //    S2 RawNet2 system ... gives a min t-DCF of 0.1175"
  //   "Results for all three RawNet classifiers are inferior to those of the baseline"
  //   Table 3: the high-spectral-resolution baseline (L) gives min t-DCF 0.0904
  // The previous EER figures (4.7% / 0.2%) could NOT be verified: 4.66% is the
  // ASVspoof *2021* RawNet2 baseline, and Tak et al. 2022 reports on ASVspoof 2021,
  // not 2019 LA - so that row was not comparing the same benchmark.
  fix('S9 header -> min t-DCF',
      '03 / EXTERNAL COMPARISON AGAINST PUBLISHED NUMBERS',
      '03 / EXTERNAL COMPARISON — ASVspoof 2019 LA eval, pooled min t-DCF');
  fix('S9 row1 label', 'RawNet2 (official baseline)', 'RawNet2, best single variant (S2)');
  fix('S9 row1 value', '4.7%', '0.1175');
  fix('S9 row2 label', 'wav2vec2  AASIST', 'LFCC-GMM high-resolution baseline');
  fix('S9 row2 value', '0.2%', '0.0904');
  fix('S9 row2 source', 'Tak et al. 2022', 'Tak et al. 2021 (ibid.)');
  fix('S9 ours label', 'This Work (frozen, no augment)', 'This work — XLS-R + Keras back-end');
  fix('S9 ours value', '0.668%', '0.0189');

  Logger.log('=== AUDIT FIXES ===\n' + log.join('\n')
    + '\n\nAny line marked "!! MISS" did not apply - the text on the slide differs '
    + 'from what was searched for. Fix those by hand.'
    + '\n\nSLIDE 9 STILL NEEDS A MANUAL LINE. Add under the table:'
    + '\n  "Compared on min t-DCF, not EER. Note the RawNet2 paper\'s own finding: '
    + 'standalone RawNet2 is INFERIOR to the LFCC-GMM baseline on pooled metrics; its '
    + 'value is on attack A17 and in fusion. Published SSL results (Tak et al. 2022) '
    + 'report on ASVspoof 2021 and are deliberately excluded as not comparable."'
    + '\n\nAlso confirm the slide-9 column header now reads "min t-DCF", not "Eval EER".');
}
