# Data and Annotation Statement

## Redistribution Boundary

This artifact does not include source benchmark images, model checkpoints, or
source-dataset annotation dumps. Users obtain TextOCR, OCRBench, TextVQA,
DocVQA, GSR-Bench, and model weights from their official distributors and
remain responsible for the corresponding terms. The supplied scripts generate
probe records, masks, and audit summaries from locally available source data.

The compact `paper_evidence/` files contain aggregate measurements and protocol
provenance only. Before public release of row-level derived annotations, their
redistribution terms should be checked against every source dataset; aggregate
tables and construction code can be released independently. The component-level
decision record is maintained in `RELEASE_AND_LICENSE_MATRIX.md`.

## Locked TextOCR-Hard Construction

The development and confirmation sets use the same region filtering, token
sampling, pairing, and output schema. They explicitly retain two construction
versions because the confirmation builder added Unicode-safe decoy candidate
handling after the development set had been locked. `development-v1` with seed
17 and `confirmation-v2` with seed 20260720 reproduce the published SHA-256
digests. The artifact's commands assert both hashes and fail on protocol drift;
the full lock record is in `LOCKED_TEXTOCR_HARD_PROTOCOL.md`.

## Hard-Negative Quality Control

- Scope: all 500 paired source words in each TextOCR-Hard split.
- Tool assistance prepared candidate fields; a primary human annotator visually
  checked source visibility, target absence, and evidence-box correctness for
  every row in both development and locked confirmation.
- Development outcome: target absence was confirmed for 486 rows; 449 also met
  the stricter source-visible, target-absent, and matching-box criterion.
- Locked-confirmation outcome: 465 pairs met the same strict criterion, 34
  sources were unreadable, and one queried target was present.
- A second non-author annotator independently labeled a frozen 100-row subset
  from each split before adjudication. Valid-versus-non-valid agreement was
  0.860 (Cohen's kappa 0.453) on development and 0.900 (kappa 0.445) on
  confirmation; all 22 and 10 respective disagreements were adjudicated.
- Restricting model readouts to the confirmed-absence or strict-valid subsets
  changed development hFPR by at most 0.013 and confirmation hFPR by at most
  0.009. The prespecified 500-image confirmation readout remains primary.
- Each positive probe queries the same source word and box as its paired
  negative. Source readability and box--text agreement therefore also audit
  positive-probe readability and localization.

## Multi-Region Evidence Audit

- Scope: 96 samples, evenly divided between the fixed 500-sample TextVQA and
  DocVQA boundary splits.
- A primary human annotator inspected and corrected tool-assisted boxes for all
  96 samples.
- A second human annotator independently labeled 12 overlapping calibration
  samples without access to the primary boxes.
- Before adjudication, box count and label-type set each matched on 5/12 rows;
  greedy matched-box IoU averaged 0.3404.
- The strict all-box gate sent all 12 rows to review. After human confirmation,
  the secondary boxes were adopted as the final result for those 12 rows.
- A third annotator independently labeled a disjoint 20-row extension. Across
  all 32 double-labeled rows, box count and label-type sets match on 12 and 16
  rows; matched-box and union-region IoU average 0.3928 and 0.3887. Answer-value
  presence is shared on 30/32 rows (F1 0.9677), while context/header roles are
  less stable.
- The extension is retained as a pre-adjudication reliability audit and does
  not replace boxes in the final 96-row package.
- The final audit contains 96 resolved rows and no empty, invalid, or unlabeled
  regions. It is a stress audit of evidence availability, not full-benchmark
  annotation or proof that a model causally uses every retained region.

The annotators were non-author student colleagues and received no monetary
compensation. This low-risk technical annotation used only public benchmark
content, collected no personal, demographic, or sensitive information, and did
not require institutional ethics review. The artifact contains no annotator
identity.
