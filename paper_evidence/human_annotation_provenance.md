# Human Annotation Provenance

This record distinguishes assisted annotation, human verification, independent
calibration, and final adjudication for the paper-facing evidence artifacts.

## Hard-Negative Quality Control

- Scope: all 500 paired source words in each TextOCR-Hard split.
- Tool assistance prepared candidate review fields. A primary human annotator
  visually checked source readability, queried-target absence, and source-box
  agreement for every row.
- Development outcome: 486 targets were confirmed absent and 449 pairs met the
  stricter source-visible, target-absent, box-matched criterion.
- Locked-confirmation outcome: 465 pairs met the same strict criterion, 34
  sources were unreadable, and one queried target was present.
- For each split, a second non-author annotator independently labeled a frozen
  100-row subset before adjudication. Development valid-versus-non-valid
  agreement was 0.860 (Cohen's kappa 0.453), with all 22 disagreements
  adjudicated. Confirmation agreement was 0.900 (kappa 0.445), with all 10
  disagreements adjudicated.
- Because each positive probe queries the same source word and uses the same
  box as its paired negative, source readability and box--text agreement also
  audit positive-probe readability and localization.
- Claim boundary: the prespecified 500-image confirmation readout remains
  primary; the 465-pair result is a post-QC sensitivity analysis.

## Multi-Region Evidence

- Primary scope: 96 samples, split evenly between the fixed 500-sample TextVQA
  and DocVQA boundary splits.
- Primary boxes used OCR/tool-assisted candidates; human annotator A inspected
  every row and corrected boxes where needed.
- Calibration scope: 12 overlapping samples independently annotated by human
  annotator B without access to the primary boxes.
- Before adjudication, box count and label-type set each matched on 5/12 rows;
  mean greedy matched-box IoU was 0.3404 under a 0.50 threshold.
- The strict all-box agreement rule sent all 12 rows to adjudication. Following
  human confirmation, annotator B's boxes were adopted for all 12 rows.
- A third annotator independently labeled a disjoint 20-row extension without
  access to the primary boxes. Across all 32 independently double-labeled rows,
  box count and label-type sets match on 12 and 16 rows; mean greedy matched-box
  IoU is 0.3928 and union-region IoU is 0.3887. Answer-value presence agrees on
  30/32 rows (F1 0.9677), while context/header role boundaries are less stable.
- The 20-row extension is retained as a pre-adjudication reliability audit and
  does not replace boxes in the final 96-row package. Any replacement requires
  an explicit subsequent human adjudication decision.
- Final outcome: 96 annotated rows, no unresolved or empty rows, no invalid or
  unlabeled boxes, and 96/96 rows ready for ECR projection.
- Claim boundary: the result is a human-corrected 96-sample stress audit of
  evidence availability. Only the original 12 calibration rows were explicitly
  adjudicated; the study is not full-benchmark annotation or proof of causal use.
