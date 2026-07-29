# Release and License Matrix

This matrix separates software licenses from dataset-content permissions. A
repository license is not treated as permission to redistribute third-party
images embedded in, or referenced by, a benchmark. This is a release-planning
record, not legal advice. Terms should be checked again against the downloaded
version immediately before public release.

## Author-Generated Materials

| Material | Contains source content? | Public-release status | Planned action |
|---|---|---|---|
| Selector, backend, audit, and construction code | No | Released under Apache License 2.0 | Publish source and environment lockfile; preserve notices for reused code. |
| Aggregate tables, confidence intervals, and timing summaries | No row-level benchmark content | Releaseable | Include in the anonymous and public artifacts. |
| TextOCR-Hard construction recipe and deterministic seeds | No images; may name source fields | Releaseable | Publish code, constraints, seeds, and hashes. |
| TextOCR-Hard row-level probes | Image identifiers, OCR strings, boxes, and derived decoys | Hold | Generate locally from official TextOCR downloads; do not redistribute until source terms are confirmed. |
| Hard-negative human-QC rows | Source/decoy strings and source-image references | Hold | Release aggregate pass counts and protocol; review row-level release with TextOCR terms. |
| TextVQA/DocVQA multi-region boxes | Derived boxes tied to source questions and images | Hold | Release aggregate agreement/coverage results; review row-level release against both source datasets. |
| Model scores, masks, and pruning traces | May contain prompts, source strings, image identifiers, or coordinates | Filtered | Release compact aggregate evidence by default; sanitize any row-level traces before release. |
| Manuscript figures and method diagram | Author-generated | Releaseable | Include with the paper under the paper's publication terms. |

No benchmark images, source annotation dumps, or model checkpoints are included
in the artifact.

## Source Datasets

| Source | Verified public information | Redistribution decision for this artifact | Recheck before release |
|---|---|---|---|
| TextOCR | The official [TextOCR/TextVQA site](https://textvqa.org/textocr/dataset/) distributes the dataset, but an explicit dataset-content redistribution license was not available in the accessible release metadata checked on 2026-07-21. | Do not redistribute images, OCR dumps, or row-level derived probes. | Official download terms and any terms inherited from source images. |
| TextVQA | The official [TextVQA site](https://textvqa.org/) states that images come from OpenImages and distributes questions/answers, but it does not display a single content license on the accessible overview page. | Do not redistribute images, question dumps, or row-level derived boxes. | Dataset download terms plus applicable OpenImages licenses/attribution. |
| DocVQA Task 1 | The official [DocVQA dataset page](https://site.docvqa.org/datasets) requires registration and instructs users to read the download-page terms; images originate from the UCSF Industry Documents Library. | Do not redistribute images, QA dumps, or row-level derived boxes. | RRC download agreement and source-document terms. |
| OCRBench | The official [OCRBench repository](https://github.com/qywh2023/OCRBench) is MIT-licensed, but OCRBench aggregates visual material from multiple tasks and the repository license is not assumed to override source-image terms. | Release evaluation code and aggregate results only; omit benchmark images and raw QA records. | Notices and terms for each included OCRBench source task. |
| GSR-Bench / What'sUp-derived data | GSR-Bench extends the previously released What'sUp benchmark; a single redistribution license covering all evaluated image content was not verified. | Omit images and row-level benchmark records. | Official release location and source-image terms. |

## Models and External Implementations

| Dependency | Verified license/status | Artifact treatment |
|---|---|---|
| Qwen3-VL-8B-Instruct | Apache-2.0 in the official/local model card | Checkpoint excluded; users download it from the official distributor. |
| InternVL3.5-8B-HF | Apache-2.0 in the official/local model card | Checkpoint excluded; users download it from the official distributor. |
| LLaVA-1.5-7B-HF | The official [Hugging Face model card](https://huggingface.co/llava-hf/llava-1.5-7b-hf) labels the checkpoint `llama2`. | Checkpoint excluded; users accept the applicable model terms at download. |
| SCOPE port | Public source pinned by commit; the vendored LLaVA license file is Apache-2.0 | Do not vendor upstream code in the release archive; provide commit and parity instructions. |
| AnchorPrune port | Official source is Apache-2.0 and pinned to commit `2e5d965a0e7291e46eeda73d678529d641ef74d2` | Release the adapted selector with upstream attribution; omit the checkout and CLIP weights, and provide exact-index parity instructions. |
| VisionZip and CDPruner checkouts | Local upstream checkouts include Apache-2.0 license files; VisionZip is pinned to commit `8f86b55c6f000eb033e6912538af2dd7dcb30502` | Do not vendor checkouts; provide source links/commits and preserve notices for adapted portions. |
| CoIn | No public implementation was linked at evaluation time; this work uses a paper-algorithm port | Release only the independently written port, clearly labeled as paper-based rather than code-parity verified. |

## Release Gate

Before creating a public archive:

1. Include the Apache License 2.0 file for author-generated code and preserve
   all notices required by reused components.
2. Recheck every official dataset download agreement and record its version and
   access date.
3. Keep source images, source annotation dumps, checkpoints, and held row-level
   derivatives outside the archive unless written redistribution permission is
   confirmed.
4. Scan all text artifacts for local paths, identities in anonymous packages,
   benchmark strings, image identifiers, and coordinates.
5. Publish a manifest of included files and their SHA-256 hashes.
