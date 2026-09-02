# HH Goa 2026 — Task 3

Face-evidence pipeline: a local face embedding drives a live reverse-image
search, candidate images are independently re-matched locally, and the
resulting evidence bundle is fingerprinted and anchored on Polygon Amoy so its
integrity can be re-verified later.

> **Status: Phases 0-2 complete.** Local vision, live reverse-image discovery
> with isolated candidate retrieval, and independent face matching with
> ranking are implemented and tested. Evidence bundling, deterministic
> hashing and the blockchain anchor are not built yet.

## Pipeline

```text
input image → face detection → ArcFace embedding → reverse image search
  → candidates → retrieval → independent face matching → best match
  → evidence bundle → SHA-256 → Polygon Amoy → re-verification
```

## What Phase 0 covers

```text
input image
    ↓  validation (decodable, ≥64px, 3-channel BGR)
    ↓  SCRFD detection  → bbox, confidence, 5-point landmarks
    ↓  quality metrics  → Laplacian variance, brightness, contrast
    ↓  ArcFace (w600k_r50) via ONNX Runtime, CPU
    ↓
512-d L2-normalized embedding
```

## Install

```bash
uv venv --python 3.12.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
cp .env.example .env
```

Python 3.12 rather than 3.14: it is the version the InsightFace/ONNX stack is
most reliably packaged for. `insightface==1.0.1` ships a pure-Python wheel
(`py3-none-any`), so no MSVC toolchain is needed on Windows — earlier
releases were source-only and did require one.

## Model

`buffalo_l`, downloaded automatically to `~/.insightface/models/buffalo_l`
(~275 MB) on first use.

| role | file | architecture |
|---|---|---|
| detection | `det_10g.onnx` | **SCRFD-10GF** |
| recognition | `w600k_r50.onnx` | ArcFace, ResNet-50, 512-d |

**On RetinaFace:** the task specifies "RetinaFace through the selected
InsightFace model pack". The current `buffalo_l` pack does not ship
RetinaFace — it ships SCRFD, the successor detector from the same authors.
This is what the code actually runs and what it reports. No RetinaFace weights
are present in any current InsightFace pack.

Only the `detection` and `recognition` modules are loaded; the gender/age and
landmark models in the pack are skipped since nothing here uses them.

## Live discovery (Phase 1)

```bash
python main.py --image inputs/target.jpg --mode live
python main.py --image inputs/target.jpg --mode diagnostic   # replay a saved response
```

The documented local-image workflow, verified against the live API:

```text
local image
  -> POST https://serpapi.com/image        multipart, field "image", max 500 KB
  -> image_id (temporary)
  -> GET  https://serpapi.com/search?engine=google_lens&image_id=...
  -> visual_matches / organic_results
```

Responses are normalized behind `SearchProvider` / `GoogleLensProvider` into
`SearchCandidate` objects. Nothing outside `src/discovery/` touches SerpApi's
JSON shape.

**Live mode has no cached path.** `GoogleLensProvider` raises on failure and
the CLI prints `SEARCH FAILED` with a non-zero exit code — it never falls back
to `cache/`. Diagnostic mode is opt-in, prints `** REPLAYING A SAVED
RESPONSE **`, and is the only path that reads cached JSON. A search that
genuinely returns nothing is reported as zero candidates and exits 0.

### Candidate isolation

Every candidate is retrieved independently; a failure is a recorded status,
never an exception. Observed states:

| status | meaning |
|---|---|
| `RETRIEVED` | downloaded and decoded |
| `HTTP_403` / `HTTP_404` / `HTTP_ERROR` | non-200 from the host |
| `TIMEOUT` | connect or read deadline exceeded |
| `FETCH_FAILED` | DNS/TLS/connection error |
| `INVALID_IMAGE` | 200, but the body was not a decodable image |
| `TOO_LARGE` | body exceeded the byte cap |
| `NO_IMAGE_URL` | provider supplied neither image nor thumbnail |

Concurrency is bounded (`RETRIEVAL_CONCURRENCY`, default 5). Timeouts are
split into connect and read: `requests` applies a scalar timeout to each phase
separately, so a single value of N actually bounds a request at ~2N. A live
run hit exactly that — a 15 s setting produced a 30 s stall — which is why
`RETRIEVAL_CONNECT_TIMEOUT` and `RETRIEVAL_READ_TIMEOUT` are separate.

### Artifacts

Each run writes `evidence/TRACE-YYYYMMDD-XXXXXX/`:

```text
input.jpg              the image as submitted
search-request.json    provider, image_id, search_id, timestamp, live flag
search-response.json   the raw provider response, unmodified
candidates.json        normalized candidates
retrieval.json         per-candidate outcome, status, timing, bytes, sha256
matching.json          threshold, distribution, per-face similarities,
                       selected match and best independent match
```

## Face matching (Phase 2)

Each retrieved candidate is scored independently against the target
embedding:

```text
candidate image -> validate -> detect all faces -> per-face quality gate
  -> ArcFace embedding per face -> cosine similarity vs target
  -> threshold -> ranked candidates
```

### Multiple-face policy

Every usable face in a candidate is embedded and scored. The candidate takes
the **highest** per-face similarity and records which face won. When the image
contains more than one face the status is `MULTIPLE_FACE_MATCH`, never plain
`MATCH` — so the record states *"one face in this image matches the target"*
and never *"this image is the target"*. Ties resolve to the lower face index;
faces are ordered largest-first, so the larger face wins. Every per-face
similarity is stored, not just the winner.

### Statuses

`MATCH` (single face, at or above threshold) · `MULTIPLE_FACE_MATCH` (≥2
faces, best one clears it) · `REJECTED` (faces found, best below threshold) ·
`NO_FACE` · `LOW_QUALITY` (faces found, none passed the quality gate) ·
plus every retrieval status carried through unchanged (`HTTP_403`,
`HTTP_404`, `HTTP_ERROR`, `TIMEOUT`, `INVALID_IMAGE`, `FETCH_FAILED`,
`TOO_LARGE`, `NO_IMAGE_URL`).

Ranking uses measured identity similarity only — never URL, domain or search
position. The threshold comes from `FACE_MATCH_THRESHOLD` and is **never
lowered at runtime to force a match**; when nothing clears it the run says so.

### Re-finding the input

A reverse image search usually rediscovers the input file itself. Candidates
whose bytes hash to the input's SHA-256 are flagged `identical_to_input` and
reported as `[SAME FILE AS INPUT]`. That locates the source but is **not**
independent corroboration, so the run also reports `best_independent`: the
highest-scoring match that is a genuinely different file.

## Run

```bash
# bundled multi-face demo
.venv/Scripts/python.exe phase0.py

# a specific image
.venv/Scripts/python.exe phase0.py --image inputs/target.jpg

# compare two images
.venv/Scripts/python.exe phase0.py --image a.jpg --compare b.jpg
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

The identity tests need a small corpus of confirmed photographs:

```bash
.venv/Scripts/python.exe scripts/fetch_test_images.py fetch     # stage candidates
#   → look at tests/data/_candidates/, list the real ones in
#     tests/data/confirmed.json
.venv/Scripts/python.exe scripts/fetch_test_images.py promote   # accept them
```

Images come from Wikimedia Commons. Category membership does **not** imply a
photo depicts the named person — the categories contain drawings, statues and
photos of other people — so a human confirms each image before it enters the
corpus. Tests that need the corpus skip cleanly when it is absent.

## Threshold

`FACE_MATCH_THRESHOLD` is configuration, not a constant, and the value shipped
in `.env.example` is **provisional**. Cosine similarity between normalized
ArcFace embeddings is a distance measure — never report it as a probability.

```text
Similarity: 0.88   Decision: MATCH        ← correct
88% probability this is the person        ← wrong
```

Measured on the local corpus (9 confirmed photographs, 6 identities):

| | n | min | mean | max |
|---|---|---|---|---|
| same person | 4 | 0.3994 | 0.6013 | 0.7774 |
| different person | 32 | −0.1381 | −0.0150 | 0.0887 |

The two distributions do not overlap: any threshold in **0.0887 – 0.3994**
separates this corpus perfectly. The shipped default of **0.30** sits inside
that band and accepts 4/4 same-person pairs while rejecting 32/32
different-person pairs.

**Four same-person pairs is not a calibration.** Per-pair scores show how much
capture conditions matter:

```text
serena_williams 01-03   0.3994    2012 vs 2013, different hair
serena_williams 02-03   0.5438
serena_williams 01-02   0.6848    two frames, same press conference
barack_obama    01-02   0.7774    official portrait vs State of the Union
```

An earlier default of 0.40 was rejecting the 0.3994 pair — a true match. Widen
the corpus before trusting any value here.

Reproduce with:

```bash
.venv/Scripts/python.exe scripts/threshold_report.py
```

## Performance

Warm path, CPU, single process (`scripts/benchmark.py`):

```text
n=9  median=253 ms  mean=361 ms  min=211 ms  max=935 ms
```

Cost scales with the number of faces, not image size — the 935 ms case is an
8-face crowd shot. Model load adds ~1.5 s once per process. GPU was not used:
`onnxruntime-gpu` would mean a CUDA/cuDNN version-matching exercise for no
meaningful gain at this scale.

## Configuration

See `.env.example`. `.env` is gitignored and must never be committed.

## Limitations

1. Public web indexing is required; private or unindexed content is invisible.
2. Some sites block automated retrieval.
3. Image compression reduces similarity.
4. ArcFace similarity is **not** proof of legal identity.
5. A blockchain anchor proves the evidence fingerprint is unaltered — not that
   the source was truthful.
6. Tightly-cropped, pre-aligned face images (e.g. 112×112 thumbnails) often
   fail detection, because SCRFD expects surrounding context.
7. Polygon Amoy is a testnet.

## Privacy

Face embeddings are computed locally and never leave the machine. They are not
written to the evidence bundle and never go on-chain.
