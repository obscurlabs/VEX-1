# HH Goa 2026 — Task 3

Face-evidence pipeline: a local face embedding drives a live reverse-image
search, candidate images are independently re-matched locally, and the
resulting evidence bundle is fingerprinted and anchored on Polygon Amoy so its
integrity can be re-verified later.

> **Status: complete and validated.** Every stage runs live: real Google Lens discovery, real candidate retrieval, real face matching, a deterministic evidence fingerprint anchored on Polygon Amoy, and independent on-chain verification with tamper detection.

## Pipeline

```text
input image → face detection → ArcFace embedding → reverse image search
  → candidates → retrieval → independent face matching → best match
  → evidence bundle → SHA-256 → Polygon Amoy → re-verification
```

## Local face pipeline

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

**Activate the venv before running anything:**

```text
Windows:        .venv\Scripts\activate
macOS / Linux:  source .venv/bin/activate
```

Every command below assumes it is active. Without it, a system `python` will
not have the dependencies — the entry points detect that and print which
interpreter is running and how to fix it, rather than an import traceback.

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

## Live discovery

```bash
python main.py --image inputs/target.jpg --mode live
python main.py --image inputs/target.jpg --mode diagnostic
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

**Those timeouts bound the sockets, not DNS.** `getaddrinfo` is a blocking OS
call, so an unresponsive resolver can hold a single candidate far longer — a
live run on 2026-09-03 recorded five candidates at ~83 s each during a
transient DNS outage, against an 18 s socket bound. Correctness is unaffected:
each was classified `FETCH_FAILED`, isolation held, and the run completed and
verified on chain. The cost is latency — that run took 105 s instead of the
usual ~30 s. Bounding this properly needs a resolver with its own timeout,
which is not worth the dependency here.

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

## Face matching

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

### The ranked set

One search yields many candidates, and collapsing them to a single answer
throws away corroboration. The pipeline keeps the whole ranked set:

```text
discovered     normalized candidates from the provider response
evaluated      candidates retrieval was attempted for
retrieved      images downloaded and decoded
face-matched   a face was embedded and scored against the target
qualifying     scored at or above the configured threshold
independent    qualifying, minus the input file rediscovered, and one
               representative per distinct source
```

Google Lens routinely returns several URLs backed by one source — three
thumbnail sizes of a video, a page listed twice under two image URLs.
Presenting those as separate corroboration would overstate how many
independent sources were found, so they are **grouped** behind the
highest-scoring member and the reason is recorded (`identical image bytes`,
`same page`, `same source domain`). Nothing is discarded: every folded match
stays in the group's `duplicates`, in `matching.json` and in the manifest.

Grouping is by exact domain string, so subdomains stay separate
(`en.wikipedia.org` and `pap.wikipedia.org` do not merge). Collapsing those
correctly needs a public-suffix list, which is not worth the dependency.

The default CLI lists the top 5 independent sources, then the selected match.
`selected_match` and `best_independent_match` keep their existing meanings —
`best_independent_match` is the same object as the first entry of the ranked
set, so nothing that read the old fields breaks.

Anchoring is unchanged: **one canonical fingerprint per investigation**,
covering the whole ranked set, not one transaction per match.

### Re-finding the input

A reverse image search usually rediscovers the input file itself. Candidates
whose bytes hash to the input's SHA-256 are flagged `identical_to_input` and
reported as `[SAME FILE AS INPUT]`. That locates the source but is **not**
independent corroboration, so the run also reports `best_independent`: the
highest-scoring match that is a genuinely different file.

## Evidence bundle

```text
evidence/TRACE-YYYYMMDD-XXXXXX/
├── input.jpg              the image as submitted
├── source-image.jpg       the anchored candidate, saved verbatim
├── search-request.json    provider, image_id, search_id, timestamp, live flag
├── search-response.json   the raw provider response, BYTE-FOR-BYTE
├── candidates.json        normalized candidates
├── retrieval.json         per-candidate outcome, status, timing, sha256
├── matching.json          per-face similarities, ranking, selection
├── manifest.json          the canonical manifest (stored AS the hashed bytes)
└── fingerprint.json       evidence_sha256 + algorithm + canonicalization spec
```

The bundle anchors the **best independent match** — the highest-scoring
candidate that is not the input file rediscovered. If the only match is the
input itself, that is bundled but flagged, not passed off as corroboration.

### Canonicalization

`manifest.json` is written as the exact bytes that were hashed, so there is no
ambiguity about what the fingerprint covers.

```text
json;sort_keys=true;separators=(',',':');ensure_ascii=false;
allow_nan=false;unicode=NFC;encoding=utf-8;floats=forbidden
```

**Floats are rejected outright.** `repr()` of a binary float is stable in
CPython, but "the same number" is ambiguous across languages and serializers —
0.9944 may be emitted as `0.9944`, `0.99440000000001` or `9.944e-1`. A
fingerprint other tools must reproduce cannot depend on that. Every real value
is quantized to a fixed-precision decimal **string** (`decimal_str`, 6 places
for similarity, 4 for detection scores) before it enters the manifest, and
`canonical_bytes()` raises on any raw float that slips through. Integers are
exact and stay JSON numbers. Strings are NFC-normalized so composed and
decomposed Unicode hash alike.

### What the hash covers

Every manifest field, **and** a SHA-256 of every other file in the bundle —
so altering any artifact, binary included, changes the fingerprint.

Deliberately **not** covered: face embeddings (biometric data is never
written to disk or hashed), API keys, private keys, `.env` contents, absolute
filesystem paths, and any wall-clock time describing the *process* rather than
the evidence. `created_at`, `requested_at` and `retrieved_at` **are** covered —
they state when the search ran and when the candidate was fetched, are written
once, and are read back verbatim on verification, never regenerated.

### Verify and tamper-test

```bash
python verify.py evidence/TRACE-20260902-F53AF4
python verify.py evidence/TRACE-20260902-F53AF4 --show-manifest
python scripts/tamper_test.py evidence/TRACE-20260902-F53AF4
```

Verification runs two independent checks, both required: every artifact's
on-disk digest must match the manifest, and re-canonicalizing `manifest.json`
must reproduce the recorded `evidence_sha256`. Exit 0 = `VERIFIED`, 1 =
`FAILED`.

## Blockchain anchoring

`contracts/IdentityAnchor.sol` — 975 bytes deployed, solc 0.8.26, optimizer on
(200 runs), EVM `paris`.

The chain does **no** face recognition, no search, no evidence processing. It
stores two 32-byte values plus the provenance the chain gives for free:

```solidity
struct EvidenceRecord {
    bytes32 evidenceHash;  // SHA-256 of the canonical manifest
    address submitter;
    uint64  timestamp;
    uint64  blockNumber;
}
```

Keyed by `keccak256(investigation_id)`. No images, no embeddings, no raw search
results, no personal data.

**Duplicate policy — rejected, both ways.** An investigation may be anchored
exactly once (`InvestigationAlreadyAnchored`), and an evidence hash may be
anchored exactly once (`EvidenceHashAlreadyAnchored`). A fingerprint can
therefore never map to two conflicting records. Re-running an investigation
needs a new investigation id, which is what a genuinely new observation is.
`investigationForHash()` resolves a fingerprint back to its record.

```bash
python scripts/deploy.py                              # deploy, writes build/deployment.json
python anchor.py evidence/TRACE-20260902-F53AF4       # anchor + read back + compare
python anchor.py evidence/TRACE-20260902-F53AF4 --verify-only   # no transaction
```

`anchor.py` recomputes the fingerprint from the bundle on disk, refuses to
anchor a bundle that does not verify locally, sends the transaction, waits for
a real receipt, reads the hash back, and compares. `anchor.json` is written
into the bundle but is **not** covered by the evidence fingerprint — the anchor
references the evidence, never the other way round.

### Safety

The private key is read from the environment, never logged, never returned,
never written to an artifact; only the derived public address is printed.
`WalletConfigError` messages deliberately omit the key. Chain id is asserted
against 80002 before any write. Balance is checked before broadcasting.
Receipt waiting is bounded (`TX_RECEIPT_TIMEOUT`, default 180 s) and a delayed
confirmation is reported as delayed with its transaction hash — never as
success, and never re-broadcast.

Polygon is proof-of-authority, so `ExtraDataToPOAMiddleware` is required;
without it every `get_block()` fails on the 105-byte `extraData`.

## End-to-end verification

```bash
python verify_chain.py evidence/TRACE-20260902-F53AF4
python verify_chain.py <bundle> --contract 0x...   # explicit contract
python verify_chain.py <bundle> --json             # machine-readable
```

Read-only: **no private key is required**. Anyone holding a bundle can check
it against the chain.

### Data path

```text
artifact files          -> SHA-256 each, compared to manifest.artifacts
manifest.json           -> canonical bytes -> SHA-256  = LOCAL FINGERPRINT
investigation_id        -> keccak256                   = on-chain record key
IdentityAnchor.getEvidence(key)                        = ON-CHAIN FINGERPRINT
local == on-chain                                      -> VERIFIED
```

`fingerprint.json` is **not** the source of truth. The fingerprint is always
recomputed from the manifest on disk; the cached value is shown only as a
cross-check, because anyone who edited the manifest would have edited that
file too. A deliberately falsified `fingerprint.json` does not change the
verdict — there is a test for exactly that.

### Two detection layers

Tampering is caught by whichever layer it touches:

| what changed | manifest SHA-256 | caught by | status |
|---|---|---|---|
| a field inside the manifest | **changes** | hash vs chain | `HASH_MISMATCH` |
| a covered artifact | unchanged | per-artifact digest | `ARTIFACT_MODIFIED` |
| a covered artifact deleted | unchanged | per-artifact digest | `ARTIFACT_MISSING` |
| artifact **and** its manifest digest re-signed | **changes** | hash vs chain | `HASH_MISMATCH` |

The last row is the interesting one: restoring local self-consistency does not
help, because the manifest hash moved and the chain still holds the original.

An artifact edit leaving the manifest hash unchanged is correct, not a gap —
the manifest carries the digests, so the digest layer is what must catch it.

### Failure states

`BUNDLE_NOT_FOUND` · `MANIFEST_MISSING` · `MANIFEST_MALFORMED` ·
`INVESTIGATION_ID_MISSING` · `ARTIFACT_MISSING` · `ARTIFACT_MODIFIED` ·
`CONTRACT_NOT_CONFIGURED` · `CONTRACT_INVALID` · `WRONG_CHAIN` ·
`RPC_FAILURE` · `NOT_ANCHORED` · `HASH_MISMATCH`

### Tamper demonstration

```bash
python scripts/tamper_chain_demo.py evidence/TRACE-20260902-F53AF4
```

Copies the bundle into a `TemporaryDirectory`, applies six mutations, compares
each recomputed fingerprint against the live on-chain value, then re-verifies
the untouched original. The real bundle is never modified.

## Commands

### Everything, in one command

```bash
python main.py --image inputs/demo-target.jpg --mode live
```

```text
[01] FACE SCAN            detect + ArcFace embedding (model loads once)
[02] WEB DISCOVERY        real SerpApi -> Google Lens
[03] CANDIDATE RETRIEVAL  bounded concurrency, isolated failures
[04] FACE MATCHING        independent re-matching against every candidate
[05] EVIDENCE             bundle + deterministic SHA-256
[06] BLOCKCHAIN           anchor on Polygon Amoy
[07] VERIFICATION         read back, compare, VERIFIED
```

Flags:

| flag | effect |
|---|---|
| `--mode live` / `--mode diagnostic` | live never reads a cached response |
| `--verbose` | per-candidate detail instead of aggregate counts |
| `--debug` | let unexpected exceptions surface with a traceback |
| `--no-chain` | stop after the evidence fingerprint (no gas) |
| `--no-retrieval` | stop after discovery |
| `--max-candidates N` | how many discovered candidates to retrieve |

### Desktop GUI

```bash
python -m gui.app
```

A thin PySide6 shell over the same pipeline. It calls `main.run()` on a worker
thread and renders the events the reporter seam delivers — it contains no
pipeline logic, parses no stdout, and shows no value the pipeline did not
produce. The CLI remains the primary interface.

### Before the demo

```bash
python preflight.py            # is the environment ready for a live run?
python preflight.py --offline  # skip the network checks

python healthcheck.py          # full audit: 78 checks, read-only, costs nothing
python healthcheck.py --json final/healthcheck.json
python healthcheck.py --section evidence --section blockchain
```

`preflight.py` answers "can I run now?". `healthcheck.py` answers "is every
claim this project makes still true?" — it re-verifies evidence integrity,
tamper detection, chain reads, secret hygiene and hash determinism without
spending a search or sending a transaction.

### Independent verification

```bash
python verify.py <bundle>                        # local integrity only
python verify_chain.py <bundle>                  # local + on-chain (no key needed)
python verify_chain.py <bundle> --json           # machine-readable
python scripts/tamper_chain_demo.py <bundle>     # tamper demonstration
python scripts/tamper_test.py <bundle>           # local-only tamper test
```

### Deployment (already done; here for completeness)

```bash
python scripts/deploy.py                         # deploy IdentityAnchor
python anchor.py <bundle>                        # anchor an existing bundle
```

### Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | unreadable image / no face |
| 2 | SerpAPI authentication failed |
| 3 | SerpAPI rate limit or quota |
| 4 | discovery failed |
| 5 | ran fine, nothing matched |
| 6 | evidence or fingerprint problem |
| 7 | RPC, wallet, or transaction failure |
| 8 | hash mismatch |
| 130 | interrupted |

A normal run never prints a Python traceback. Unexpected exceptions are caught
at the CLI boundary and summarised; `--debug` re-raises them.

### Duplicate anchoring

The contract rejects duplicates by design, so the CLI checks `isAnchored()`
before spending gas. A normal run generates a fresh investigation id and
anchors once. Re-running `anchor.py` on an already-anchored bundle prints
`already anchored; skipping the write`, sends **no** transaction, and still
verifies against the chain. No transaction is ever sent for visual effect.

## Recording the demo

Four commands, in order. Every value shown comes from the run in front of you.

```bat
.venv\Scripts\activate

python preflight.py
python main.py --image inputs/demo-target.jpg --mode live
python verify_chain.py evidence\TRACE-<id-from-the-run>
python scripts\tamper_chain_demo.py evidence\TRACE-<id-from-the-run>
```

Copy the investigation ID out of the RESULT block of the second command.
A live run takes roughly 30 seconds. `verify_chain.py` needs no private key,
so it demonstrates that anyone holding the bundle can check it independently.

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
