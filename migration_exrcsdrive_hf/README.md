# SoccerNet EXRCS → Hugging Face migration toolkit

Generic scripts for moving SoccerNet datasets off the KAUST EXRCS ownCloud
(`exrcsdrive.kaust.edu.sa`) onto the `SoccerNet` org on Hugging Face. These
are reused across every dataset we migrate — nothing here is hardcoded to
one task.

Claude will not run these scripts directly; run them yourself (in the
`SoccerNet` conda env: `conda activate SoccerNet`) and report back the
output.

## Scripts

- **`probe_share.py`** — sends one HTTP HEAD request to an EXRCS share
  token/file, read-only, to confirm the share is still alive before
  downloading anything.
- **`download_task.py`** — stages a task from EXRCS to a local folder using
  the existing `SoccerNet.Downloader.SoccerNetDownloader.downloadDataTask`
  (not raw HTTP), so the local layout matches what the shipped library
  already produces.
- **`upload_to_hf.py`** — creates (or reuses) a Hugging Face dataset repo at
  a given visibility and uploads the staged files into it, refusing to run
  if the repo's live visibility doesn't match what you asked for.

## Per-dataset workflow

1. Probe every share token the task uses.
2. Download the confirmed-alive files to a local staging folder.
3. Upload public files to their public repo, held-out ground truth to
   `SoccerNet/SN-GroundTruth` (private).
4. Verify on huggingface.co.
5. (Later, separately) update `SoccerNet/Downloader.py` to fetch that task
   via `snapshot_download` instead of EXRCS.

## Current status / blockers

- HF auth: logged in as `SilvioGiancola`, member of the `SoccerNet` org.
- **Blocker**: the current HF token's fine-grained scopes only show
  `repo.write` for `OpenSportsLab` and the personal `SilvioGiancola`
  namespace — not for the `SoccerNet` org. `upload_to_hf.py` will fail with
  a permissions error against `SoccerNet/...` repos until this token is
  re-scoped (or a new one generated) at
  https://huggingface.co/settings/tokens.
- Decided: for `calibration-2023`, public files (`train`/`valid`/`test`/
  `challenge`) go to `SoccerNet/SN-Calibration-2023`; `test_labels`/
  `challenge_labels` go to `SoccerNet/SN-GroundTruth` (private).
- **Correction**: initially assumed `reid-2023`/`tracking-2023`/`jersey-2023`
  main splits needed a real NDA password (→ gated). Probing found that
  password actually returns 401 on all three tokens, while the generic
  `SoccerNet` password returns 200 — so these were never more protected than
  calibration. Confirmed with Silvio: their main-split repos
  (`SoccerNet/SN-ReID-2023`, `SoccerNet/SN-Tracking-2023`,
  `SoccerNet/SN-Jersey-2023`) are **public**, like `SN-Calibration-2023`.
  Their held-out GT still goes to the private `SoccerNet/SN-GroundTruth`,
  gated by each task's own real password (see env vars below, get the actual
  values from Silvio — never written to a file).

## Worked example: calibration-2023

```bash
conda activate SoccerNet
cd ~/git/SoccerNet

# 1. Probe (already done once manually — re-run any time to re-check liveness)
python migration_exrcsdrive_hf/probe_share.py --token DxdZ39FT9GqCkEe --file train.zip --password SoccerNet
python migration_exrcsdrive_hf/probe_share.py --token D4xoTmkC7SEJGZp --file test_secret.zip --password-env SOCCERNET_NDA_PASSWORD

# 2. Download
python migration_exrcsdrive_hf/download_task.py \
    --task calibration-2023 --split train valid test challenge --password SoccerNet

export SOCCERNET_NDA_PASSWORD='...'   # the real NDA password - set once in your shell, never in a file
python migration_exrcsdrive_hf/download_task.py \
    --task calibration-2023 --split test_labels challenge_labels --password-env SOCCERNET_NDA_PASSWORD

# 3. Upload (only once the HF token scope blocker above is fixed)
python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-Calibration-2023 --visibility public \
    --local-dir ~/soccernet_hf_migration/calibration-2023 \
    --files train.zip valid.zip test.zip challenge.zip

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-GroundTruth --visibility private \
    --local-dir ~/soccernet_hf_migration/calibration-2023 \
    --files test_secret.zip challenge_secret.zip \
    --path-prefix calibration-2023
```

Note: `test_secret.zip`/`challenge_secret.zip` returned HTTP 401 with the
generic `SoccerNet` password when probed — they need the real NDA password
via `SOCCERNET_NDA_PASSWORD` for both the probe and the download step.

## Worked example: reid-2023, tracking-2023, jersey-2023

Same shape as calibration-2023: main splits use the generic `SoccerNet`
password and go to a **public** repo; GT uses each task's own real NDA
password and goes to the private `SoccerNet/SN-GroundTruth`.

```bash
conda activate SoccerNet
cd ~/git/SoccerNet

export SOCCERNET_REID_GT_PASSWORD='...'    
export SOCCERNET_TRACKING_GT_PASSWORD='...'
export SOCCERNET_JERSEY_GT_PASSWORD='...'  

# --- reid-2023 ---
python migration_exrcsdrive_hf/probe_share.py --token UNYXsOv9RJFU3ax --file train.zip --password SoccerNet
python migration_exrcsdrive_hf/probe_share.py --token GXmA6cNHEJS0MAj --file test_bbox_info.json --password-env SOCCERNET_REID_GT_PASSWORD

python migration_exrcsdrive_hf/download_task.py \
    --task reid-2023 --split train valid test challenge --password SoccerNet
python migration_exrcsdrive_hf/download_task.py \
    --task reid-2023 --split test_labels challenge_labels --password-env SOCCERNET_REID_GT_PASSWORD

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-ReID-2023 --visibility public \
    --local-dir ~/soccernet_hf_migration/reid-2023 \
    --files train.zip valid.zip test.zip challenge.zip

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-GroundTruth --visibility private \
    --local-dir ~/soccernet_hf_migration/reid-2023 \
    --files test_bbox_info.json challenge_bbox_info.json \
    --path-prefix reid-2023

# --- tracking-2023 (no valid split; challenge file is named challenge2023.zip) ---
python migration_exrcsdrive_hf/probe_share.py --token OP7fl7h25NqGfcN --file train.zip --password SoccerNet
python migration_exrcsdrive_hf/probe_share.py --token PvYoZG1INJEp7fk --file test2023-private.zip --password-env SOCCERNET_TRACKING_GT_PASSWORD

python migration_exrcsdrive_hf/download_task.py \
    --task tracking-2023 --split train test challenge --password SoccerNet
python migration_exrcsdrive_hf/download_task.py \
    --task tracking-2023 --split test_labels challenge_labels --password-env SOCCERNET_TRACKING_GT_PASSWORD

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-Tracking-2023 --visibility public \
    --local-dir ~/soccernet_hf_migration/tracking-2023 \
    --files train.zip test.zip challenge2023.zip

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-GroundTruth --visibility private \
    --local-dir ~/soccernet_hf_migration/tracking-2023 \
    --files test2023-private.zip challenge2023-private.zip \
    --path-prefix tracking-2023

# --- jersey-2023 (no valid split) ---
python migration_exrcsdrive_hf/probe_share.py --token ejGBsiDr47cTXnf --file train.zip --password SoccerNet
python migration_exrcsdrive_hf/probe_share.py --token KQ8VcRcKTSbYx1S --file test_gt.json --password-env SOCCERNET_JERSEY_GT_PASSWORD

python migration_exrcsdrive_hf/download_task.py \
    --task jersey-2023 --split train test challenge --password SoccerNet
python migration_exrcsdrive_hf/download_task.py \
    --task jersey-2023 --split test_labels challenge_labels --password-env SOCCERNET_JERSEY_GT_PASSWORD

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-Jersey-2023 --visibility public \
    --local-dir ~/soccernet_hf_migration/jersey-2023 \
    --files train.zip test.zip challenge.zip

python migration_exrcsdrive_hf/upload_to_hf.py \
    --repo-id SoccerNet/SN-GroundTruth --visibility private \
    --local-dir ~/soccernet_hf_migration/jersey-2023 \
    --files test_gt.json challenge_gt.json \
    --path-prefix jersey-2023
```

`Downloader.py` has already been updated for all three tasks (same pattern as
calibration-2023: `source="HuggingFace"` default hitting the repos above,
`source="EXRCSDrive"` as a working fallback using the original tokens).

## Next dataset

Once these are verified on Hugging Face, continue down the self-contained-first
ordering — see project memory — for the remaining 2022/2023/2024 archive-only
tasks (`spotting-ball-2023`, `spotting-ball-2024`, `mvfoul-2024`,
`gamestate-2024`, `spotting-OSL`, and the legacy `calibration`/`reid`/`tracking`).
