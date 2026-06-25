# DentalMind — Dataset Manifest

Organized dental imaging datasets for the DentalMind pipeline.
Raw download archives live in `datasets/`; extracted + organized data lives in `data/`.

## Organized layout (`data/`)

| Dataset | Path | Modality | Files | Source | License |
|---|---|---|---|---|---|
| DENTEX (validation) | `data/dentex/validation/` | Panoramic (OPG) | 51 | Zenodo 7812323 | CC-BY |
| DENTEX (training) | `data/dentex/training/` | Panoramic (OPG) | *(extracting — 10.18 GB zip)* | Zenodo 7812323 | CC-BY |
| Zenodo Dental Cavity | `data/caries_zenodo/Dataset/` | Periapical / caries | 2038 | Zenodo 4907880 | CC |
| Mendeley Mandible | `data/mandible_mendeley/` | Panoramic + mandible seg | 348 | Mendeley hxt48yk462 | CC-BY |
| Tufts Dental DB 2022 | `data/tufts/` | Panoramic + expert/student gaze + masks | 9005 | Kaggle (tommyngx mirror) | Research |

### DENTEX (validation) detail
Real panoramic X-rays with quadrant + enumeration + disease annotations:
`data/dentex/validation/quadrant_enumeration_disease/xrays/val_*.png`
The Phase-1 test fixture (`dentalmine/tests/fixtures/test_opg.jpg`) is `val_0.png` converted to JPEG.

### Tufts subfolders
- `data/tufts/radiographs/` — 1000 panoramic JPGs
- `data/tufts/expert/` — expert labels, gaze maps, masks (+ `expert.json`)
- `data/tufts/student/` — student labels, gaze maps, masks
- `data/tufts/segmentation/` — maxillomandibular + teeth masks

## Raw archives (`datasets/`)
- `training_data.zip` — DENTEX training (downloading via Zenodo API URL, resumable)
- `validation_data.zip` — DENTEX validation (already extracted)
- `Dental_Cavity_Dataset.zip` — Zenodo caries (already extracted)
- `DentalPanoramicXrays.zip` — Mendeley mandible (already extracted)

## Pending / not downloaded
- **Roboflow caries** (project-group13/dental-caries-detection-using-dl, v10) — blocked: the
  supplied API key returned 401 (revoked). Needs a valid key from
  roboflow.com → Account → Roboflow Keys.
- **DENTEX training_data.zip** — large (10.18 GB); download in progress on the laptop.
  Recommended alternative: download directly on the A100 server for the training phases
  (`zenodo_get 7812323` or the API content URL) — far faster bandwidth there.

## Config mapping
`dentalmine/config/default.yaml` data paths point at this tree:
- `data.dentex_path: ../data/dentex/`
- `data.bitewing_path: ../data/caries_zenodo/`  *(closest available caries set; replace with true bitewing set when available)*
- `data.cbct_path: ../data/cbct/`  *(no CBCT dataset downloaded yet)*
