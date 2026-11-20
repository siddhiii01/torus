# `data/` is not for real documents

Customer PDFs, images, spreadsheets, OCR crops and answer keys **must not**
live in this repository. Put them under `TORUS_DATA_DIR` (outside the repo).

See `docs/SETUP.md` for the layout and how to point tools at that folder.

The former bake-off cases under `data/deb/cases/` were moved out of the
working tree as part of Phase 0 (P0.8). They are not re-created here.
Purging them from git history requires explicit owner approval.
