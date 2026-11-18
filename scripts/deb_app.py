"""Local Streamlit UI: upload a DEB pack → OCR → Excel-header table.

Localhost only. Uploads and OCR cache land under reports/deb/ (gitignored; PII).

usage:
  streamlit run scripts/deb_app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deb_extract import extract_rows, render_pdf_pages  # noqa: E402
from ocr.engine import DPI, IMAGE_EXTS, read_document  # noqa: E402

# Not OCR field values — structural/heuristic defaults from extract_rows.
# Hidden from the Streamlit grid + downloads; also ignored for overlay green matches.
_NON_OCR_KEYS = {
    "address_type",  # hardcoded H1 / W by section
    "address_source",  # always "Application form"
    "customer_type",
    "constitution",
    "application_form",  # "Yes" if /application/i in page text
    "page_count",  # OCR page list length, not a form field
    "source_of_voterid_kyc_id",  # literal "Voterid" when id found
    "source_of_pan_kyc_id",  # literal "PAN" when id found
}
_SKIP_EXTRACT_KEYS = _NON_OCR_KEYS

SCHEMA = ROOT / "schemas" / "deb" / "excel_columns.json"
UPLOAD_DIR = ROOT / "reports" / "deb" / "uploads"
OCR_CACHE = ROOT / "reports" / "deb" / "ocr_cache"
PRED_DIR = ROOT / "reports" / "deb" / "preds"
SCORE_DIR = ROOT / "reports" / "deb" / "scores"
BAKEOFF_PATH = SCORE_DIR / "bakeoff_digit_crop.json"
CASES_DIR = ROOT / "data" / "deb" / "cases"

ALLOWED_EXTS = IMAGE_EXTS | {".pdf"}


@st.cache_data(show_spinner=False)
def load_excel_columns() -> list[dict]:
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return sorted(data["columns"], key=lambda c: c["excel_col"])


def case_id_from_name(name: str) -> str:
    stem = Path(name).stem
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    return (slug or "upload").casefold()[:80]


def file_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def save_upload(name: str, data: bytes) -> tuple[Path, str]:
    """Persist upload; return path and a content-addressed case id (stem_digest)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    case_id = f"{case_id_from_name(name)}_{file_digest(data)}"
    dest = UPLOAD_DIR / f"{case_id}{Path(name).suffix.lower()}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest, case_id


def ocr_status_rows(ocr: dict) -> list[dict]:
    rows = []
    for page in ocr.get("pages", []):
        words = page.get("words") or []
        confs = [w.get("conf") for w in words if isinstance(w, dict) and w.get("conf") is not None]
        avg = round(sum(confs) / len(confs), 3) if confs else None
        rows.append(
            {
                "page": page.get("index"),
                "reader": page.get("reader"),
                "status": page.get("status"),
                "words": len(words),
                "avg_conf": avg,
                "seconds": page.get("seconds"),
                "reason": page.get("reason") or "",
            }
        )
    return rows


def ocr_only_columns(columns: list[dict]) -> list[dict]:
    """Drop schema columns that extract fills with hardcodes, not OCR content."""
    return [c for c in columns if c.get("key") not in _NON_OCR_KEYS]


def rows_to_excel_frame(pred_rows: list[dict], columns: list[dict]) -> pd.DataFrame:
    # Headers are not unique (e.g. two "Bank Lan" cols) — build by position.
    headers = [c["excel_header"] for c in columns]
    keys = [c["key"] for c in columns]
    if not pred_rows:
        return pd.DataFrame(columns=headers)
    matrix = [[row.get(k, "") for k in keys] for row in pred_rows]
    return pd.DataFrame(matrix, columns=headers)


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DEB")
    return buf.getvalue()


def load_page_images(path: Path) -> list[Image.Image]:
    """Re-render PDF/image pages for overlay when OCR came from cache (no pixels)."""
    if path.suffix.lower() in IMAGE_EXTS:
        return [Image.open(path).convert("RGB")]
    images: list[Image.Image] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise PermissionError("password-protected")
        for pdf_page in doc:
            pix = pdf_page.get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


def extracted_tokens(pred: dict) -> set[str]:
    """Casefolded tokens from non-empty DEB extract values (exact match for green)."""
    tokens: set[str] = set()
    for row in pred.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for key, val in row.items():
            if key in _SKIP_EXTRACT_KEYS or val is None or val == "":
                continue
            for tok in re.split(r"\s+", str(val).casefold()):
                tok = tok.strip(".,;:()[]{}\"'")
                if tok:
                    tokens.add(tok)
    return tokens


def overlay_words(img: Image.Image, words: list[dict], tokens: set[str]) -> Image.Image:
    """Marker-style highlight: green if OCR word matches an extracted token, else red.

    Draws on a separate RGBA layer then alpha-composites so PIL does not replace
    pixels with opaque RGB (which made fills look solid and hid the letters).
    """
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = base.size
    for word in words:
        bbox = word.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        box = [x0 * w, y0 * h, x1 * w, y1 * h]
        text = (word.get("text") or "").casefold().strip().strip(".,;:()[]{}\"'")
        parts = [p for p in re.split(r"\s+", text) if p]
        hit = bool(tokens) and (text in tokens or any(p in tokens for p in parts))
        # Low alpha ≈ highlighter wash; outline keeps the box readable.
        fill = (40, 180, 70, 70) if hit else (220, 70, 70, 55)
        outline = (20, 140, 50, 200) if hit else (180, 40, 40, 160)
        draw.rectangle(box, fill=fill, outline=outline, width=2)
    return Image.alpha_composite(base, layer).convert("RGB")


def run_pipeline(
    path: Path,
    case_id: str,
    use_cache: bool,
    *,
    digit_crop: bool = True,
    crop_vlm: bool = False,
) -> tuple[dict, dict, list[Image.Image], float]:
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer dpi300 bake-off cache when present (gold cases).
    dpi300 = OCR_CACHE / f"{case_id}_dpi300_none.json"
    plain = OCR_CACHE / f"{case_id}.json"
    if use_cache and dpi300.exists():
        cache_path = dpi300
    else:
        cache_path = plain

    t0 = time.perf_counter()
    if use_cache and cache_path.exists():
        ocr = json.loads(cache_path.read_text(encoding="utf-8"))
        images = load_page_images(path)
    else:
        doc, np_images = read_document(path)
        ocr = doc.to_json()
        cache_path.write_text(json.dumps(ocr), encoding="utf-8")
        images = [Image.fromarray(im) for im in np_images]

    page_images = None
    if digit_crop or crop_vlm:
        page_indexes = [int(p.get("index") or i) for i, p in enumerate(ocr.get("pages") or [])] or [0]
        try:
            page_images = render_pdf_pages(path, page_indexes, dpi=DPI)
        except Exception:
            # Image uploads: use already-loaded RGB arrays.
            import numpy as np

            page_images = [np.asarray(im.convert("RGB")) for im in images]

    pred = extract_rows(
        ocr,
        case_id=case_id,
        page_images=page_images,
        digit_crop=digit_crop,
        crop_vlm=crop_vlm,
    )
    (PRED_DIR / f"{case_id}.json").write_text(
        json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ocr, pred, images, time.perf_counter() - t0


def load_bakeoff() -> list[dict] | None:
    if not BAKEOFF_PATH.exists():
        return None
    try:
        data = json.loads(BAKEOFF_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _fmt3(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "—"


def bakeoff_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        case_id = r.get("case_id", "?")
        keep = r.get("keep") or {}
        for name, v in (r.get("variants") or {}).items():
            if not isinstance(v, dict) or "error" in v:
                rows.append(
                    {
                        "case": case_id,
                        "variant": name,
                        "rowF1": "—",
                        "fldP": "—",
                        "fldR": "—",
                        "fldF1": "—",
                        "keep": False,
                        "note": (v or {}).get("error", "error"),
                    }
                )
                continue
            micro = v.get("extract_fields_micro") or {}
            rows_m = v.get("extract_rows") or {}
            if name == "digit_crop":
                recommend = (keep.get("digit_crop_vs_layout") or {}).get("recommend")
            elif name == "crop_vlm":
                recommend = (keep.get("crop_vlm_vs_digit_crop") or {}).get("recommend")
            else:
                recommend = None
            rows.append(
                {
                    "case": case_id,
                    "variant": name,
                    "rowF1": _fmt3(rows_m.get("f1")),
                    "fldP": _fmt3(micro.get("precision")),
                    "fldR": _fmt3(micro.get("recall")),
                    "fldF1": _fmt3(micro.get("f1")),
                    "keep": recommend,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def show_bakeoff_panel() -> None:
    st.subheader("Bake-off scores")
    results = load_bakeoff()
    if not results:
        st.caption(
            f"No score file at `{BAKEOFF_PATH.relative_to(ROOT)}`. "
            "Run `python scripts/deb_bakeoff_digit_crop.py --cases binani` first."
        )
        return
    st.caption(f"From `{BAKEOFF_PATH.relative_to(ROOT)}` (metadata only).")
    st.dataframe(bakeoff_table(results), use_container_width=True, hide_index=True)
    for r in results:
        case_id = r.get("case_id", "?")
        with st.expander(f"{case_id} · digit recoverability / escalate"):
            for name in ("digit_crop", "crop_vlm"):
                v = (r.get("variants") or {}).get(name) or {}
                if not v or "error" in v:
                    continue
                st.markdown(f"**{name}**")
                dr = (v.get("digit_recoverability") or {}).get("per_field") or {}
                if dr:
                    st.dataframe(pd.DataFrame(dr).T, use_container_width=True)
                meta = v.get("digit_recovery_meta") or {}
                esc = (v.get("digit_recoverability") or {}).get("escalated_keys") or []
                st.caption(f"crops={v.get('crops_located')} · escalated={esc}")
                if meta:
                    st.json(meta)


def list_gold_cases() -> list[str]:
    if not CASES_DIR.exists():
        return []
    out = []
    for p in sorted(CASES_DIR.iterdir()):
        if p.is_dir() and (p / "source.pdf").exists():
            out.append(p.name)
    return out


def render_pred_block(pred: dict, columns: list[dict], show_filled_only: bool, case_id: str) -> None:
    # UI + downloads: OCR-derived fields only (no H1/W, Application form, etc.).
    ocr_cols = ocr_only_columns(columns)
    df = rows_to_excel_frame(pred.get("rows") or [], ocr_cols)
    if show_filled_only and not df.empty:
        keep_idx = [
            i
            for i, col in enumerate(df.columns)
            if df.iloc[:, i].astype(str).str.strip().ne("").any()
        ]
        df_view = df.iloc[:, keep_idx] if keep_idx else df
    else:
        df_view = df

    st.subheader("Excel-header grid")
    st.caption(
        f"Extractor: `{pred.get('extractor')}` · OCR fields only "
        f"({len(df_view.columns)} shown · {len(_NON_OCR_KEYS)} heuristic cols hidden)"
    )
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    recovery = pred.get("digit_recovery") or {}
    if recovery:
        with st.expander("Digit / crop recovery (readers fired)"):
            st.json(recovery)
    if pred.get("crop_vlm"):
        st.caption(f"crop_vlm status: `{pred.get('crop_vlm')}`")

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    xlsx_bytes = dataframe_to_xlsx_bytes(df)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{case_id}_deb.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download XLSX",
        data=xlsx_bytes,
        file_name=f"{case_id}_deb.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.set_page_config(page_title="DEB extract", layout="wide")
    st.title("DEB pack → Excel columns")
    st.caption(
        "Local only. Upload a PDF or open a gold case, run layout + optional digit_crop, "
        "and view bake-off scores from the last `deb_bakeoff_digit_crop.py` run."
    )

    columns = load_excel_columns()

    with st.sidebar:
        st.header("Options")
        use_cache = st.checkbox("Reuse OCR cache for this file name", value=True)
        show_filled_only = st.checkbox("Hide empty columns in table", value=True)
        digit_crop = st.checkbox("Digit crop recover (PIN/DOB/phones)", value=True)
        crop_vlm = st.checkbox(
            "Crop VLM escalate (TrOCR) — bake-off said no on binani",
            value=False,
        )
        st.markdown(
            f"- Upload dir: `{UPLOAD_DIR.relative_to(ROOT)}`  \n"
            f"- OCR cache: `{OCR_CACHE.relative_to(ROOT)}`  \n"
            f"- Preds: `{PRED_DIR.relative_to(ROOT)}`  \n"
            f"- Scores: `{SCORE_DIR.relative_to(ROOT)}`"
        )
        st.info(
            "Default path: layout + digit_crop (P≈1.0 on binani). "
            "Crop VLM stays off unless you opt in."
        )

    show_bakeoff_panel()
    st.divider()

    source = st.radio(
        "Input",
        ["Upload file", "Gold case"],
        horizontal=True,
    )

    path: Path | None = None
    case_id = ""
    session_key = ""
    run = False

    if source == "Gold case":
        cases = list_gold_cases()
        if not cases:
            st.warning(f"No cases with source.pdf under `{CASES_DIR.relative_to(ROOT)}`.")
            return
        pick = st.selectbox("Case", cases, index=cases.index("binani") if "binani" in cases else 0)
        path = CASES_DIR / pick / "source.pdf"
        case_id = pick
        session_key = f"gold:{pick}"
        st.write(f"**Case:** `{pick}` · `{path.relative_to(ROOT)}`")
        # Prefer dpi300 OCR cache for gold cases when present.
        dpi300 = OCR_CACHE / f"{pick}_dpi300_none.json"
        if use_cache and dpi300.exists():
            st.caption(f"Will reuse `{dpi300.name}` if present.")
        run = st.button("Run extract on gold case", type="primary")
    else:
        uploaded = st.file_uploader(
            "Drop a DEB pack (PDF) or page image",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        )
        run = st.button("Run OCR + extract", type="primary", disabled=uploaded is None)
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in ALLOWED_EXTS:
                st.error(f"Unsupported type: {suffix}")
                return
            st.write(f"**File:** `{uploaded.name}`")
            session_key = uploaded.name
            if run:
                path, case_id = save_upload(uploaded.name, uploaded.getvalue())

    if run:
        if path is None:
            return
        spinner = (
            "Running extract (digit_crop on located boxes)…"
            if source == "Gold case"
            else "Running OCR + extract (first docTR load can take a while)…"
        )
        with st.spinner(spinner):
            try:
                # Gold cases: point cache lookup at case id (dpi300 preferred inside pipeline).
                ocr, pred, images, elapsed = run_pipeline(
                    path,
                    case_id,
                    use_cache=use_cache,
                    digit_crop=digit_crop,
                    crop_vlm=crop_vlm,
                )
            except PermissionError:
                st.error("Password-protected PDF — cannot open.")
                return
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return
        st.session_state["last_result"] = {
            "case_id": case_id,
            "session_key": session_key,
            "path": str(path),
            "ocr": ocr,
            "pred": pred,
            "images": images,
            "elapsed": elapsed,
        }

    result = st.session_state.get("last_result")
    if not result:
        return
    if session_key and result.get("session_key") != session_key:
        st.info("Run extract to load this selection.")
        return

    case_id = result["case_id"]
    st.write(f"**Case id:** `{case_id}`")

    ocr = result["ocr"]
    pred = result["pred"]
    status = ocr_status_rows(ocr)
    readers = sorted({r["reader"] for r in status if r.get("reader")})
    ocr_secs = sum(float(r.get("seconds") or 0) for r in status)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pages", len(status))
    m2.metric("Reader(s)", ", ".join(readers) or "—")
    m3.metric("OCR sec", f"{ocr_secs:.1f}")
    m4.metric("Extract sec", f"{pred.get('seconds', 0):.3f}")
    m5.metric("Rows", len(pred.get("rows") or []))

    st.subheader("Page status")
    st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)

    render_pred_block(pred, columns, show_filled_only, case_id)

    images: list[Image.Image] = list(result.get("images") or [])
    pages = ocr.get("pages") or []
    if not images and result.get("path"):
        try:
            images = load_page_images(Path(result["path"]))
            result["images"] = images
        except Exception:
            images = []

    st.subheader("OCR overlay")
    st.caption(
        "Green = word text matches an extracted value token; light red = other OCR words. "
        "Match is text-based (not provenance). Cache hits re-render the PDF/image for pixels."
    )
    if not images or not pages:
        st.info("No page images available for overlay.")
    else:
        n = min(len(images), len(pages))
        labels = []
        for i in range(n):
            page_i = pages[i]
            raw_idx = page_i.get("index", i) if isinstance(page_i, dict) else i
            labels.append(f"Page {int(raw_idx) + 1}")
        choice = st.selectbox("Page", options=list(range(n)), format_func=lambda i: labels[i])
        page = pages[choice]
        words = (page.get("words") or []) if isinstance(page, dict) else []
        tokens = extracted_tokens(pred)
        overlaid = overlay_words(images[choice], words, tokens)
        st.caption(f"{len(tokens)} extracted token(s) · {len(words)} OCR word(s) on this page")
        st.image(overlaid, use_container_width=True)


if __name__ == "__main__":
    main()

