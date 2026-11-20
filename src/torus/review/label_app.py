"""Optional Streamlit UI for local labelling (P1.5).

Requires the ``review`` extra::

    pip install -e ".[review]"
    streamlit run src/torus/review/label_app.py

Writes only under ``TORUS_DATA_DIR``. Never uploads documents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from torus.config import default_config_dir
from torus.core.errors import ConfigError, EvalError
from torus.eval.label import (
    LabelCaseRequest,
    label_fields,
    resolve_template,
    write_labelled_case,
)
from torus.eval.schema import (
    ExpectedCell,
    HandwritingLevel,
    ScanType,
    Split,
)


def _run() -> None:
    import streamlit as st

    st.set_page_config(page_title="Torus label", layout="wide")
    st.title("Torus labelling (local)")
    st.caption(
        "Fill template columns by hand. The PDF stays on this machine under "
        "TORUS_DATA_DIR. Nothing is uploaded."
    )

    config_dir = st.text_input(
        "Config dir",
        value=str(default_config_dir()),
        help="Folder containing templates/.",
    )
    template_id = st.text_input("Template id", value="starter_v1")
    set_id = st.text_input("Set id", value="deb_dev")
    case_id = st.text_input("Case id", value="")
    split = st.selectbox("Split", options=["dev", "hidden"], index=0)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        year_raw = st.text_input("Year (optional)", value="")
    with col_b:
        scan_type = st.selectbox(
            "Scan type",
            options=["", "digital", "scan", "photo", "fax", "mixed"],
            index=0,
        )
    with col_c:
        handwriting = st.selectbox(
            "Handwriting",
            options=["", "none", "light", "medium", "heavy"],
            index=0,
        )

    notes = st.text_input("Operator notes (not field values)", value="")
    force = st.checkbox("Overwrite existing case", value=False)
    uploaded = st.file_uploader("Source PDF", type=["pdf"])

    template = None
    try:
        if template_id.strip():
            template = resolve_template(
                template_id.strip(), config_dir=Path(config_dir)
            )
    except (ConfigError, EvalError) as exc:
        st.error(f"Template: {exc}")
        return

    if template is None:
        st.info("Enter a template id to load columns.")
        return

    fields = label_fields(template)
    st.subheader(f"Columns ({template.id})")
    row_count = st.number_input("Number of rows", min_value=1, max_value=50, value=1)
    rows: list[dict[str, ExpectedCell]] = []
    for row_i in range(int(row_count)):
        st.markdown(f"**Row {row_i + 1}**")
        row: dict[str, ExpectedCell] = {}
        for spec in fields:
            label = f"{spec.key} — {spec.excel_header}"
            if spec.critical:
                label += " (critical)"
            raw = st.text_input(
                label,
                key=f"r{row_i}_{spec.key}",
                help="Leave blank to omit; type - for explicit null.",
            )
            text = raw.strip()
            if not text:
                continue
            if text == "-":
                row[spec.key] = None
            else:
                row[spec.key] = text
        rows.append(row)

    if st.button("Save case", type="primary"):
        if not case_id.strip():
            st.error("Case id is required.")
            return
        if uploaded is None:
            st.error("Choose a local PDF first.")
            return
        try:
            year = int(year_raw) if year_raw.strip() else None
        except ValueError:
            st.error("Year must be an integer.")
            return

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "source.pdf"
            pdf_path.write_bytes(uploaded.getvalue())
            try:
                result = write_labelled_case(
                    LabelCaseRequest(
                        pdf_path=pdf_path,
                        set_id=set_id.strip(),
                        case_id=case_id.strip(),
                        template=template,
                        split=Split(split),
                        rows=rows,
                        year=year,
                        scan_type=ScanType(scan_type) if scan_type else None,
                        handwriting_level=(
                            HandwritingLevel(handwriting) if handwriting else None
                        ),
                        notes=notes,
                        force=force,
                    )
                )
            except (ConfigError, EvalError) as exc:
                st.error(str(exc))
                return

        st.success(
            f"Wrote {result.set_id}/{result.case_id} "
            f"(template={result.template_id}) under TORUS_DATA_DIR."
        )
        st.code(str(result.path))


if __name__ == "__main__":
    _run()
