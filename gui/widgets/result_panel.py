"""Result panel, populated from the real ``RunResult``.

Every value is read off the object ``main.run()`` emitted. When a field is
absent - no transaction was broadcast, the chain step was skipped - the panel
says so explicitly rather than leaving a blank that could read as a value.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.pipeline import RunResult

#: Shown wherever the pipeline produced no value.
UNAVAILABLE = "— not performed"
NONE_YET = "—"

FIELDS = (
    "investigation",
    "match status",
    "source",
    "domain",
    "similarity",
    "evidence SHA-256",
    "bundle",
    "network",
    "contract",
    "transaction",
    "block",
    "on-chain hash",
    "verification",
    "elapsed",
)


class ResultPanel(QFrame):
    """Read-only view of one completed run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResultPanel")

        self._headline = QLabel("No investigation yet")
        self._headline.setObjectName("ResultHeadline")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._values: dict[str, QLabel] = {}
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for name in FIELDS:
            value = QLabel(NONE_YET)
            value.setObjectName("ResultValue")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setWordWrap(True)
            key = QLabel(name)
            key.setObjectName("ResultKey")
            self._values[name] = value
            form.addRow(key, value)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addWidget(self._headline)
        layout.addLayout(form)
        layout.addStretch(1)

    # -- inspection (used by tests) -------------------------------------

    def value_of(self, field: str) -> str:
        return self._values[field].text()

    def headline(self) -> str:
        return self._headline.text()

    # -- state ----------------------------------------------------------

    def reset(self, message: str = "Investigation running…") -> None:
        self._headline.setText(message)
        self._set_headline_state("running")
        for label in self._values.values():
            label.setText(NONE_YET)

    def show_failure(self, headline: str, detail: str = "") -> None:
        self._headline.setText(headline)
        self._set_headline_state("failed")
        if detail:
            self._values["investigation"].setText(detail)

    def show_result(self, result: RunResult) -> None:
        """Populate from a real RunResult. Absent values are labelled absent."""
        self._values["investigation"].setText(result.investigation_id)
        self._values["evidence SHA-256"].setText(result.evidence_sha256)
        self._values["bundle"].setText(result.bundle_path)
        self._values["elapsed"].setText(f"{result.elapsed_seconds:.1f}s")

        match = result.match
        if match is not None:
            self._values["match status"].setText(match.status.value)
            self._values["source"].setText(match.candidate.url)
            self._values["domain"].setText(match.candidate.source_domain)
            similarity = match.best_similarity
            self._values["similarity"].setText(
                f"{similarity:.6f}" if similarity is not None else UNAVAILABLE
            )

        receipt = result.receipt
        if receipt is not None:
            self._values["transaction"].setText(receipt.tx_hash)
            self._values["block"].setText(str(receipt.block_number))
        else:
            # Either --no-chain, or the investigation was already anchored.
            self._values["transaction"].setText(UNAVAILABLE)
            self._values["block"].setText(UNAVAILABLE)

        check = result.verification
        if check is not None:
            self._values["network"].setText(
                f"chain id {check.chain_id}" if check.chain_id else UNAVAILABLE)
            self._values["contract"].setText(check.contract_address or UNAVAILABLE)
            self._values["on-chain hash"].setText(check.on_chain_sha256 or UNAVAILABLE)
            self._values["verification"].setText(check.status.value)
        else:
            for name in ("network", "contract", "on-chain hash", "verification"):
                self._values[name].setText(UNAVAILABLE)

        if result.verified:
            self._headline.setText("✓ BLOCKCHAIN VERIFIED")
            self._set_headline_state("verified")
        elif check is not None:
            self._headline.setText(f"✗ {check.status.value}")
            self._set_headline_state("failed")
        else:
            self._headline.setText("Evidence fingerprint ready")
            self._set_headline_state("partial")

    def _set_headline_state(self, state: str) -> None:
        self._headline.setProperty("state", state)
        self._headline.style().unpolish(self._headline)
        self._headline.style().polish(self._headline)
