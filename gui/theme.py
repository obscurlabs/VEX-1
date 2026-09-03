"""Dark, restrained styling. Phase 7.C handles serious visual polish."""
from __future__ import annotations

BG = "#0f1115"
PANEL = "#161922"
PANEL_ALT = "#1c202b"
BORDER = "#262b38"
TEXT = "#e4e7ee"
MUTED = "#7b8496"
ACCENT = "#4f8cff"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
MONO = "Cascadia Mono, Consolas, DejaVu Sans Mono, monospace"
SANS = "Segoe UI, Inter, DejaVu Sans, sans-serif"

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    font-size: 13px;
}}
QLabel#Title {{ font-size: 17px; font-weight: 600; letter-spacing: 0.4px; }}
QLabel#Subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#SectionTitle {{
    color: {MUTED}; font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; padding: 2px 0 6px 0;
}}

QFrame#Card, QFrame#ResultPanel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

QFrame#DropZone {{
    background: {PANEL};
    border: 1px dashed {BORDER};
    border-radius: 8px;
}}
QFrame#DropZone[hover="true"] {{ border: 1px dashed {ACCENT}; background: {PANEL_ALT}; }}
QFrame#DropZone[state="ready"] {{ border: 1px solid {BORDER}; }}
QFrame#DropZone[state="rejected"] {{ border: 1px dashed {BAD}; }}
QFrame#DropZone[locked="true"] {{ background: {BG}; }}
QLabel#DropCaption {{ color: {TEXT}; font-size: 13px; }}
QLabel#DropMeta {{ color: {MUTED}; font-size: 11px; font-family: {MONO}; }}

QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 9px 18px;
    font-weight: 600;
}}
QPushButton:hover:enabled {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; background: {PANEL}; }}
QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #06090f; }}
QPushButton#Primary:disabled {{ background: {PANEL}; color: {MUTED}; border-color: {BORDER}; }}
QPushButton#Danger:enabled {{ border-color: {BAD}; color: {BAD}; }}

QCheckBox {{ color: {MUTED}; spacing: 7px; }}
QCheckBox:disabled {{ color: #4a5261; }}

QWidget#StageRow {{ border-radius: 5px; }}
QLabel#StageMarker {{ color: {MUTED}; font-family: {MONO}; font-size: 14px; }}
QLabel#StageLabel {{ color: {MUTED}; font-family: {MONO}; font-size: 12px; }}
QLabel#StageStatus {{ color: {MUTED}; font-size: 10px; letter-spacing: 0.8px; }}

QLabel#StageMarker[state="RUNNING"] {{ color: {ACCENT}; }}
QLabel#StageLabel[state="RUNNING"]  {{ color: {TEXT}; font-weight: 600; }}
QLabel#StageStatus[state="RUNNING"] {{ color: {ACCENT}; }}
QLabel#StageMarker[state="DONE"] {{ color: {GOOD}; }}
QLabel#StageLabel[state="DONE"]  {{ color: {TEXT}; }}
QLabel#StageStatus[state="DONE"] {{ color: {GOOD}; }}
QLabel#StageMarker[state="FAILED"] {{ color: {BAD}; }}
QLabel#StageLabel[state="FAILED"]  {{ color: {TEXT}; }}
QLabel#StageStatus[state="FAILED"] {{ color: {BAD}; }}
QLabel#StageStatus[state="SKIPPED"], QLabel#StageStatus[state="NOT REACHED"] {{ color: #565f70; }}

QPlainTextEdit#LogPanel {{
    background: #0b0d12;
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-family: {MONO};
    font-size: 11px;
    color: #b9c0cd;
    padding: 8px;
}}

QLabel#ResultHeadline {{
    font-size: 15px; font-weight: 700; letter-spacing: 0.6px;
    padding: 8px; border-radius: 6px; background: {PANEL_ALT}; color: {MUTED};
}}
QLabel#ResultHeadline[state="verified"] {{ color: {GOOD}; }}
QLabel#ResultHeadline[state="failed"]   {{ color: {BAD}; }}
QLabel#ResultHeadline[state="running"]  {{ color: {ACCENT}; }}
QLabel#ResultHeadline[state="partial"]  {{ color: {WARN}; }}

QLabel#ResultKey {{ color: {MUTED}; font-size: 11px; }}
QLabel#ResultValue {{ color: {TEXT}; font-family: {MONO}; font-size: 11px; }}

QLabel#StatusBar {{ color: {MUTED}; font-size: 11px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
