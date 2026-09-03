"""Visual language for the desktop application.

Dark, dense, forensic. Colour carries meaning and nothing else: green is a
verified fact, amber is in-flight, red is a failure, violet is the one action
the operator can take. Nothing pulses or animates, because nothing here should
imply activity the pipeline has not reported.
"""
from __future__ import annotations

# -- surfaces
BG = "#0b0e14"
SURFACE = "#11151f"
SURFACE_2 = "#161b28"
SURFACE_3 = "#1c2230"
BORDER = "#232a3a"
BORDER_SOFT = "#1a2030"

# -- text
TEXT = "#e6eaf2"
TEXT_DIM = "#98a2b8"
TEXT_FAINT = "#5f6a80"

# -- meaning
ACCENT = "#7c5cff"          # the primary action
ACCENT_DIM = "#5b41c8"
GOOD = "#3ddc84"
GOOD_DIM = "#1c6b41"
WARN = "#ffb340"
BAD = "#ff5f56"
INFO = "#4aa8ff"

MONO = "'Cascadia Mono','JetBrains Mono',Consolas,'DejaVu Sans Mono',monospace"
SANS = "'Segoe UI Variable','Segoe UI',Inter,'DejaVu Sans',sans-serif"

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    font-size: 13px;
}}
/* Labels must not paint the base colour over a lighter panel, which would
   draw a dark band behind every caption. */
QLabel {{ background: transparent; }}
QCheckBox {{ background: transparent; }}

QToolTip {{
    background: {SURFACE_3}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 6px 8px; border-radius: 4px;
}}

/* ---------- shell ---------- */
QFrame#HeaderBar {{
    background: {SURFACE};
    border: none;
    border-bottom: 1px solid {BORDER};
}}
QLabel#Wordmark {{ font-size: 20px; font-weight: 800; letter-spacing: 1.5px; }}
QLabel#AppTitle {{ font-size: 15px; font-weight: 600; letter-spacing: 2.2px; }}
QLabel#AppFlow  {{ color: {TEXT_FAINT}; font-size: 11px; letter-spacing: 0.3px; }}

QLabel#ChainLabel {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#RunState   {{ font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }}
QLabel#RunState[state="idle"]    {{ color: {TEXT_FAINT}; }}
QLabel#RunState[state="running"] {{ color: {WARN}; }}
QLabel#RunState[state="done"]    {{ color: {GOOD}; }}
QLabel#RunState[state="failed"]  {{ color: {BAD}; }}

QFrame#IdCard {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 7px;
}}
QLabel#IdCaption {{ color: {TEXT_FAINT}; font-size: 9px; letter-spacing: 1.4px; font-weight: 700; }}
QLabel#IdValue   {{ font-family: {MONO}; font-size: 14px; font-weight: 700; letter-spacing: 0.5px; }}

/* ---------- columns ---------- */
QFrame#Column {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#ColumnTitle {{
    color: {TEXT_DIM}; font-size: 10px; font-weight: 800; letter-spacing: 1.6px;
}}
QLabel#ColumnIndex {{
    color: {ACCENT}; font-size: 10px; font-weight: 800; letter-spacing: 1.6px;
}}
QFrame#ColumnHead {{ background: {SURFACE_2}; border: none;
                     border-bottom: 1px solid {BORDER_SOFT}; }}
QFrame#Divider {{ background: {BORDER_SOFT}; border: none; max-height: 1px; }}

/* ---------- input ---------- */
QFrame#DropZone {{
    background: {SURFACE_2}; border: 1px dashed {BORDER}; border-radius: 9px;
}}
QFrame#DropZone[hover="true"]      {{ border: 1px dashed {ACCENT}; background: {SURFACE_3}; }}
QFrame#DropZone[state="ready"]     {{ border: 1px solid {BORDER}; background: {SURFACE_2}; }}
QFrame#DropZone[state="rejected"]  {{ border: 1px dashed {BAD}; }}
QFrame#DropZone[locked="true"]     {{ border: 1px solid {BORDER_SOFT}; }}
QLabel#Preview      {{ background: transparent; }}
QLabel#DropCaption  {{ font-size: 13px; font-weight: 600; }}
QLabel#DropHint     {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#DropMeta     {{ color: {TEXT_DIM}; font-family: {MONO}; font-size: 10px; }}
QLabel#DropError    {{ color: {BAD}; font-size: 11px; }}

QFrame#FaceCard {{
    background: {SURFACE_3}; border: 1px solid {BORDER}; border-radius: 8px;
}}
QLabel#FaceCaption {{ color: {TEXT_FAINT}; font-size: 9px; font-weight: 800; letter-spacing: 1.3px; }}
QLabel#FaceHeadline {{ color: {GOOD}; font-size: 12px; font-weight: 700; }}
QLabel#FaceDetail  {{ color: {TEXT_DIM}; font-family: {MONO}; font-size: 10px; }}

/* ---------- buttons ---------- */
QPushButton {{
    background: {SURFACE_3}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 9px 14px; font-weight: 600; font-size: 12px;
}}
QPushButton:hover:enabled  {{ border-color: {ACCENT}; background: {SURFACE_2}; }}
QPushButton:disabled       {{ color: {TEXT_FAINT}; background: {SURFACE}; border-color: {BORDER_SOFT}; }}
QPushButton#Primary {{
    background: {ACCENT}; border: 1px solid {ACCENT}; color: #0a0713;
    padding: 13px 16px; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;
}}
QPushButton#Primary:hover:enabled {{ background: #8d70ff; border-color: #8d70ff; }}
QPushButton#Primary:disabled {{ background: {SURFACE_2}; color: {TEXT_FAINT}; border-color: {BORDER_SOFT}; }}
QPushButton#Danger:enabled  {{ color: {BAD}; border-color: #4a2b2b; }}
QPushButton#Danger:hover:enabled {{ border-color: {BAD}; }}
QPushButton#Ghost {{
    background: transparent; border: 1px solid {BORDER_SOFT};
    color: {TEXT_DIM}; padding: 6px 10px; font-size: 11px;
}}
QPushButton#Ghost:hover:enabled {{ color: {TEXT}; border-color: {BORDER}; }}
QPushButton#IconButton {{
    background: transparent; border: 1px solid {BORDER_SOFT};
    border-radius: 5px; padding: 3px 7px; font-size: 11px; color: {TEXT_DIM};
}}
QPushButton#IconButton:hover:enabled {{ color: {TEXT}; border-color: {ACCENT}; }}

QCheckBox {{ color: {TEXT_DIM}; spacing: 8px; font-size: 12px; }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid {BORDER}; background: {SURFACE_2};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* ---------- stages ---------- */
QLabel#StageIcon {{
    font-size: 13px; font-weight: 700;
    border-radius: 15px; border: 1px solid {BORDER}; background: {SURFACE_2};
    color: {TEXT_FAINT};
}}
QLabel#StageIcon[state="RUNNING"] {{ color: {WARN}; border-color: {WARN}; background: #2a2113; }}
QLabel#StageIcon[state="DONE"]    {{ color: {GOOD}; border-color: {GOOD_DIM}; background: #10241a; }}
QLabel#StageIcon[state="FAILED"]  {{ color: {BAD};  border-color: #5a2a28; background: #2a1414; }}

QLabel#StageNumber {{ color: {TEXT_FAINT}; font-family: {MONO}; font-size: 12px; font-weight: 700; }}
QLabel#StageNumber[state="RUNNING"] {{ color: {WARN}; }}
QLabel#StageNumber[state="DONE"]    {{ color: {TEXT_DIM}; }}
QLabel#StageTitle  {{ color: {TEXT_DIM}; font-size: 12px; font-weight: 700; letter-spacing: 0.6px; }}
QLabel#StageTitle[state="RUNNING"] {{ color: {TEXT}; }}
QLabel#StageTitle[state="DONE"]    {{ color: {TEXT}; }}
QLabel#StageTitle[state="FAILED"]  {{ color: {TEXT}; }}
QLabel#StageNote   {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#StageNote[state="RUNNING"] {{ color: {TEXT_DIM}; }}

QLabel#StageBadge {{ font-size: 9px; font-weight: 800; letter-spacing: 1.1px; color: {TEXT_FAINT}; }}
QLabel#StageBadge[state="RUNNING"] {{ color: {WARN}; }}
QLabel#StageBadge[state="DONE"]    {{ color: {GOOD}; }}
QLabel#StageBadge[state="FAILED"]  {{ color: {BAD}; }}
QLabel#StageTiming {{ color: {TEXT_FAINT}; font-family: {MONO}; font-size: 10px; }}
QFrame#StageConnector {{ background: {BORDER}; border: none; max-width: 1px; }}
QFrame#StageConnector[state="DONE"] {{ background: {GOOD_DIM}; }}

/* ---------- events ---------- */
QScrollArea#EventScroll, QWidget#EventCanvas {{ background: #080b11; border: none; }}
QFrame#EventArea {{ background: #080b11; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#EventTime {{ color: {TEXT_FAINT}; font-family: {MONO}; font-size: 10px; }}
QLabel#EventText {{ color: #c3cbdb; font-family: {MONO}; font-size: 11px; }}
QLabel#EventText[severity="warn"]    {{ color: {WARN}; }}
QLabel#EventText[severity="fail"]    {{ color: {BAD}; }}
QLabel#EventText[severity="stage"]   {{ color: {TEXT}; font-weight: 700; }}
QLabel#EventText[severity="verdict"] {{ color: {GOOD}; font-weight: 700; }}
QLabel#EventDot {{ font-size: 9px; color: {TEXT_FAINT}; }}
QLabel#EventDot[severity="ok"]      {{ color: {GOOD}; }}
QLabel#EventDot[severity="warn"]    {{ color: {WARN}; }}
QLabel#EventDot[severity="fail"]    {{ color: {BAD}; }}
QLabel#EventDot[severity="stage"]   {{ color: {ACCENT}; }}
QLabel#EventDot[severity="verdict"] {{ color: {GOOD}; }}
QLabel#EventEmpty {{ color: {TEXT_FAINT}; font-size: 11px; }}

/* ---------- results ---------- */
QFrame#Verdict {{ border-radius: 10px; border: 1px solid {BORDER}; background: {SURFACE_2}; }}
QFrame#Verdict[state="verified"] {{ border-color: {GOOD_DIM}; background: #0e1f17; }}
QFrame#Verdict[state="failed"]   {{ border-color: #4a2523; background: #1f1211; }}
QFrame#Verdict[state="partial"]  {{ border-color: #4a3a1a; background: #1e1a10; }}
QFrame#Verdict[state="running"]  {{ border-color: {BORDER}; }}
QLabel#VerdictMark {{ font-size: 26px; }}
QLabel#VerdictTitle {{ font-size: 17px; font-weight: 800; letter-spacing: 0.8px; color: {TEXT_DIM}; }}
QLabel#VerdictTitle[state="verified"] {{ color: {GOOD}; }}
QLabel#VerdictTitle[state="failed"]   {{ color: {BAD}; }}
QLabel#VerdictTitle[state="partial"]  {{ color: {WARN}; }}
QLabel#VerdictSub {{ color: {TEXT_DIM}; font-size: 11px; }}

QFrame#Panel {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 9px; }}
QLabel#PanelTitle {{ color: {TEXT_FAINT}; font-size: 9px; font-weight: 800; letter-spacing: 1.4px; }}

QLabel#MetricCaption {{ color: {TEXT_FAINT}; font-size: 9px; font-weight: 700; letter-spacing: 1.2px; }}
QLabel#MetricValue   {{ font-family: {MONO}; font-size: 27px; font-weight: 700; color: {TEXT}; }}
QLabel#MetricValue[state="match"]  {{ color: {GOOD}; }}
QLabel#MetricValue[state="reject"] {{ color: {TEXT_DIM}; }}
QLabel#MetricNote    {{ color: {TEXT_DIM}; font-size: 10px; font-family: {MONO}; }}
QFrame#ScaleTrack {{ background: {SURFACE_3}; border-radius: 3px; border: none; }}
QFrame#ScaleFill  {{ background: {GOOD}; border-radius: 3px; border: none; }}
QFrame#ScaleFill[state="reject"] {{ background: {TEXT_FAINT}; }}
QFrame#ScaleThreshold {{ background: {WARN}; border: none; max-width: 2px; }}

QLabel#SourceThumb  {{ background: {SURFACE_3}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLabel#SourceDomain {{ font-size: 14px; font-weight: 700; }}
QLabel#SourceTitle  {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#SourceUrl    {{ color: {TEXT_FAINT}; font-family: {MONO}; font-size: 10px; }}
QLabel#SourceMeta   {{ color: {TEXT_FAINT}; font-family: {MONO}; font-size: 10px; }}
QLabel#SourceFlag   {{ color: {WARN}; font-size: 10px; font-weight: 700; }}

QLabel#DetailKey   {{ color: {TEXT_FAINT}; font-size: 10px; }}
QLabel#DetailValue {{ color: {TEXT}; font-family: {MONO}; font-size: 10px; }}
QLabel#DetailValue[state="good"] {{ color: {GOOD}; }}
QLabel#DetailValue[state="absent"] {{ color: {TEXT_FAINT}; font-style: italic; }}

QPushButton#Disclosure {{
    background: transparent; border: none; color: {TEXT_FAINT};
    font-size: 9px; font-weight: 800; letter-spacing: 1.4px;
    padding: 4px 0; text-align: left;
}}
QPushButton#Disclosure:hover {{ color: {TEXT_DIM}; }}

/* ---------- footer ---------- */
QFrame#FooterBar {{ background: {SURFACE}; border: none; border-top: 1px solid {BORDER}; }}
QLabel#FooterKey  {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#FooterVal  {{ color: {TEXT_DIM}; font-size: 11px; font-weight: 600; }}
QLabel#FooterVal[state="running"] {{ color: {WARN}; }}
QLabel#FooterVal[state="done"]    {{ color: {GOOD}; }}
QLabel#FooterVal[state="failed"]  {{ color: {BAD}; }}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #303a4e; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0px; }}
"""
