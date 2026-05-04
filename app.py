"""
AM Motors — Sticker Detection Dashboard  (production-grade)
Run:  streamlit run app.py
"""

import base64
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import StickerDetector, DetectionResult, OCR_AVAILABLE
from logger import ExcelLogger, SH_ALL, SH_INSTALL, SH_UNKNOWN
from tracker import VehicleTracker

# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
LOG_FILE  = BASE / "logs" / "detections.xlsx"
VIDEO_FILE = str(BASE / "testing video.mp4")

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AM Motors Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Global ── */
  html, body, [data-testid="stAppViewContainer"] { background:#0a0f1e; }
  [data-testid="stSidebar"]  { background:#0d1526; border-right:1px solid #1e2d47; }
  h1,h2,h3,p,label,span      { color:#e2e8f0 !important; }

  /* ── KPI card ── */
  .kpi { background:linear-gradient(135deg,#111827,#1e293b);
         border:1px solid #334155; border-radius:14px;
         padding:18px 14px; text-align:center; margin-bottom:6px; }
  .kpi .lbl { color:#64748b; font-size:11px; font-weight:700;
               text-transform:uppercase; letter-spacing:.07em; }
  .kpi .val { font-size:30px; font-weight:800; margin-top:6px; }
  .kpi .sub { color:#64748b; font-size:11px; margin-top:4px; }
  .green  { color:#4ade80; }
  .yellow { color:#fbbf24; }
  .red    { color:#f87171; }
  .blue   { color:#60a5fa; }
  .purple { color:#c084fc; }

  /* ── Badge ── */
  .badge { display:inline-block; padding:2px 9px; border-radius:999px;
           font-size:11px; font-weight:700; }
  .bg  { background:#14532d; color:#86efac; }
  .br  { background:#7f1d1d; color:#fca5a5; }
  .by  { background:#713f12; color:#fde68a; }
  .bgy { background:#1e3a5f; color:#93c5fd; }
  .bpu { background:#3b0764; color:#d8b4fe; }

  /* ── Status panel ── */
  .spanel { background:#0f172a; border:1px solid #1e293b;
            border-radius:10px; padding:14px 16px; }
  .srow   { display:flex; align-items:center; gap:10px;
            margin-bottom:9px; }
  .slbl   { color:#64748b; font-size:13px; width:165px; flex-shrink:0; }

  /* ── Period selector ── */
  div[data-testid="stHorizontalBlock"] button {
    border-radius:8px !important; font-size:12px !important; }

  /* ── Tab styling ── */
  button[data-baseweb="tab"] { font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "running":        False,
    "stop_event":     threading.Event(),
    "frame_q":        queue.Queue(maxsize=2),
    "result_q":       queue.Queue(maxsize=2),
    "error_q":        queue.Queue(maxsize=5),
    "last_result":    None,
    "tracker_state":  "IDLE",
    "total_logged":   0,
    "fps_display":    0.0,
    "period":         "today",
    "thread_error":   None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

logger = ExcelLogger(str(LOG_FILE))   # shared read-only logger for UI


def _do_reset():
    """Delete the Excel file, recreate it with empty sheets, reset all counters."""
    try:
        LOG_FILE.unlink(missing_ok=True)
    except PermissionError:
        st.error(
            "Cannot reset — the Excel file is open in another program (e.g. Excel). "
            "Close it and try again."
        )
        return
    except Exception as e:
        st.error(f"Reset failed: {e}")
        return

    ExcelLogger(str(LOG_FILE))          # recreates the file with headers + 3 sheets
    st.session_state.total_logged   = 0
    st.session_state.last_result    = None
    st.session_state.tracker_state  = "IDLE"
    st.session_state.fps_display    = 0.0
    st.toast("All data reset — starting from line 1.", icon="🗑")
    st.rerun()


def _frame_html(frame: np.ndarray, max_width: int = 960) -> str:
    """Encode a BGR frame as base64 JPEG and return an <img> HTML string.
    Resizes to max_width first so large CCTV frames don't bloat the page.
    Avoids Streamlit's MediaFileHandler entirely — no missing-file errors."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    b64 = base64.b64encode(buf).decode()
    return (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:100%;border-radius:8px;display:block">'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AM Motors")
    st.caption("Sticker Detection & Ad-Reach Tracker")
    st.divider()

    st.markdown("### 📷 Source")
    src_choice = st.radio(
        "src", ["Testing Video", "Webcam 0", "Webcam 1", "Webcam 2", "IP / RTSP"],
        label_visibility="collapsed",
    )
    if src_choice == "IP / RTSP":
        st.caption("Supports RTSP (DVR/NVR/IP cam) and HTTP MJPEG streams")
        _ip_mode = st.radio("Input mode", ["Build URL", "Manual URL"],
                            label_visibility="collapsed", horizontal=True)
        if _ip_mode == "Build URL":
            _proto = st.selectbox("Protocol", ["rtsp://", "http://", "https://"],
                                  label_visibility="collapsed")
            _ip    = st.text_input("Camera IP", placeholder="192.168.1.100",
                                   help="LAN IP of the camera, DVR, or NVR")
            _port  = st.number_input("Port",
                                     value=554 if "rtsp" in _proto else 8080,
                                     min_value=1, max_value=65535)
            _path  = st.text_input("Stream path", value="stream",
                                   placeholder="stream  /  ch01  /  video")
            _user  = st.text_input("Username", placeholder="admin  (leave blank if none)")
            _pwd   = st.text_input("Password", type="password",
                                   placeholder="leave blank if none")
            if _ip.strip():
                _cred = (f"{_user.strip()}:{_pwd.strip()}@"
                         if (_user.strip() and _pwd.strip()) else "")
                source = f"{_proto}{_cred}{_ip.strip()}:{int(_port)}/{_path.strip()}"
                st.caption(f"`{source}`")
            else:
                source = ""
                st.caption("Enter the camera IP address above")
        else:
            _raw = st.text_input(
                "Full stream URL",
                placeholder="rtsp://192.168.1.100:554/stream",
                help="RTSP, HTTP MJPEG, or any OpenCV-compatible stream URL",
            )
            source = _raw.strip()
    else:
        source = {"Testing Video": VIDEO_FILE,
                  "Webcam 0": 0, "Webcam 1": 1, "Webcam 2": 2}[src_choice]

    if src_choice == "IP / RTSP":
        with st.expander("📋 Common camera URL formats"):
            st.markdown("""
**Hikvision DVR/NVR**
`rtsp://admin:pass@IP:554/Streaming/Channels/101`
*(ch1 main stream — use 102 for sub-stream)*

**Dahua DVR/NVR**
`rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0`

**CP Plus / Tiandy**
`rtsp://admin:pass@IP:554/stream1`

**Generic / unknown**
Try these paths one by one:
`/stream` · `/live` · `/live.sdp` · `/h264` · `/video1`

**HTTP MJPEG (cheap IP cams)**
`http://IP:8080/video` or `http://IP/mjpeg`

**Tips:**
- Default port is **554** for RTSP
- Username is usually **admin**
- If no password was set, leave it blank
- Check the camera's sticker or manual for the exact path
""")

    st.divider()
    st.markdown("### ⚙️ Detection")
    threshold   = st.slider("Logo threshold",  0.10, 1.0, 0.65, 0.05)
    use_logo    = st.toggle("Triangle logo",   value=True)
    use_ocr     = st.toggle("OCR (text + plate)", value=True)
    use_motion  = st.toggle("Motion-triggered ROI", value=True)

    if not OCR_AVAILABLE:
        st.warning("Tesseract not found — plate/OCR detection disabled. "
                   "Set env var TESSERACT_CMD or install Tesseract.", icon="⚠️")
    st.divider()
    st.markdown("### 🎯 ROI")
    use_roi = st.checkbox("Fixed ROI", value=False)
    roi: tuple | None = None
    if use_roi:
        c1, c2 = st.columns(2)
        roi = (int(c1.number_input("X", 0, 3840, 0,   step=10)),
               int(c2.number_input("Y", 0, 2160, 0,   step=10)),
               int(c1.number_input("W", 10, 3840, 640, step=10)),
               int(c2.number_input("H", 10, 2160, 480, step=10)))

    st.divider()
    st.markdown("### 🚗 Tracker")
    exit_timeout = st.slider("Exit timeout (s)", 1.0, 15.0, 3.0, 0.5)
    min_frames   = st.number_input("Min frames", 1, 20, 2)
    cooldown     = st.slider("Cooldown (s)", 1.0, 20.0, 4.0, 0.5)
    reach_factor = st.number_input("Reach multiplier", 1, 500, 50,
                                   help="Estimated people who see each stickered vehicle per day")

    st.divider()
    st.markdown("### 🗃️ Logs")
    if LOG_FILE.exists():
        # Read into memory so the file handle is closed before any reset attempt
        _excel_bytes = LOG_FILE.read_bytes()
        st.download_button("⬇ Export Excel", _excel_bytes,
                           file_name="am_motors_log.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.button("🗑 Reset Data", help="Delete all logs and start fresh"):
        _do_reset()


# ─────────────────────────────────────────────────────────────────────────────
# Detection thread
# ─────────────────────────────────────────────────────────────────────────────

def _detection_loop(
    source, threshold, use_logo, use_ocr, use_motion, roi,
    exit_timeout, min_frames, cooldown,
    frame_q, result_q, stop_event, error_q,
):
    def _put(q, item):
        if q.full():
            try: q.get_nowait()
            except queue.Empty: pass
        try: q.put_nowait(item)
        except queue.Full: pass

    def _open(src):
        # Optimization: Use FFMPEG for RTSP and set a short timeout/buffer
        if isinstance(src, str) and "rtsp" in src.lower():
            # Standard OpenCV RTSP flags for low latency
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" 
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(src)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keep buffer tiny for live
        return cap

    cap = None
    try:
        detector = StickerDetector(str(BASE), threshold=threshold)
        _logger  = ExcelLogger(str(LOG_FILE))
        tracker  = VehicleTracker(exit_timeout=exit_timeout,
                                   min_frames=min_frames, cooldown=cooldown)

        cap = _open(source)
        if not cap.isOpened():
            _put(error_q, f"Connection Failed: {source!r}. Check IP/Credentials.")
            return

        last_t = time.monotonic()
        
        while not stop_event.is_set():
            # --- LIVE STREAM STABILITY TRICK ---
            # Flush the buffer: grab frames until the latest one to stay "Real Time"
            if not isinstance(source, str) or not source.lower().endswith(".mp4"):
                for _ in range(5): # Skip up to 5 stale frames
                    cap.grab()
            
            ret, frame = cap.read()
            
            if not ret:
                # Handle Disconnection
                cap.release()
                time.sleep(2) # Cooldown before retry
                cap = _open(source)
                continue

            now = time.monotonic()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            # Video file specific logic (skip ahead)
            if isinstance(source, str) and source.lower().endswith(".mp4"):
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 2) # Slight speed up for testing

            # Core Logic
            m_roi  = detector.motion_roi(frame, roi) if use_motion else None
            result = detector.detect_all(frame, roi=roi, use_logo=use_logo,
                                         use_ocr=use_ocr, motion_hint=m_roi)
            
            logged = tracker.update(result, _logger, frame=frame)
            ann    = detector.annotate_frame(frame, result, roi=roi, motion_roi=m_roi)

            _put(frame_q, ann)
            _put(result_q, (result, tracker.state, logged, fps))

    except Exception as exc:
        import traceback as _tb
        _put(error_q, f"Error: {exc}\n{_tb.format_exc()}")
    finally:
        if cap is not None:
            cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Header + controls
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 AM Motors — Detection & Ad-Reach Dashboard")

# Period selector
PERIODS = {"today": "Today", "yesterday": "Yesterday",
           "week": "This Week", "month": "This Month",
           "year": "This Year", "all": "All Time"}
pc = st.columns(len(PERIODS))
for i, (k, label) in enumerate(PERIODS.items()):
    if pc[i].button(label, use_container_width=True,
                    type="primary" if st.session_state.period == k else "secondary"):
        st.session_state.period = k
        st.rerun()

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
stats = logger.get_period_stats(st.session_state.period, reach_factor=reach_factor)

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _kpi(col, label, value, sub="", color="blue"):
    col.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val {color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

_kpi(k1, "Total Vehicles",    stats["total"],   PERIODS[st.session_state.period], "blue")
_kpi(k2, "Fully Tagged (3/3)", stats["full"],   "All 3 stickers",  "green")
_kpi(k3, "Partial (1–2)",      stats["partial"],"1 or 2 stickers", "yellow")
_kpi(k4, "No Stickers",        stats["none"],   "Need installation","red")
_kpi(k5, "Ad Reach",           f"~{stats['reach']:,}",
     f"people ({reach_factor}x/vehicle)", "purple")
_kpi(k6, "Status",
     "LIVE" if st.session_state.running else "IDLE",
     f"{st.session_state.fps_display:.1f} fps",
     "green" if st.session_state.running else "")

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Main layout — feed left, tabs right
# ─────────────────────────────────────────────────────────────────────────────
feed_col, tabs_col = st.columns([3, 2])

with feed_col:
    st.markdown("### Live Feed")
    frame_ph = st.empty()
    bc1, bc2, bc3 = st.columns(3)
    start_btn  = bc1.button("▶ Start", type="primary",
                            disabled=st.session_state.running, use_container_width=True)
    stop_btn   = bc2.button("⏹ Stop",
                            disabled=not st.session_state.running, use_container_width=True)
    reset_btn  = bc3.button("🗑 Reset Data", use_container_width=True,
                            help="Clear all logs and restart counters from zero")

with tabs_col:
    tab_det, tab_all, tab_inst, tab_unk, tab_chart = st.tabs(
        ["Detection", "All Logs", "Should Install", "Unknown Stickers", "Analytics"]
    )

    with tab_det:
        status_ph = st.empty()

    with tab_all:
        log_ph = st.empty()

    with tab_inst:
        install_ph = st.empty()

    with tab_unk:
        unknown_ph = st.empty()

    with tab_chart:
        chart_ph = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Button logic
# ─────────────────────────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    if not source and source != 0:
        st.error("No source selected — enter a camera URL or pick a webcam.")
    else:
        while not st.session_state.error_q.empty():
            try: st.session_state.error_q.get_nowait()
            except queue.Empty: break
        st.session_state.thread_error = None
        st.session_state.stop_event.clear()
        st.session_state.running = True
        t = threading.Thread(
            target=_detection_loop,
            args=(source, threshold, use_logo, use_ocr, use_motion, roi,
                  exit_timeout, min_frames, cooldown,
                  st.session_state.frame_q, st.session_state.result_q,
                  st.session_state.stop_event, st.session_state.error_q),
            daemon=True,
        )
        t.start()
        st.rerun()

if stop_btn and st.session_state.running:
    st.session_state.stop_event.set()
    st.session_state.running = False
    st.rerun()

if reset_btn:
    if st.session_state.running:
        st.session_state.stop_event.set()
        st.session_state.running = False
    _do_reset()


# ─────────────────────────────────────────────────────────────────────────────
# Helper renderers
# ─────────────────────────────────────────────────────────────────────────────

def _badge(hit: bool, y="YES", n="NO", cls_y="bg", cls_n="br") -> str:
    return (f'<span class="badge {cls_y}">{y}</span>' if hit
            else f'<span class="badge {cls_n}">{n}</span>')


def _render_detection(result: DetectionResult, state: str):
    state_cls = {"TRACKING": "bg", "COOLDOWN": "by", "IDLE": "bgy"}.get(state, "bgy")
    html = f"""
    <div class="spanel">
      <div style="display:flex;justify-content:space-between;margin-bottom:12px">
        <span style="font-size:13px;font-weight:700;color:#94a3b8;text-transform:uppercase">Tracker State</span>
        <span class="badge {state_cls}">{state}</span>
      </div>
      <hr style="border-color:#1e293b;margin:6px 0 10px">

      <div class="srow">
        <span class="slbl">Triangle Logo</span>
        {_badge(result.logo1)}
        <span style="color:#475569;font-size:11px">conf {result.logo1_conf:.2f}</span>
      </div>
      <div class="srow">
        <span class="slbl">A.M.MOTORS text</span>
        {_badge(result.logo2)}
      </div>
      <div class="srow">
        <span class="slbl">www.ammotors.in</span>
        {_badge(result.logo3)}
      </div>
      <div class="srow">
        <span class="slbl">Unknown sticker</span>
        {_badge(result.unknown_sticker, "DETECTED", "none", "bpu", "bgy")}
      </div>

      <hr style="border-color:#1e293b;margin:8px 0">
      <div class="srow">
        <span class="slbl">Vehicle plate</span>
        <span style="color:#f1f5f9;font-family:monospace;font-size:15px;font-weight:700">
          {result.vehicle_number or "—"}
        </span>
      </div>
      <div class="srow">
        <span class="slbl">Stickers found</span>
        <span class="badge {'bg' if result.sticker_count==3 else 'by' if result.sticker_count>0 else 'br'}">{result.sticker_count}/3</span>
      </div>
      <div style="margin-top:10px;color:#334155;font-size:11px">{result.timestamp}</div>
    </div>"""
    status_ph.markdown(html, unsafe_allow_html=True)


def _styled_df(rows: list, highlight_col: str | None = None) -> None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if highlight_col and highlight_col in df.columns:
        def _style(v):
            if v == "YES":   return "background:#14532d;color:#86efac"
            if v == "NO":    return "background:#7f1d1d;color:#fca5a5"
            return ""
        yes_no_cols = [c for c in ["Logo_Triangle","AM_MOTORS","Website_Sticker"] if c in df.columns]
        if yes_no_cols:
            return df.style.map(_style, subset=yes_no_cols)
    return df


def _render_logs():
    rows = logger.get_recent(20, SH_ALL)
    if rows:
        styled = _styled_df(rows)
        log_ph.dataframe(styled if styled is not None else pd.DataFrame(rows),
                         use_container_width=True, hide_index=True, height=360)
    else:
        log_ph.info("No vehicle exits logged yet.")


def _render_install():
    rows = logger.get_recent(20, SH_INSTALL)
    if rows:
        install_ph.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True, height=360)
    else:
        install_ph.success("No vehicles need sticker installation right now.")


def _render_unknown():
    rows = logger.get_recent(20, SH_UNKNOWN)
    if rows:
        # Show without Screenshot column (binary image data)
        df = pd.DataFrame(rows)
        if "Screenshot" in df.columns:
            df = df.drop(columns=["Screenshot"])
        unknown_ph.dataframe(df, use_container_width=True, hide_index=True, height=300)
        unknown_ph.caption("Screenshots are embedded in the Excel file — open it to view images.")
    else:
        unknown_ph.info("No unknown stickers detected yet.")


def _render_chart():
    # Daily trend (last 14 days)
    series = logger.get_daily_series(14)
    if any(v > 0 for v in series.values()):
        df_trend = pd.DataFrame({
            "Date": list(series.keys()),
            "Vehicles": list(series.values()),
        }).set_index("Date")
        chart_ph.markdown("**Daily vehicle count (last 14 days)**")
        chart_ph.bar_chart(df_trend, color="#60a5fa")
    else:
        chart_ph.info("Not enough data yet — charts appear after vehicles are logged.")

    # Sticker distribution for current period
    breakdown = logger.get_sticker_breakdown(st.session_state.period)
    if sum(breakdown.values()) > 0:
        with chart_ph.container():
            st.markdown(f"**Sticker distribution — {PERIODS[st.session_state.period]}**")
            labels = {0: "No stickers", 1: "1 sticker", 2: "2 stickers", 3: "All 3"}
            df_dist = pd.DataFrame({
                "Category": [labels[k] for k in sorted(breakdown)],
                "Count":    [breakdown[k] for k in sorted(breakdown)],
            }).set_index("Category")
            st.bar_chart(df_dist, color="#4ade80")


# ─────────────────────────────────────────────────────────────────────────────
# Live display loop
# ─────────────────────────────────────────────────────────────────────────────

# Pull any error from the detection thread and auto-stop
if st.session_state.running:
    try:
        _err = st.session_state.error_q.get_nowait()
        st.session_state.thread_error = _err
        st.session_state.running = False
    except queue.Empty:
        pass

if st.session_state.thread_error:
    st.error(st.session_state.thread_error.splitlines()[0], icon="🚨")

if st.session_state.running:
    frame = None
    payload = None
    try: frame   = st.session_state.frame_q.get_nowait()
    except queue.Empty: pass
    try: payload = st.session_state.result_q.get_nowait()
    except queue.Empty: pass

    if frame is not None:
        frame_ph.markdown(_frame_html(frame), unsafe_allow_html=True)
    else:
        frame_ph.info("Waiting for first frame…")

    if payload:
        result, t_state, logged, fps = payload
        st.session_state.last_result  = result
        st.session_state.tracker_state = t_state
        st.session_state.fps_display   = round(fps, 1)
        if logged:
            st.session_state.total_logged += 1
        _render_detection(result, t_state)
    elif st.session_state.last_result:
        _render_detection(st.session_state.last_result, st.session_state.tracker_state)
    else:
        status_ph.info("Detection starting…")

    _render_logs()
    _render_install()
    _render_unknown()
    _render_chart()

    time.sleep(0.4)
    st.rerun()

else:
    frame_ph.markdown(
        '<div style="height:340px;background:#0f172a;border-radius:12px;'
        'display:flex;align-items:center;justify-content:center;'
        'border:2px dashed #1e293b">'
        '<div style="text-align:center;color:#334155">'
        '<div style="font-size:48px">📷</div>'
        '<div style="font-size:15px;margin-top:8px">Press <b>▶ Start</b></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.last_result:
        _render_detection(st.session_state.last_result, st.session_state.tracker_state)
    else:
        status_ph.markdown(
            '<div class="spanel" style="text-align:center;color:#334155;padding:28px">'
            'Idle — start detection to see results</div>',
            unsafe_allow_html=True,
        )
    _render_logs()
    _render_install()
    _render_unknown()
    _render_chart()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns(3)
fc1.caption(f"Log: `{LOG_FILE.name}`  —  {logger.total_logged} total vehicles")
fc2.caption(f"Period: **{PERIODS[st.session_state.period]}**  |  Reach ×{reach_factor}")
fc3.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
