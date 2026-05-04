"""
Core sticker and plate detection — classical CV, no ML/DL.

Detection methods:
  Logo 1 (triangle):  HSV color filter  +  ORB/RANSAC  +  multi-scale template  (ensemble)
  Logo 2 (A.M.MOTORS text):  OCR with CLAHE + adaptive threshold + multi-PSM
  Logo 3 (www.ammotors.in):  OCR (same pipeline)
  Plate:               contour → aspect ratio filter → OCR whitelist
"""

import os
import re
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Tesseract (graceful degradation if not installed) ─────────────────────────
try:
    import pytesseract
    _tess_candidates = [c for c in [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ] if c]
    for _cmd in _tess_candidates:
        if Path(_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = _cmd
            break
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    timestamp: str = ""
    vehicle_number: str = ""
    logo1: bool = False          # Triangle graphical logo
    logo2: bool = False          # A.M.MOTORS text sticker
    logo3: bool = False          # www.ammotors.in sticker
    logo1_conf: float = 0.0
    unknown_sticker: bool = False  # Non-AM-Motors sticker detected
    frame_annotated: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def any_detected(self) -> bool:
        return self.logo1 or self.logo2 or self.logo3 or bool(self.vehicle_number)

    @property
    def sticker_count(self) -> int:
        return sum([self.logo1, self.logo2, self.logo3])

    def merge(self, other: "DetectionResult") -> "DetectionResult":
        merged = copy(self)
        merged.logo1 = self.logo1 or other.logo1
        merged.logo2 = self.logo2 or other.logo2
        merged.logo3 = self.logo3 or other.logo3
        merged.logo1_conf = max(self.logo1_conf, other.logo1_conf)
        merged.unknown_sticker = self.unknown_sticker or other.unknown_sticker
        if other.vehicle_number and not self.vehicle_number:
            merged.vehicle_number = other.vehicle_number
        return merged


# ── Detector ──────────────────────────────────────────────────────────────────
class StickerDetector:
    ORB_MIN_INLIERS = 8
    PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # HSV red ranges (hue wraps around 180→0)
    _RED_LO1 = np.array([0,   55, 55])
    _RED_HI1 = np.array([15, 255, 255])
    _RED_LO2 = np.array([155, 55, 55])
    _RED_HI2 = np.array([180, 255, 255])

    def __init__(self, template_dir: str, threshold: float = 0.65):
        self.threshold = threshold
        template_dir = Path(template_dir)

        # Load templates (both gray and colour for ensemble)
        self._tpl_gray: dict = {}
        self._tpl_bgr: dict = {}
        for i in range(1, 4):
            p = template_dir / f"logo {i}.jpeg"
            if p.exists():
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    self._tpl_bgr[f"logo{i}"] = bgr
                    self._tpl_gray[f"logo{i}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ORB detector — more features for complex logo
        self._orb = cv2.ORB_create(
            nfeatures=2000, scaleFactor=1.2, nlevels=10, edgeThreshold=10
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        tpl1 = self._tpl_gray.get("logo1")
        if tpl1 is not None:
            self._kp1, self._des1 = self._orb.detectAndCompute(tpl1, None)
        else:
            self._kp1, self._des1 = None, None

        # CLAHE for contrast normalisation under all lighting
        self._clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))

        # Background subtractor — used to find moving regions (fast cars)
        self._bgsub = cv2.createBackgroundSubtractorMOG2(
            history=60, varThreshold=36, detectShadows=False
        )
        self._morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _roi(self, frame: np.ndarray, roi: Optional[Tuple]) -> np.ndarray:
        if roi is None:
            return frame
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w < 1 or h < 1:
            return frame
        return frame[y:y + h, x:x + w]

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _enhance(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE + mild denoise — stable across day/night/glare."""
        return self._clahe.apply(gray)

    # ── Motion ROI (for fast-moving cars) ────────────────────────────────────

    def motion_roi(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> Optional[Tuple]:
        """
        Returns a tight bounding box around the largest moving region,
        or None if nothing significant is moving.
        Feeding this back as the detection ROI means we only analyse
        the car, not the whole frame — faster and more accurate.
        """
        region = self._roi(frame, roi)
        gray_small = cv2.resize(self._gray(region), None, fx=0.5, fy=0.5)
        blur = cv2.GaussianBlur(gray_small, (21, 21), 0)
        fg = self._bgsub.apply(blur)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._morph_k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._morph_k)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < 800:   # ignore noise
            return None

        # Upscale back and add generous padding
        x, y, w, h = cv2.boundingRect(largest)
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        pad = 50
        fh, fw = region.shape[:2]
        roi_x = max(0, x - pad)
        roi_y = max(0, y - pad)
        roi_w = min(fw - roi_x, w + 2 * pad)
        roi_h = min(fh - roi_y, h + 2 * pad)

        # Offset by outer roi if there is one
        if roi:
            ox, oy = roi[0], roi[1]
            roi_x += ox
            roi_y += oy

        return (roi_x, roi_y, roi_w, roi_h)

    # ── Logo 1: colour + ORB + template ensemble ──────────────────────────────

    def _logo1_color(self, bgr: np.ndarray) -> Tuple[bool, float]:
        """HSV red-triangle detector — robust to brightness / exposure changes."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, self._RED_LO1, self._RED_HI1),
            cv2.inRange(hsv, self._RED_LO2, self._RED_HI2),
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  k)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = bgr.shape[0] * bgr.shape[1]
        best = 0.0

        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(cnt)
            if area < 25:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            ar = w / h
            # Triangular shape (3–5 verts) + reasonable aspect ratio
            if 3 <= len(approx) <= 5 and 0.35 < ar < 2.8:
                conf = min(area / max(frame_area * 0.0005, 1), 1.0)
                best = max(best, conf)

        # Fallback: enough red pixels anywhere (logo at distance)
        red_ratio = cv2.countNonZero(red) / max(frame_area, 1)
        if red_ratio > 0.0015:
            best = max(best, min(red_ratio * 120, 0.85))

        return best >= 0.3, best

    def _logo1_orb(self, gray: np.ndarray) -> Tuple[bool, float]:
        if self._des1 is None:
            return False, 0.0
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des2 is None or len(des2) < 5:
            return False, 0.0

        raw = self._bf.knnMatch(self._des1, des2, k=2)
        good = [m for m, n in raw if len([m, n]) == 2 and m.distance < 0.72 * n.distance]
        if len(good) < 4:
            return False, 0.0

        src = np.float32([self._kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            return False, 0.0
        inliers = int(mask.ravel().sum())
        conf = min(inliers / max(len(self._kp1), 1) * 3, 1.0)
        return inliers >= self.ORB_MIN_INLIERS, conf

    def _logo1_template(self, gray: np.ndarray) -> Tuple[bool, float]:
        tpl = self._tpl_gray.get("logo1")
        if tpl is None:
            return False, 0.0
        best = 0.0
        for scale in np.linspace(0.08, 1.6, 18):
            th, tw = int(tpl.shape[0] * scale), int(tpl.shape[1] * scale)
            if th < 10 or tw < 10 or th > gray.shape[0] or tw > gray.shape[1]:
                continue
            r = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(gray, r, cv2.TM_CCOEFF_NORMED)
            _, v, _, _ = cv2.minMaxLoc(res)
            best = max(best, v)
        return best >= self.threshold, best

    def detect_logo1(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> Tuple[bool, float]:
        region = self._roi(frame, roi)
        gray_raw = self._gray(region)
        gray_enh = self._enhance(gray_raw)

        tpl_hit, tpl_conf = self._logo1_template(gray_enh)
        orb_hit, orb_conf = self._logo1_orb(gray_enh)

        # Template is the primary gate — must clear the threshold on its own.
        # Color check acts as a tie-breaker only when template is already strong.
        detected = tpl_hit or (orb_hit and tpl_conf >= self.threshold * 0.85)

        # Use the highest single-method confidence as the display value
        conf = max(tpl_conf, orb_conf)
        return detected, min(conf, 1.0)

    # ── OCR helpers ───────────────────────────────────────────────────────────

    def _ocr_versions(self, gray: np.ndarray) -> List[np.ndarray]:
        """Up to 5 preprocessed variants for maximum OCR recall."""
        # Upscale small regions — Tesseract loves ≥300 DPI equivalent
        h, w = gray.shape[:2]
        if max(h, w) < 200:
            scale = max(2, 300 // max(h, w, 1))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        enh = self._clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enh, h=9, templateWindowSize=7, searchWindowSize=21)

        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adp = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        return [otsu, cv2.bitwise_not(otsu), adp, cv2.bitwise_not(adp), denoised]

    def _run_ocr(self, gray: np.ndarray) -> str:
        if not OCR_AVAILABLE:
            return ""
        texts = []
        for img in self._ocr_versions(gray):
            for psm in ("6", "7", "11"):
                try:
                    texts.append(
                        pytesseract.image_to_string(img, config=f"--psm {psm} --oem 3")
                    )
                except Exception:
                    pass
        return " ".join(texts).lower()

    def detect_logo2(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> bool:
        """A.M.MOTORS sticker."""
        text = re.sub(r"[^a-z]", "", self._run_ocr(self._gray(self._roi(frame, roi))))
        return "ammotors" in text or "ammotor" in text or ("am" in text and "motors" in text)

    def detect_logo3(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> bool:
        """www.ammotors.in sticker."""
        text = re.sub(r"[^a-z.]", "", self._run_ocr(self._gray(self._roi(frame, roi))))
        return "ammotors.in" in text or ("ammotors" in text and ".in" in text)

    # ── Plate detection ───────────────────────────────────────────────────────

    def detect_plate(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> str:
        if not OCR_AVAILABLE:
            return ""
        region = self._roi(frame, roi)
        gray   = self._gray(region)

        # Try multiple preprocessed versions
        for src in (
            cv2.bilateralFilter(gray, 11, 17, 17),
            self._enhance(gray),
            gray,
        ):
            txt = self._plate_from_gray(src)
            if txt:
                return txt
        return ""

    def _plate_from_gray(self, gray: np.ndarray) -> str:
        edges = cv2.Canny(gray, 25, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        for cnt in cnts:
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < 60 or h < 14:
                continue
            if not (2.0 < w / float(h) < 6.5):
                continue
            plate = cv2.resize(gray[y:y+h, x:x+w], None, fx=3, fy=3,
                               interpolation=cv2.INTER_CUBIC)
            for img in (
                cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2),
            ):
                try:
                    t = pytesseract.image_to_string(
                        img,
                        config=f"--psm 8 --oem 3 -c tessedit_char_whitelist={self.PLATE_CHARS}",
                    )
                    cleaned = re.sub(r"[^A-Z0-9]", "", t.upper())
                    if 5 <= len(cleaned) <= 12:
                        return cleaned
                except Exception:
                    pass
        return ""

    # ── Unknown sticker detection ─────────────────────────────────────────────

    def detect_unknown_sticker(
        self, frame: np.ndarray, roi: Optional[Tuple] = None, logo1_conf: float = 0.0
    ) -> bool:
        """
        Returns True when there appears to be a non-AM-Motors sticker present.
        Strategy: find high-edge-density rectangular patches in the frame that
        are sticker-sized but did not match any AM Motors template.
        Only fires when AM Motors stickers were NOT found (logo1_conf < threshold).
        """
        if logo1_conf >= self.threshold:
            return False  # confirmed AM Motors sticker — not unknown

        region = self._roi(frame, roi)
        gray   = self._enhance(self._gray(region))
        edges  = cv2.Canny(gray, 40, 130)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = region.shape[0] * region.shape[1]

        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(cnt)
            # Must be large enough to be a sticker but not the whole car
            if area < 400 or area > frame_area * 0.20:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            ar = w / float(h)
            # Sticker aspect ratios: square-ish to banner-wide
            if not (3 <= len(approx) <= 6 and 0.4 < ar < 7.0):
                continue
            patch = self._gray(region)[y:y + h, x:x + w]
            if patch.size == 0:
                continue
            # Edge density inside the patch — real stickers have internal pattern/text
            inner_edges = cv2.Canny(patch, 40, 130)
            density = cv2.countNonZero(inner_edges) / patch.size
            if density > 0.07:
                return True

        return False

    # ── Full detection pipeline ───────────────────────────────────────────────

    def detect_all(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple] = None,
        use_ocr: bool = True,
        use_logo: bool = True,
        motion_hint: Optional[Tuple] = None,
    ) -> DetectionResult:
        result = DetectionResult(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        active_roi = motion_hint or roi

        if use_logo:
            result.logo1, result.logo1_conf = self.detect_logo1(frame, active_roi)
        if use_ocr:
            result.logo2 = self.detect_logo2(frame, active_roi)
            result.logo3 = self.detect_logo3(frame, active_roi)
            result.vehicle_number = self.detect_plate(frame, active_roi)

        # Unknown sticker: something is there but it's not AM Motors
        if not result.logo1 and not result.logo2 and not result.logo3:
            result.unknown_sticker = self.detect_unknown_sticker(
                frame, active_roi, result.logo1_conf
            )

        return result

    # ── Frame annotation ──────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        roi: Optional[Tuple] = None,
        motion_roi: Optional[Tuple] = None,
    ) -> np.ndarray:
        out = frame.copy()

        if roi:
            x, y, w, h = roi
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 220, 0), 2)
            cv2.putText(out, "ROI", (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

        if motion_roi:
            mx, my, mw, mh = motion_roi
            cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (0, 200, 255), 2)
            cv2.putText(out, "Motion", (mx + 4, my + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        lines = [
            (f"Logo (Triangle): {'YES' if result.logo1 else 'NO'}  ({result.logo1_conf:.2f})", result.logo1),
            (f"A.M.MOTORS:      {'YES' if result.logo2 else 'NO'}", result.logo2),
            (f"www.ammotors.in: {'YES' if result.logo3 else 'NO'}", result.logo3),
            (f"Plate: {result.vehicle_number or 'N/A'}", bool(result.vehicle_number)),
            (f"Unknown sticker: {'DETECTED' if result.unknown_sticker else 'no'}", result.unknown_sticker),
        ]

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (370, len(lines) * 36 + 15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        for i, (text, hit) in enumerate(lines):
            color = (0, 255, 80) if hit else (60, 60, 255)
            cv2.putText(out, text, (8, 30 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return out
