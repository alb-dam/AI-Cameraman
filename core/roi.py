"""Gestione unificata della Region of Interest (ROI).

Combina caricamento/salvataggio file, stato di editing (per la GUI),
e logica matematica vettoriale per le maschere di ritaglio.
"""

import json
import os
from typing import List, Tuple, Optional
import logging

import cv2
import numpy as np

from core.geometry import GeometryService
from core.models import ROI, Detection

logger = logging.getLogger(__name__)


class ROIManager:
    """Coordinatore ROI: persistenza, editing, maschere vettoriali."""

    def __init__(self) -> None:
        self.roi: ROI = ROI()
        self.roi_points: List[Tuple[float, float]] = []
        self.editing_mode: bool = False
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_mask_size: Optional[Tuple[int, int]] = None  # (w, h)

    @property
    def is_valid(self) -> bool:
        """True se il poligono ha almeno 3 vertici ed è stato caricato."""
        return self.roi.is_valid and self.roi.polygon is not None

    # ── Persistenza ─────────────────────────────────────────────────────

    def load_roi(self, filepath: str) -> bool:
        """Carica punti normalizzati da file JSON e aggiorna il mask engine."""
        if not os.path.exists(filepath):
            logger.warning("File ROI non trovato: %s", filepath)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Errore lettura ROI da %s: %s", filepath, e)
            return False

        if "points" not in data:
            logger.warning("Formato file ROI non valido: manca 'points' in %s", filepath)
            return False

        self.roi_points = [tuple(p) for p in data["points"]]
        self._update_polygon(self.roi_points)
        logger.info("ROI caricata: %d punti da %s", len(self.roi_points), filepath)
        return True

    def save_roi(self, filepath: str) -> bool:
        """Salva i punti ROI normalizzati in un file JSON."""
        try:
            data = {"normalized": True, "points": self.roi_points}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info("ROI salvata in %s", filepath)
            return True
        except OSError as e:
            logger.error("Errore salvataggio ROI in %s: %s", filepath, e)
            return False

    # ── Creazione da punti (API diretta) ────────────────────────────────

    def create_from_points(self, points: List[Tuple[float, float]]) -> None:
        """Imposta la ROI da una lista e aggiorna il mask engine."""
        self.roi_points = list(points)
        self._update_polygon(self.roi_points)

    def _update_polygon(self, points: List[Tuple[float, float]]) -> None:
        """Aggiorna la shape matematica del poligono e invalida la cache della maschera."""
        self._cached_mask = None
        self._cached_mask_size = None
        if len(points) >= 3:
            self.roi.polygon = np.array(points, dtype=np.float32)
        else:
            self.roi.polygon = None
            logger.info("Punti insufficienti per creare un poligono (min 3). ROI disattivata.")

    # ── Editing State ───────────────────────────────────────────────────

    def start_roi_selection(self) -> None:
        """Reset punti e attiva modalità editing."""
        self.roi_points = []
        self.roi.polygon = None
        self.editing_mode = True
        logger.info("Modalità editing ROI avviata.")

    def add_point(self, norm_x: float, norm_y: float) -> None:
        """Aggiunge un punto normalizzato alla ROI (solo se in editing)."""
        if not self.editing_mode:
            return
        self.roi_points.append((norm_x, norm_y))
        logger.debug("ROI punto aggiunto: (%.3f, %.3f)", norm_x, norm_y)

    def finalize_roi(self) -> None:
        """Disattiva editing e aggiorna il mask engine con i punti raccolti."""
        if not self.editing_mode:
            return
        self.editing_mode = False
        self._update_polygon(self.roi_points)
        logger.info("Modalità editing ROI terminata (%d punti).", len(self.roi_points))

    # ── OpenCV Applicazione ed Esposizione ──────────────────────────────

    def apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Applica maschera ROI al frame con cache (zero-alloc dopo il primo frame).
        Se in modalità editing o invalida, restituisce il frame invariato.
        """
        if self.editing_mode or not self.is_valid:
            return frame
        
        h, w = frame.shape[:2]
        current_size = (w, h)
        
        # Ricostruisci la cache solo se la dimensione del frame è cambiata o la cache è vuota
        if self._cached_mask is None or self._cached_mask_size != current_size:
            self._cached_mask = GeometryService.build_polygon_mask(self.roi.polygon, w, h)
            self._cached_mask_size = current_size
            logger.debug("ROI mask cache rigenerata per dimensione %dx%d", w, h)
        
        return GeometryService.apply_precomputed_mask(frame, self._cached_mask)

    def filter_detections_by_feet(
        self,
        detections: List[Detection],
        frame_w: int,
        frame_h: int,
    ) -> List[Detection]:
        """Filtra le detection in base alla posizione dei piedi rispetto alla ROI.

        - Piedi dentro la ROI → detection inclusa (anche se il busto esce)
        - Piedi fuori dalla ROI → detection esclusa (anche se il busto è dentro)

        Il punto 'piedi' è definito come (centro_x della bbox, bordo inferiore y2).
        Se la ROI non è valida o si è in modalità editing, tutte le detection passano.
        """
        if self.editing_mode or not self.is_valid or self.roi.polygon is None:
            return detections

        # Converti il poligono normalizzato in coordinate pixel assolute
        abs_polygon = (self.roi.polygon * [frame_w, frame_h]).astype(np.float32)

        filtered: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            foot_x = (x1 + x2) / 2.0
            foot_y = float(y2)  # bordo inferiore = piedi

            # pointPolygonTest: > 0 dentro, = 0 sul bordo, < 0 fuori
            dist = cv2.pointPolygonTest(abs_polygon, (foot_x, foot_y), measureDist=False)
            if dist >= 0:
                filtered.append(det)

        return filtered

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        """Disegna il poligono ROI sul frame (solo se non in editing)."""
        if self.editing_mode or not self.is_valid:
            return
        GeometryService.draw_polygon(
            frame, self.roi.polygon, color=color, thickness=thickness
        )

    # ── Generazione automatica ROI con SAM3 ─────────────────────────────

    def generate_roi_from_video(self, frames: List[np.ndarray], save_path: str) -> bool:
        """Genera automaticamente la ROI del campo da basket usando SAM3.

        1. Calcola la mediana dei frame per rimuovere i giocatori
        2. Applica filtro Gaussiano per ridurre artefatti residui
        3. Segmenta il campo con SAM3 (text prompt 'basketball court')
        4. Estrae il contorno della maschera più grande come punti normalizzati

        Args:
            frames: Lista di frame (np.ndarray BGR) campionati dal video.
            save_path: Percorso dove salvare il file roi.json generato.

        Returns:
            True se la ROI è stata generata con successo, False altrimenti.
        """
        if len(frames) < 2:
            logger.error("Genera ROI: servono almeno 2 frame, ricevuti %d.", len(frames))
            return False

        try:
            # 1. Mediana dei frame per rimuovere oggetti in movimento (giocatori)
            logger.info("Genera ROI: calcolo mediana di %d frame...", len(frames))
            stacked = np.stack(frames, axis=0)
            median_frame = np.median(stacked, axis=0).astype(np.uint8)

            # 2. Filtro bilaterale: sfuma il rumore ma preserva i bordi netti del campo
            median_frame = cv2.bilateralFilter(median_frame, d=9, sigmaColor=75, sigmaSpace=75)

            # 3. CLAHE (contrasto adattivo) + riduzione esposizione per esaltare il campo
            lab = cv2.cvtColor(median_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            l = np.clip(l * 0.8, 0, 255).astype(np.uint8)  # riduzione esposizione 20%
            lab = cv2.merge([l, a, b])
            median_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            cv2.imwrite("debug_median_frame.png", median_frame)
            logger.info("Genera ROI: filtro bilaterale + CLAHE + riduzione esposizione applicati. "
                        "Mediana salvata in debug_median_frame.png.")

            # 3. Segmentazione con SAM3
            logger.info("Genera ROI: avvio segmentazione SAM3...")
            masks, scores = self._run_sam3_segmentation(median_frame)
            if masks is None or len(masks) == 0:
                logger.error("Genera ROI: SAM3 non ha prodotto maschere.")
                return False

            # 4. Unisci maschere significative (>5% del frame) per includere zone colorate
            h, w = median_frame.shape[:2]
            min_area = h * w * 0.05  # soglia 5% dell'area totale
            merged_mask = np.zeros((h, w), dtype=np.uint8)
            included = 0
            for m in masks:
                m_uint8 = (m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8)
                if m_uint8.shape != (h, w):
                    m_uint8 = cv2.resize(m_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                # Scarta maschere troppo piccole (tabelloni, panchine, ecc.)
                if cv2.countNonZero(m_uint8) < min_area:
                    continue
                merged_mask = cv2.bitwise_or(merged_mask, m_uint8)
                included += 1
            logger.info("Genera ROI: unite %d/%d maschere SAM3 (scartate %d < 5%% frame).",
                        included, len(masks), len(masks) - included)

            # 5. Post-processing morfologico: kernel grande per unire zone separate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
            logger.info("Genera ROI: post-processing morfologico applicato.")

            # 6. Estrai contorno e applica convex hull per includere tutto il campo
            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.error("Genera ROI: nessun contorno trovato nella maschera.")
                return False

            # Prendi il contorno più grande e calcola il convex hull
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Semplifica il convex hull per un poligono più pulito
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # Normalizza i punti (0-1)
            norm_points = [(float(p[0][0]) / w, float(p[0][1]) / h) for p in approx]

            if len(norm_points) < 3:
                logger.error("Genera ROI: contorno troppo semplice (%d punti).", len(norm_points))
                return False

            # Margine di sicurezza 2%: espandi i punti dal centroide
            cx = sum(p[0] for p in norm_points) / len(norm_points)
            cy = sum(p[1] for p in norm_points) / len(norm_points)
            margin = 1.02  # 2% di espansione
            norm_points = [
                (max(0.0, min(1.0, cx + (px - cx) * margin)),
                 max(0.0, min(1.0, cy + (py - cy) * margin)))
                for px, py in norm_points
            ]

            # Chiudi il poligono
            if norm_points[0] != norm_points[-1]:
                norm_points.append(norm_points[0])

            # 6. Imposta la ROI e salva
            self.create_from_points(norm_points)
            self.save_roi(save_path)
            logger.info("Genera ROI: completata con successo (%d punti).", len(norm_points))
            return True

        except Exception as e:
            logger.error("Genera ROI: errore durante la generazione: %s", e)
            return False

    @staticmethod
    def _run_sam3_segmentation(frame: np.ndarray):
        """Esegue la segmentazione SAM3 con text prompt 'basketball court'.

        Returns:
            Tupla (masks, scores) dove masks è una lista di array binari
            e scores le relative confidenze, oppure (None, None) in caso di errore.
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            logger.error("Genera ROI: ultralytics SAM3SemanticPredictor non disponibile. "
                         "Aggiornare ultralytics: pip install -U ultralytics")
            return None, None

        import sys
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        model_path = os.path.join(base_path, "assets", "sam3.pt")
        
        if not os.path.exists(model_path):
            logger.error("Genera ROI: modello SAM3 non trovato in %s. "
                         "Scaricarlo da HuggingFace: "
                         "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt",
                         model_path)
            return None, None

        try:
            overrides = dict(
                conf=0.50,
                task="segment",
                mode="predict",
                model=model_path,
                save=False,
                verbose=False,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
            predictor.set_image(frame)
            results = predictor(text=["basketball court"])

            if results is None or len(results) == 0:
                return None, None

            result = results[0]
            if result.masks is None or result.masks.data is None:
                return None, None

            masks = result.masks.data.cpu().numpy()
            # Threshold esplicito: solo pixel con confidenza >= 0.7
            masks = (masks >= 0.7).astype(np.uint8)
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
            return masks, scores

        except Exception as e:
            logger.error("Genera ROI: errore SAM3: %s", e)
            return None, None


def roi_manager_run(filepath: Optional[str] = None) -> ROIManager:
    """Entry point per la creazione del ROIManager."""
    rm = ROIManager()
    if filepath and os.path.exists(filepath):
        rm.load_roi(filepath)
    return rm

