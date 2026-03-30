"""Inferenza YOLOE isolata dal resto del codice.

Riceve un frame e restituisce bounding box grezzi e maschere di segmentazione.
Non mantiene stato di tracking.

Utilizza YOLOE-26 (open-vocabulary) per:
- Detection giocatori e pallone tramite text prompt
- Segmentazione del campo per la generazione automatica della ROI
"""

from typing import List, Dict, Any, Tuple, Optional
import threading
import logging
import numpy as np

from core.models import Detection

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLOE  # type: ignore
except ImportError:
    logger.warning("Modulo ultralytics (YOLOE) non trovato. AI disabilitata.")
    YOLOE = None  # type: ignore


class YoloDetector:
    """Inferenza YOLOE: riceve un frame, restituisce bounding box grezzi."""

    # Nomi classi open-vocabulary per detection
    PLAYER_CLASS_NAME = "person"
    BALL_CLASS_NAME = "sports ball"
    # Soglie morfologiche per il filtraggio del pallone
    MAX_BALL_SIZE_RATIO: float = 0.20   # La palla non deve superare il 20% del lato minore
    MAX_BALL_ASPECT_RATIO: float = 3.0  # Aspect ratio massimo accettabile (filtra allucinazioni)

    def __init__(self, model_name: str = "assets/yoloe-26s-seg.pt", ball_conf_thresh: float = 0.4) -> None:
        self._ensure_mobileclip_exists()
        self.device, self.use_half = self._detect_device()
        resolved_path = self._resolve_model_path(model_name)
        self.model: Any = self._load_model(resolved_path)
        self.ball_conf_thresh: float = ball_conf_thresh
        self._inference_counter = 0
        self._class_names: Dict[int, str] = {}  # Mappa id -> nome classe dal modello

    @staticmethod
    def _ensure_mobileclip_exists() -> None:
        """Assicura che mobileclip2_b.ts sia in assets/ invece che nella root."""
        import os
        import sys
        
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        assets_dir = os.path.join(base_path, 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        clip_path = os.path.join(assets_dir, "mobileclip2_b.ts")
        
        # Aggiorniamo ultralytics per fargli cercare i modelli in assets
        try:
            from ultralytics import settings as ultra_settings
            ultra_settings.update({'weights_dir': assets_dir})
        except ImportError:
            pass

        if not os.path.exists(clip_path):
            import urllib.request
            logger.warning("YOLOE: mobileclip2_b.ts non trovato in assets/. Avvio download...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"
            try:
                def report_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(int(block_num * block_size * 100 / total_size), 100)
                        sys.stdout.write(f"\rScaricamento mobileclip2_b.ts: {percent}%")
                        sys.stdout.flush()

                urllib.request.urlretrieve(url, clip_path, reporthook=report_progress)
                sys.stdout.write("\n")
                logger.info("YOLOE: Download di mobileclip2_b.ts in assets/ completato con successo.")
            except Exception as e:
                logger.error(f"YOLOE: Errore nel download di mobileclip2_b.ts: {e}")

    @staticmethod
    def _resolve_model_path(base_name: str, force_pt: bool = False) -> str:
        """Cerca versioni ottimizzate del modello (CoreML, TensorRT, ONNX) prima di usare o scaricare il file .pt.
        
        Se force_pt=True, bypassa la ricerca della versione ottimizzata (necessario per segmentazioni
        con prompt dinamici, in quanto i text-embeddings dei modelli esportati sono statici e baked-in).
        """
        import os
        import platform
        import sys
        
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        if not os.path.isabs(base_name):
            base_name = os.path.join(base_path, base_name)
            
        name_no_ext, ext = os.path.splitext(base_name)
        
        # Se l'utente ha esplicitamente richiesto un'estensione non .pt (es in config.json), forziamo fiduciosi
        if ext != '.pt':
             return base_name
             
        system = platform.system()
        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass
            
        optimized_exts = []
        if system == "Darwin":
            optimized_exts = [".mlpackage", "_mac.mlpackage"]
        elif has_cuda:
            optimized_exts = [".engine"]
        else:
            optimized_exts = [".onnx"]
            
        if not force_pt:
            for opt_ext in optimized_exts:
                opt_path = name_no_ext + opt_ext
                if os.path.exists(opt_path):
                    logger.info(f"YOLOE: Modello ottimizzato rilevato e selezionato: {opt_path}")
                    return opt_path
                
        # Se non c'è la versione ottimizzata, cerchiamo il fallback .pt
        if os.path.exists(base_name):
            logger.info(f"YOLOE: Rilevato il base .pt: {base_name}. Utilizza scripts/export_model.py per ottimizzarlo.")
            return base_name
            
        # Altrimenti, scarica automatico
        logger.warning("YOLOE: Modello non trovato. Avvio download automatico...")
        os.makedirs(os.path.dirname(base_name), exist_ok=True)
        try:
            import urllib.request
            filename = os.path.basename(base_name)
            url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{filename}"
            
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(int(block_num * block_size * 100 / total_size), 100)
                    sys.stdout.write(f"\rScaricamento {filename}: {percent}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, base_name, reporthook=report_progress)
            sys.stdout.write("\n")
            logger.info(f"YOLOE: Download di {filename} completato con successo.")
        except Exception as e:
            logger.error(f"YOLOE: Errore nel download automatico: {e}")
            
        return base_name

    @staticmethod
    def _detect_device() -> Tuple[str, bool]:
        """Detects the best available hardware device and whether to use half precision."""
        device = "cpu"
        use_half = False
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
                use_half = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                use_half = False  # YOLOE text embeddings non supportano half su MPS
        except ImportError:
            pass
        logger.info(f"YOLOE configurato per usare device: {device}, precisione half: {use_half}")
        return device, use_half

    def detect(self, frame: np.ndarray, imgsz: int = 640) -> Tuple[List[Detection], Optional[Detection]]:
        """Esegue l'inferenza e ritorna (raw_players, raw_ball)."""
        if self.model is None:
            return [], None

        try:
            results = self.model.predict(source=frame, imgsz=imgsz, verbose=False, half=self.use_half,
                                         device=self.device)
        except Exception as e:
            self._free_memory()
            raise e
        finally:
            self._inference_counter += 1
            if self._inference_counter >= 1000:
                self._free_memory_async()
                self._inference_counter = 0

        raw_players: List[Detection] = []
        raw_ball: Optional[Detection] = None

        img_h, img_w = frame.shape[:2]

        # Aggiorna la mappa nomi classi dal modello
        if hasattr(results[0], 'names') and results[0].names:
            self._class_names = results[0].names

        for box in results[0].boxes:
            detection = self._parse_box(box)
            if detection is None:
                continue

            cls_id, entry = detection
            cls_name = self._class_names.get(cls_id, "").lower()

            if cls_name == self.PLAYER_CLASS_NAME:
                raw_players.append(entry)
            elif cls_name == self.BALL_CLASS_NAME:
                if entry.conf < self.ball_conf_thresh:
                    continue
                
                # Verifica morfologica
                bw = entry.box[2] - entry.box[0]
                bh = entry.box[3] - entry.box[1]
                aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
                max_ball_dim = min(img_w, img_h) * self.MAX_BALL_SIZE_RATIO
                
                # Rifiuta box troppo grandi o troppo schiacciati (allucinazioni)
                if bw > max_ball_dim or bh > max_ball_dim or aspect_ratio > self.MAX_BALL_ASPECT_RATIO:
                    continue

                if raw_ball is None or entry.conf > raw_ball.conf:
                    raw_ball = entry

        return raw_players, raw_ball

    def segment(self, frame: np.ndarray, text_prompts: List[str], conf: float = 0.05, imgsz: int = 640):
        """Esegue segmentazione con text prompt tramite YOLOE.

        Args:
            frame: frame BGR da segmentare.
            text_prompts: lista di classi testuali (es. ["basketball court"]).
            conf: soglia di confidenza minima (default basso per superfici ampie).
            imgsz: dimensione immagine per l'inferenza.

        Returns:
            Tupla (masks, scores) dove masks è un array numpy di maschere binarie
            e scores le relative confidenze, oppure (None, None) in caso di errore.
        """
        if YOLOE is None:
            logger.error("Segmentazione YOLOE: modulo ultralytics non disponibile.")
            return None, None

        try:
            import os
            import sys

            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

            model_base = os.path.join(base_path, "assets", "yoloe-26s-seg.pt")
            
            # Forza l'uso del .pt per la ROI perché i formati esportati (mlpackage/engine)
            # nascono con i text embeddings del training/export statici (person/ball). 
            # I custom prompt dinamici richiedono il grafo PyTorch originale modificabile runtime.
            model_path = self._resolve_model_path(model_base, force_pt=True)

            # Crea un'istanza separata per la segmentazione con text prompt
            seg_model = YOLOE(model_path)
            seg_model.set_classes(text_prompts)

            results = seg_model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=self.device,
                half=self.use_half,
            )

            if results is None or len(results) == 0:
                logger.warning("Segmentazione YOLOE: nessun risultato dal modello.")
                return None, None

            result = results[0]

            # Log diagnostico
            n_boxes = len(result.boxes) if result.boxes is not None else 0
            logger.info("Segmentazione YOLOE: %d detection trovate (conf >= %.2f).", n_boxes, conf)
            for b in result.boxes:
                cls_id = int(b.cls[0])
                cls_name = result.names.get(cls_id, "?")
                c = float(b.conf[0])
                logger.info("  → cls=%d (%s) conf=%.3f", cls_id, cls_name, c)

            if result.masks is None or result.masks.data is None:
                logger.warning("Segmentazione YOLOE: nessuna maschera prodotta.")
                return None, None

            masks = result.masks.data.cpu().numpy()
            masks = (masks >= 0.5).astype(np.uint8)
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
            logger.info("Segmentazione YOLOE: %d maschere estratte.", len(masks))
            return masks, scores

        except Exception as e:
            logger.error("Segmentazione YOLOE: errore: %s", e)
            return None, None

    def _free_memory_async(self) -> None:
        """Avvia la pulizia memoria in background per non bloccare il thread di inferenza."""
        t = threading.Thread(target=self._free_memory, daemon=True)
        t.start()

    def _free_memory(self) -> None:
        """Svuota la cache GPU per prevenire Memory Leaks a lungo termine."""
        try:
            import gc
            gc.collect(0)  # Solo generazione 0 — incrementale, ~1ms vs ~5-50ms
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _load_model(self, model_path: str) -> Any:
        """Tenta il caricamento del modello YOLOE e configura le classi di detection."""
        if YOLOE is None:
            return None
            
        
        try:
            model = YOLOE(model_path)

            # set_classes funziona (e serve) SOLO sui modelli PyTorch nativi (.pt).
            # I modelli ottimizzati (.mlpackage, .engine, etc.) hanno le classi 'baked in' durante l'export script.
            if model_path.endswith('.pt'):
                model.set_classes([self.PLAYER_CLASS_NAME, self.BALL_CLASS_NAME])
                logger.info(f"YOLOE: classi impostate su dinamico: [{self.PLAYER_CLASS_NAME}, {self.BALL_CLASS_NAME}]")
            else:
                logger.info("YOLOE: usa modello ottimizzato. I test embeddings sono statici nel grafo pre-compilato.")

            if model_path.endswith('.pt') and self.device != "cpu":
                try:
                    model.to(self.device)
                    logger.info(f"Modello caricato su acceleratore hardware {self.device}.")
                except Exception as e:
                    logger.warning(f"Impossibile spostare il modello su {self.device}: {e}")
            return model
        except Exception as e:
            logger.error(f"Errore caricamento modello {model_path}: {e}")
            return None

    @staticmethod
    def _parse_box(box: Any) -> Optional[Tuple[int, Detection]]:
        """Estrae classe, confidence e coordinate da un singolo box YOLO."""
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cls_id, Detection(box=(x1, y1, x2, y2), center=(cx, cy), conf=conf)
