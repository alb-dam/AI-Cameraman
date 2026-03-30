#!/usr/bin/env python3
"""
Script per esportare il modello YOLOE-26 nel formato ottimale per il sistema corrente:
- macOS: CoreML (.mlpackage)
- Windows/Linux con GPU NVIDIA: TensorRT (.engine)
- Altro (es. CPU-only o hardware non supportato): ONNX (.onnx)
"""

import os
import sys
import logging
import platform

# Configurazione logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ExportScript")

try:
    import torch
    from ultralytics import YOLOE
except ImportError:
    logger.error("Dipendenze mancanti. Assicurati di aver installato ultralytics e torch nel tuo virtual environment.")
    logger.info("Esegui: pip install -r requirements.txt")
    sys.exit(1)

def export_model():
    # Percorso del modello
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "assets", "yoloe-26s-seg.pt")
    
    if not os.path.exists(model_path):
        logger.warning(f"Modello sorgente non trovato in: {model_path}")
        logger.info("Avvio del download automatico (yoloe-26s-seg.pt)...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            import urllib.request
            url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt"
            
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(int(block_num * block_size * 100 / total_size), 100)
                    sys.stdout.write(f"\rScaricamento: {percent}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
            sys.stdout.write("\n")
            logger.info("✅ Download completato con successo.")
        except Exception as e:
            logger.error(f"❌ Errore durante il download del modello: {e}")
            sys.exit(1)

    logger.info(f"Caricamento modello YOLOE da: {model_path}")
    # Carica il modello
    try:
        model = YOLOE(model_path)
        
        # ⚠️ FONDAMENTALE PER I MODELLI OPEN-VOCABULARY ⚠️
        # Inserisce in modo permanente le estrazioni del testo (text-embeddings) nel grafo computazionale
        # statico del TensorRT / CoreML / ONNX. Senza questo diventeranno un coco80 base rotto.
        logger.info("Impostazione statiche delle classi (person, sports ball) per l'export...")
        model.set_classes(["person", "sports ball"])
        
    except Exception as e:
        logger.error(f"Errore nel caricamento del modello: {e}")
        sys.exit(1)

    # Rilevamento piattaforma e accelerazione
    system = platform.system()
    has_cuda = torch.cuda.is_available()

    logger.info(f"Sistema operativo rilevato: {system}")
    logger.info(f"GPU NVIDIA (CUDA) disponibile: {has_cuda}")

    # Parametri di esportazione base
    export_kwargs = {
        "imgsz": 640,
        "half": False,  # Default sicuro per prevenire Type mismatches (soprattutto con il text encoder di YOLOE)
    }

    # Determinazione target di export
    if system == "Darwin":
        logger.info("➡ Piattaforma macOS. Esportazione in formato CoreML (.mlpackage).")
        export_format = "coreml"
        export_kwargs["nms"] = True  # Aggiunge logic NMS all'interno del modello CoreML per massimizzare la velocità
    elif has_cuda:
        logger.info("➡ GPU NVIDIA rilevata. Esportazione in formato TensorRT (.engine).")
        export_format = "engine"
        export_kwargs["half"] = True      # TensorRT trae forte beneficio da FP16
        export_kwargs["workspace"] = 4    # GB di VRAM da dedicare alla built (può essere aumentato/diminuito)
    else:
        logger.info("➡ Nessuna ottimizzazione specifica target trovata. Esportazione in formato ONNX per inferenza CPU.")
        export_format = "onnx"

    logger.info(f"Avvio esportazione in formato '{export_format}'. **QUESTO PROCESSO POTREBBE RICHIEDERE DIVERSI MINUTI**...")
    
    try:
        exported_path = model.export(format=export_format, **export_kwargs)
        logger.info(f"✅ Esportazione completata con successo!")
        logger.info(f"Modello salvato in: {exported_path}")
        logger.info("L'applicazione lo rileverà automaticamente se il suo nome verrà aggiornato nella configurazione o se l'estensione supportata è preferita dall'app.")
    except Exception as e:
        logger.error(f"❌ Errore critico durante l'esportazione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("--- Inizio script di esportazione YOLOE ---")
    export_model()
