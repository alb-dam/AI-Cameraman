# 🎥 AI-Cameraman

**Regia virtuale automatica per il basket, basata su intelligenza artificiale.**

AI-Cameraman trasforma una qualsiasi sorgente video fissa (webcam, file, stream SRT) in un'inquadratura dinamica e professionale: rileva i giocatori e il pallone in tempo reale con YOLOE-26, ne traccia i movimenti con filtri di Kalman, e genera automaticamente pan, tilt e zoom fluidi per seguire l'azione. L'output viene trasmesso via **NDI** per l'integrazione diretta in OBS Studio, vMix o qualsiasi software compatibile.

---

## ✨ Funzionalità Principali

| Funzionalità | Descrizione |
|---|---|
| **Rilevamento AI in tempo reale** | YOLOE-26 (open-vocabulary) per il rilevamento di giocatori (classe `person`) e pallone (classe `sports ball`) |
| **Tracking Kalman** | Filtri di Kalman 2D per stabilizzare le posizioni e predire i movimenti tra un'inferenza e l'altra |
| **Regia virtuale (Virtual PTZ)** | Modello matematico unificato per pan, tilt e zoom con deadzone spaziale (leash) e smoothing cinematico |
| **Controllo UI Avanzato** | Regolazione in tempo reale della Regia con slider dedicati per Tolleranza (Deadzone) e Reattività (Inerzia/Smoothing) |
| **Zoom dinamico** | Lo zoom si adatta automaticamente allo spread dei giocatori: più sono raggruppati, più si zooma |
| **Dual NDI Output** | Due canali NDI indipendenti: **AI** (inquadratura elaborata) e **Native** (passthrough originale) |
| **ROI (Region of Interest)** | Maschera poligonale per limitare il rilevamento al solo campo di gioco |
| **Generazione ROI automatica** | Segmentazione automatica del campo con **YOLOE-26** (text-prompted segmentation) |
| **Sorgenti multiple** | Webcam, file video (MP4/AVI/MKV) e stream **SRT** con riconnessione automatica |
| **Accelerazione hardware** | GPU CUDA, Apple MPS o CPU |
| **GUI PySide6** | Interfaccia grafica con preview live, controlli real-time e pannello log |
| **Performance Monitor** | FPS per stage, latenza end-to-end e identificazione automatica del collo di bottiglia |

---

## 🏗️ Architettura

Il sistema è progettato con una **Clean Architecture** a 4 layer disaccoppiati, orchestrati tramite **Dependency Injection** (Factory pattern) e comunicanti attraverso **Protocol** (interfacce).

### Pipeline Multithread a 4 Stadi

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐
│ Capture  │───▶│ Inference  │───▶│  Tracking  │───▶│  Render  │
│ Thread 1 │    │  Thread 2  │    │  Thread 3  │    │ Thread 4 │
└──────────┘    └────────────┘    └────────────┘    └──────────┘
  Lettura         YOLO detect       Kalman filter     NDI output
  frame           (skip pattern)    + Director        + GUI preview
  + NDI Native
```

I thread comunicano tramite **DropFrameQueue** (code a capacità limitata che scartano i frame vecchi) per garantire latenza minima senza accumulo.

### Struttura del Progetto

```
AI-Cameraman/
├── main.py                  # Entry point dell'applicazione
├── config.json              # Configurazione persistente (JSON)
├── requirements.txt         # Dipendenze Python
├── roi.json                 # ROI poligonale salvata (normalizzata 0-1)
│
├── app/                     # Application Layer
│   ├── controller.py        # Lifecycle, threading, I/O video (4 worker thread)
│   ├── factory.py           # IoC Factory: istanzia e inietta le dipendenze
│   ├── pipeline.py          # Pipeline computazionale sincrona (YOLO → Tracking → Regia)
│   ├── state.py             # Stato mutabile globale condiviso tra i thread
│   └── logger.py            # Configurazione centralizzata del logging (console + file rotativo)
│
├── core/                    # Domain Layer (logica pura, nessuna dipendenza I/O)
│   ├── interfaces.py        # Protocol/Interfacce per Dependency Injection
│   ├── models.py            # Dataclass di dominio (Detection, TrackedObject, CameraInstruction, ROI, ...)
│   ├── vision.py            # Detector: compone YOLO + KalmanTracker + ActionCenterCalculator
│   ├── yolo_model.py        # Inferenza YOLOE isolata (detect, segment, device auto-detect)
│   ├── director.py          # Regia virtuale: zoom, pan/tilt, smoothing, deadzone
│   ├── tracking.py          # ActionCenterCalculator (baricentro ponderato dei giocatori)
│   ├── roi.py               # ROIManager: persistenza, editing, maschere vettoriali, generazione YOLOE
│   ├── geometry.py          # Servizi matematici puri (crop, maschere OpenCV, poligoni)
│   ├── performance.py       # PerformanceMonitor: FPS per stage e latenza end-to-end
│   └── thread_manager.py    # DropFrameQueue e WorkerThread
│
├── video/                   # Infrastructure Layer (I/O video)
│   ├── input.py             # VideoInput: webcam, file, SRT (con riconnessione automatica)
│   ├── output.py            # VideoOutput: dual NDI (AI + Native) con letterbox
│   ├── ndi_output.py        # NDISender: wrapper cyndilib con buffer pre-allocato
│   └── overlay.py           # DebugOverlay: disegno bbox, tracking, crop, FPS
│
├── gui/                     # Presentation Layer (PySide6)
│   ├── main_window.py       # Finestra principale: orchestrazione pannelli + segnali Qt
│   └── panels/
│       ├── control_panel.py # Pannello controlli: sorgente, zoom, kalman, ROI, output
│       ├── preview_panel.py # Preview video live con editing ROI interattivo
│       └── log_panel.py     # Pannello log messaggi
│
├── assets/                  # Modelli AI (non versionati in git)
│   └── yoloe-26m-seg.pt     # Modello YOLOE-26m-seg (detection + segmentazione)
│
├── tests/                   # Test suite (pytest)
│   ├── conftest.py          # Fixture condivise
│   ├── app/                 # Test del layer applicativo
│   └── core/                # Test del layer di dominio
│
└── logs/                    # Log rotativi (max 5MB × 3 backup)
```

---

## 🚀 Installazione

### Prerequisiti

- **Python 3.10+**
- **NDI Runtime** installato sul sistema ([download NDI Tools](https://ndi.video/tools/))
- **GPU** (consigliata): NVIDIA con CUDA, oppure Apple Silicon (MPS)

### Setup

```bash
# 1. Clona il repository
git clone https://github.com/alb-dam/AI-Cameraman.git
cd AI-Cameraman

# 2. Crea un ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt
```

> **Nota:** Su Windows le dipendenze PyTorch vengono scaricate automaticamente dal registry CUDA 12.8 grazie all'`--extra-index-url` nel `requirements.txt`.

### Modelli AI

I modelli non sono inclusi nel repository (`.gitignore`). Il modello viene scaricato automaticamente da ultralytics al primo avvio, oppure può essere scaricato manualmente:

| Modello | Utilizzo | Note |
|---|---|---|
| `yoloe-26m-seg.pt` | Rilevamento giocatori/pallone + segmentazione campo | Unico modello per detection e ROI automatica (~67 MB) |

---

## ▶️ Utilizzo

### Avvio

```bash
python main.py
```

L'applicazione si avvia con la GUI PySide6 e inizia immediatamente la cattura dalla sorgente configurata.

### Workflow Tipico

1. **Seleziona la sorgente** dal pannello controlli (webcam, file video, o SRT)
2. **Carica una ROI** (opzionale) per limitare il rilevamento al campo — oppure usa **"Genera ROI"** per la segmentazione automatica con YOLOE-26
3. **Regola i parametri** in tempo reale:
   - **Zoom fisso** — livello base di ingrandimento
   - **Zoom dinamico** — intensità dello zoom adattivo basato sullo spread dei giocatori
   - **Tolleranza Movimento** — quanto la camera ignora i micromovimenti (Reattiva ↔ Tollerante)
   - **Velocità Movimento Regia** — inerzia e fluidità della telecamera (Lento ↔ Rapido)
4. **Avvia l'elaborazione NDI AI** con il pulsante "Avvia Elaborazione"
5. **Ricevi l'output** in OBS Studio o vMix aggiungendo una sorgente NDI

### Uscite NDI

| Canale NDI | Contenuto |
|---|---|
| `AI-Cameraman AI` | Inquadratura elaborata dall'AI (con pan/tilt/zoom automatico) |
| `AI-Cameraman Native` | Video originale passthrough (senza elaborazione) |

---

## ⚙️ Configurazione

La configurazione è persistente nel file `config.json`, modificabile sia dalla GUI che manualmente. Tutti i parametri vengono applicati in tempo reale senza riavvio.

### Parametri Principali

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `source_type` | `str` | `"webcam"` | Tipo sorgente: `webcam`, `file`, `srt` |
| `source_path` | `str` | `"0"` | Indice webcam, percorso file, o porta SRT |
| `debug_mode` | `bool` | `false` | Abilita overlay visivo (bbox, tracking, crop, FPS) |
| `enable_performance_monitor` | `bool` | `false` | Abilita il log delle performance per stage |

### Zoom e Regia

| Parametro | Default | Descrizione |
|---|---|---|
| `fixed_zoom_percent` | `25.0` | Zoom base fisso (0 = nessuno, 100 = 3×) |
| `dynamic_zoom_percent` | `50.0` | Intensità zoom adattivo (0 = disattivato) |
| `director_deadzone_preset_percent` | `25.0` | Tolleranza di movimento del modello PTZ (0 = reattiva, 100 = tollerante) |
| `director_inertia_preset_percent` | `25.0` | Inerzia del modello PTZ (0 = lento/cinematico, 100 = veloce/rapido) |

> I valori di `min` e `max` per l'interpolazione delle 6 variabili interne (es. `director_zoom_deadzone_min`, `director_zoom_smoothing_max`, ecc.) sono configurabili all'interno di `config.json` o editando direttamente `config/settings.py` qualora si desideri cambiare il comportamento matematico degli slider.

### YOLO (Detection)

| Parametro | Default | Descrizione |
|---|---|---|
| `yolo_model` | `"assets/yoloe-26m-seg.pt"` | Percorso del modello YOLOE |
| `yolo_imgsz` | `640` | Risoluzione di inferenza YOLO |
| `yolo_inference_interval` | `6` | Inferenza completa ogni N frame (il tracking invisibile di Kalman in frame-intermedio è fissato matematicamente sulla massima reattività) |

### NDI

| Parametro | Default | Descrizione |
|---|---|---|
| `ndi_ai_name` | `"AI-Cameraman AI"` | Nome del canale NDI elaborato |
| `ndi_native_name` | `"AI-Cameraman Native"` | Nome del canale NDI passthrough |

---

## 🧪 Test

```bash
# Esegui tutti i test
pytest

# Con report di copertura
pytest --cov=app --cov=core --cov=video
```

---

## 📦 Dipendenze Principali

| Pacchetto | Ruolo |
|---|---|
| `torch` + `torchvision` | Backend ML (CUDA 12.8 su Windows, MPS su macOS) |
| `ultralytics` | Framework YOLOE-26 (detection + segmentazione open-vocabulary) |
| `opencv-python` | Cattura video, processing frame, rendering overlay |
| `PySide6` | GUI desktop cross-platform (Qt6) |
| `cyndilib` | Invio video via protocollo NDI |
| `numpy` | Calcolo matriciale (Kalman, geometria, maschere) |

### Dipendenze OS-Specific

| Pacchetto | Piattaforma | Ruolo |
|---|---|---|
| `pygrabber` | Windows | Enumerazione dispositivi DirectShow |
| `onnx` + `onnxruntime-gpu` | Windows | Accelerazione inferenza tramite ONNX Runtime |
| `pyobjc-framework-AVFoundation` | macOS | Enumerazione webcam native |

---

## 🔧 Note Tecniche

- **Risoluzione output**: determinata dinamicamente dalla sorgente video di input.
- **Letterbox**: i frame vengono adattati alla risoluzione output con padding nero per preservare l'aspect ratio.
- **Skip Inference Pattern**: YOLOE viene eseguito ogni `yolo_inference_interval` frame; nei frame intermedi, il tracking continua tramite la sola predizione Kalman, riducendo drasticamente il carico GPU.
- **Modello unico**: YOLOE-26 gestisce sia la detection (giocatori + pallone) tramite `set_classes()` che la segmentazione del campo tramite text prompt, eliminando la necessità di SAM3 (~3.4 GB).
- **Pulizia memoria GPU**: ogni 1000 inferenze viene eseguita una garbage collection incrementale e svuotamento della cache GPU in background.
- **Thread safety**: gli assegnamenti di reference NumPy sono atomici sotto il GIL di Python; le code `DropFrameQueue` garantiscono comunicazione thread-safe senza backpressure.
- **Salvataggio atomico config**: il `config.json` viene scritto in un file temporaneo e poi rinominato con `os.replace()` per prevenire corruzione in caso di crash.

---

## 📄 Licenza

Questo progetto è ad uso personale / educational. Per utilizzo commerciale, contattare l'autore.
