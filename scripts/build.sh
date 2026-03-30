#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# Build script per AI-Cameraman
# Genera l'eseguibile standalone con PyInstaller
# ─────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "═══════════════════════════════════════════════════════"
echo "  AI-Cameraman Build"
echo "═══════════════════════════════════════════════════════"

# 1. Attiva il virtualenv se presente
if [ -d ".venv" ]; then
    echo "📦 Attivazione virtualenv .venv..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 2. Installa PyInstaller se necessario
if ! python -m PyInstaller --version &>/dev/null; then
    echo "📥 Installazione PyInstaller..."
    pip install pyinstaller
fi

# 3. Pulizia build precedenti
echo "🧹 Pulizia build precedenti..."
rm -rf build/ dist/

# 4. Build
echo ""
echo "🔨 Avvio build PyInstaller..."
echo "  Il build richiederà alcuni minuti."
echo ""
python -m PyInstaller scripts/ai_cameraman.spec --noconfirm

# 5. Pulizia file temporanei
rm -f _runtime_hook.py

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Build completato!"
echo ""
if [ "$(uname)" = "Darwin" ]; then
    echo "  📁 App:    dist/AI-Cameraman.app"
    echo "  📁 Bundle: dist/AI-Cameraman/"
else
    echo "  📁 Output: dist/AI-Cameraman/"
fi
echo ""
echo "  Per eseguire:"
if [ "$(uname)" = "Darwin" ]; then
    echo "    open dist/AI-Cameraman.app"
    echo "    oppure: dist/AI-Cameraman/AI-Cameraman"
else
    echo "    dist/AI-Cameraman/AI-Cameraman"
fi
echo "═══════════════════════════════════════════════════════"
