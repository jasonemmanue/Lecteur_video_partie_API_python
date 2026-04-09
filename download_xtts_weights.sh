#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LinguaPlay — Telechargement des poids XTTS-v2 SANS token HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
# Ce script telecharge les poids du modele XTTS-v2 directement depuis
# les URLs publiques de HuggingFace (CDN sans auth) ou depuis le miroir
# hf-mirror.com (acces libre, pas de compte requis).
#
# Usage :
#   bash download_xtts_weights.sh [dossier_destination]
#   bash download_xtts_weights.sh /app/.cache/tts/tts_models--multilingual--multi-dataset--xtts_v2
#
# Par defaut les poids sont places dans le dossier attendu par Coqui TTS :
#   ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Dossier de destination ────────────────────────────────────────────────────
DEFAULT_TTS_HOME="${HOME}/.local/share/tts"
TTS_MODEL_DIR="${1:-${DEFAULT_TTS_HOME}/tts_models--multilingual--multi-dataset--xtts_v2}"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  LinguaPlay — Telechargement XTTS-v2 (sans token HuggingFace)"
echo "══════════════════════════════════════════════════════════════════"
echo "  Destination : ${TTS_MODEL_DIR}"
echo ""

mkdir -p "${TTS_MODEL_DIR}"

# ── Choix du miroir ───────────────────────────────────────────────────────────
# Miroir 1 : hf-mirror.com (pas de compte requis, serveurs en Chine/Europe)
# Miroir 2 : huggingface.co CDN direct (fonctionne sans token pour les fichiers
#            individuels meme sur les repos gates dans certains cas)
# On essaie le miroir 1 en priorite, fallback sur miroir 2.

HF_MIRROR="https://hf-mirror.com"
HF_DIRECT="https://huggingface.co"
REPO_PATH="coqui/XTTS-v2/resolve/main"

# Fichiers du modele XTTS-v2
FILES=(
    "config.json"
    "model.pth"
    "vocab.json"
    "hash.md5"
    "LICENSE.txt"
    "README.md"
    "speakers_xtts.pth"
)

# ── Fonction de telechargement avec fallback ──────────────────────────────────
download_file() {
    local filename="$1"
    local dest="${TTS_MODEL_DIR}/${filename}"

    # Skip si deja present et non vide
    if [[ -f "${dest}" && -s "${dest}" ]]; then
        echo "  [SKIP] ${filename} (deja present)"
        return 0
    fi

    echo "  [DL]   ${filename}..."

    # Essai miroir hf-mirror.com
    if curl --silent --show-error --location --fail \
        --connect-timeout 30 --max-time 1800 \
        --output "${dest}" \
        "${HF_MIRROR}/${REPO_PATH}/${filename}" 2>/dev/null; then
        echo "  [OK]   ${filename} (via hf-mirror.com)"
        return 0
    fi

    # Fallback HuggingFace direct
    if curl --silent --show-error --location --fail \
        --connect-timeout 30 --max-time 1800 \
        --output "${dest}" \
        "${HF_DIRECT}/${REPO_PATH}/${filename}" 2>/dev/null; then
        echo "  [OK]   ${filename} (via huggingface.co direct)"
        return 0
    fi

    echo "  [ERREUR] Impossible de telecharger ${filename}"
    echo "           Essayez manuellement depuis : ${HF_MIRROR}/${REPO_PATH}/${filename}"
    return 1
}

# ── Telechargement de tous les fichiers ───────────────────────────────────────
echo "  Telechargement des fichiers du modele..."
echo ""

ERRORS=0
for f in "${FILES[@]}"; do
    download_file "${f}" || ERRORS=$((ERRORS + 1))
done

echo ""

# ── Verification finale ───────────────────────────────────────────────────────
REQUIRED=("config.json" "model.pth" "vocab.json")
ALL_OK=true

for f in "${REQUIRED[@]}"; do
    fpath="${TTS_MODEL_DIR}/${f}"
    if [[ -f "${fpath}" && -s "${fpath}" ]]; then
        size=$(du -sh "${fpath}" | cut -f1)
        echo "  [OK] ${f} (${size})"
    else
        echo "  [MANQUANT] ${f}"
        ALL_OK=false
    fi
done

echo ""

if $ALL_OK && [[ $ERRORS -eq 0 ]]; then
    echo "  Tous les poids XTTS-v2 sont disponibles."
    echo "  Dossier : ${TTS_MODEL_DIR}"
    echo ""
    echo "  Vous pouvez maintenant demarrer le pipeline sans HF_TOKEN."
else
    echo "  ATTENTION : ${ERRORS} fichier(s) n'ont pas pu etre telecharges."
    echo "  Le modele risque de ne pas fonctionner correctement."
    echo ""
    echo "  Solutions alternatives :"
    echo "    - Verifier la connexion internet"
    echo "    - Telecharger manuellement depuis : ${HF_MIRROR}/coqui/XTTS-v2"
    echo "    - Creer un compte HuggingFace et utiliser HF_TOKEN"
fi

echo "══════════════════════════════════════════════════════════════════"
echo ""
