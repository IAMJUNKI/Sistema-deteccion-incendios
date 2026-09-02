#!/usr/bin/env bash
# Smoke test de una instalación del servidor.
#
# No descarga ni modifica datos. Para validar también el último resultado:
#   sudo -u fire-risk /srv/fire-risk/app/scripts/check_server_deployment.sh \
#     --require-output

set -Eeuo pipefail

BASE_DIR="/srv/fire-risk"
PYTHON_BIN="/opt/miniconda3/envs/incendios-forestales/bin/python"
REQUIRE_OUTPUT=0
ALLOW_STALE=0

usage() {
  cat <<'EOF'
Uso:
  check_server_deployment.sh [opciones]

Opciones:
  --base-dir PATH     Raíz del servidor
  --python PATH       Python del entorno
  --require-output    Ejecutar check_operational_run.py
  --allow-stale       Permitir stale en la validación del output
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --base-dir."
      BASE_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "Falta el valor de --python."
      PYTHON_BIN="$2"
      shift 2
      ;;
    --require-output)
      REQUIRE_OUTPUT=1
      shift
      ;;
    --allow-stale)
      ALLOW_STALE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Opción desconocida: $1"
      ;;
  esac
done

CURRENT_LINK="$BASE_DIR/app"
DATA_DIR="$BASE_DIR/data"
ENV_FILE="$BASE_DIR/config/.env"

[[ -L "$CURRENT_LINK" ]] || die "$CURRENT_LINK no es un enlace simbólico."
CURRENT_DIR="$(readlink -f "$CURRENT_LINK")"
[[ -d "$CURRENT_DIR" ]] || die "La release activa no existe: $CURRENT_DIR."
[[ -f "$CURRENT_DIR/app.py" ]] || die "Falta app.py en la release activa."
[[ -f "$CURRENT_DIR/scripts/run_daily_inference.py" ]] ||
  die "Falta run_daily_inference.py en la release activa."
[[ -L "$CURRENT_DIR/data" ]] || die "La release activa no enlaza data persistente."
[[ "$(readlink -f "$CURRENT_DIR/data")" == "$(readlink -f "$DATA_DIR")" ]] ||
  die "El enlace data no apunta al almacenamiento esperado."
[[ -r "$ENV_FILE" ]] || die "Falta la configuración $ENV_FILE."
[[ -x "$PYTHON_BIN" ]] || die "No existe el Python: $PYTHON_BIN."

if command -v stat >/dev/null 2>&1; then
  env_mode="$(stat -c '%a' "$ENV_FILE" 2>/dev/null || true)"
  [[ -z "$env_mode" || "$env_mode" == "600" ]] ||
    die "$ENV_FILE debe tener permisos 600; tiene $env_mode."
fi

(
  cd "$CURRENT_DIR"
  PYTHONPATH=. "$PYTHON_BIN" -c \
    "import pandas, pyarrow, geopandas, lightgbm, streamlit; print('Dependencias Python: OK')"
)

if [[ "$REQUIRE_OUTPUT" -eq 1 ]]; then
  check_args=()
  [[ "$ALLOW_STALE" -eq 1 ]] && check_args+=(--allow-stale)
  (
    cd "$CURRENT_DIR"
    PYTHONPATH=. "$PYTHON_BIN" scripts/check_operational_run.py "${check_args[@]}"
  )
fi

echo "Instalación del servidor: OK"
echo "Release activa: $CURRENT_DIR"

