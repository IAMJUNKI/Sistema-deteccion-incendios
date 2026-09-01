#!/usr/bin/env bash
# Instala las unidades systemd versionadas en deploy/systemd.
#
# Ejecutar desde el checkout activo como root:
#   sudo bash scripts/install_systemd_units.sh --enable --start

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENABLE_UNITS=0
START_UNITS=0

usage() {
  cat <<'EOF'
Uso:
  install_systemd_units.sh [--enable] [--start]

Opciones:
  --enable   Habilitar dashboard y timers al arrancar
  --start    Arrancar dashboard y timers ahora
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable)
      ENABLE_UNITS=1
      shift
      ;;
    --start)
      START_UNITS=1
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

[[ "$(id -u)" -eq 0 ]] || die "Debe ejecutarse como root."
command -v systemctl >/dev/null 2>&1 || die "No se encuentra systemctl."

for unit in \
  fire-risk-dashboard.service \
  fire-risk-inference.service \
  fire-risk-inference.timer \
  fire-risk-health.service \
  fire-risk-health.timer
do
  [[ -f "$PROJECT_ROOT/deploy/systemd/$unit" ]] ||
    die "Falta la plantilla $PROJECT_ROOT/deploy/systemd/$unit."
  install -o root -g root -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/$unit" "/etc/systemd/system/$unit"
done

systemctl daemon-reload

if [[ "$ENABLE_UNITS" -eq 1 ]]; then
  systemctl enable fire-risk-dashboard.service
  systemctl enable fire-risk-inference.timer
  systemctl enable fire-risk-health.timer
fi

if [[ "$START_UNITS" -eq 1 ]]; then
  systemctl start fire-risk-dashboard.service
  systemctl start fire-risk-inference.timer
  systemctl start fire-risk-health.timer
fi

echo "Unidades instaladas. Verificar con:"
echo "  systemctl status fire-risk-dashboard.service"
echo "  systemctl list-timers fire-risk-inference.timer fire-risk-health.timer"

