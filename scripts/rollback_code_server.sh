#!/usr/bin/env bash
# Cambia el enlace app a una release ya descargada.
#
#   sudo -u fire-risk /srv/fire-risk/bin/rollback_code_server.sh \
#     --release 20260901T050000Z_a1b2c3d4e5f6 --restart-dashboard

set -Eeuo pipefail
umask 027

BASE_DIR="/srv/fire-risk"
CURRENT_LINK="$BASE_DIR/app"
RELEASES_DIR="$BASE_DIR/releases"
RELEASE=""
RESTART_DASHBOARD=0

usage() {
  cat <<'EOF'
Uso:
  rollback_code_server.sh --release NOMBRE_RELEASE [opciones]

Opciones:
  --release PATH_O_NOMBRE  Release dentro de /srv/fire-risk/releases
  --base-dir PATH          Raíz del servidor
  --restart-dashboard      Reiniciar Streamlit
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      [[ $# -ge 2 ]] || die "Falta el valor de --release."
      RELEASE="$2"
      shift 2
      ;;
    --base-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --base-dir."
      BASE_DIR="$2"
      CURRENT_LINK="$BASE_DIR/app"
      RELEASES_DIR="$BASE_DIR/releases"
      shift 2
      ;;
    --restart-dashboard)
      RESTART_DASHBOARD=1
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

[[ -n "$RELEASE" ]] || die "--release es obligatorio."
[[ "$(id -u)" -ne 0 ]] || die "Ejecuta el rollback como fire-risk, no como root."
[[ "$RELEASE" != *[[:space:]]* && "$RELEASE" != *..* ]] || die "Ruta no válida."

if [[ "$RELEASE" == /* ]]; then
  TARGET="$RELEASE"
else
  TARGET="$RELEASES_DIR/$RELEASE"
fi

case "$TARGET" in
  "$RELEASES_DIR"/*) ;;
  *) die "La release debe estar dentro de $RELEASES_DIR." ;;
esac
[[ -d "$TARGET" && ! -L "$TARGET" ]] || die "Release inexistente o no válida: $TARGET."
[[ -f "$TARGET/app.py" ]] || die "La release no contiene app.py."
[[ -f "$TARGET/scripts/run_daily_inference.py" ]] ||
  die "La release no contiene el pipeline."

if [[ -e "$CURRENT_LINK" && ! -L "$CURRENT_LINK" ]]; then
  die "$CURRENT_LINK existe y no es un enlace simbólico."
fi

NEXT_LINK="$BASE_DIR/.app.rollback.next.$$"
rm -f "$NEXT_LINK"
ln -s "$TARGET" "$NEXT_LINK"
mv -Tf "$NEXT_LINK" "$CURRENT_LINK"

CONFIG_DIR="$BASE_DIR/config"
manifest_tmp="$CONFIG_DIR/current_release.json.tmp.$$"
{
  printf '%s\n' '{'
  printf '  "ref": "rollback",\n'
  printf '  "commit": "unknown",\n'
  printf '  "release_dir": "%s",\n' "$TARGET"
  printf '  "deployed_at": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' '}'
} > "$manifest_tmp"
mv -f "$manifest_tmp" "$CONFIG_DIR/current_release.json"

if [[ "$RESTART_DASHBOARD" -eq 1 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo -n systemctl restart fire-risk-dashboard.service
  else
    die "No se puede reiniciar Streamlit: falta sudo sin contraseña."
  fi
fi

echo "Rollback activo: $TARGET"

