#!/usr/bin/env bash
# Despliega un commit o tag desde GitHub en una release inmutable.
#
# Ejecutar como el usuario de servicio:
#   sudo -u fire-risk /srv/fire-risk/bin/deploy_code_server.sh \
#     --ref v0.3.0 --run-tests --restart-dashboard

set -Eeuo pipefail
umask 027

BASE_DIR="/srv/fire-risk"
CURRENT_LINK="$BASE_DIR/app"
RELEASES_DIR="$BASE_DIR/releases"
CONFIG_DIR="$BASE_DIR/config"
DATA_DIR="$BASE_DIR/data"
REPO_FILE="$CONFIG_DIR/repository_url"
SSH_CONFIG="$BASE_DIR/.ssh/config"
PYTHON_BIN="/opt/miniconda3/envs/incendios-forestales/bin/python"
REF=""
RUN_TESTS=0
RESTART_DASHBOARD=0
RELEASE_TMP=""

usage() {
  cat <<'EOF'
Uso:
  deploy_code_server.sh --ref TAG_O_COMMIT [opciones]

Opciones:
  --ref REF              Tag, rama o commit a desplegar (obligatorio)
  --python PATH          Python del entorno de producción
  --base-dir PATH        Raíz del servidor (por defecto: /srv/fire-risk)
  --run-tests            Ejecutar pytest -q antes de cambiar la release
  --restart-dashboard    Reiniciar Streamlit después del cambio
  -h, --help             Mostrar esta ayuda
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$RELEASE_TMP" && -d "$RELEASE_TMP" ]]; then
    rm -rf "$RELEASE_TMP"
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      [[ $# -ge 2 ]] || die "Falta el valor de --ref."
      REF="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "Falta el valor de --python."
      PYTHON_BIN="$2"
      shift 2
      ;;
    --base-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --base-dir."
      BASE_DIR="$2"
      CURRENT_LINK="$BASE_DIR/app"
      RELEASES_DIR="$BASE_DIR/releases"
      CONFIG_DIR="$BASE_DIR/config"
      DATA_DIR="$BASE_DIR/data"
      REPO_FILE="$CONFIG_DIR/repository_url"
      SSH_CONFIG="$BASE_DIR/.ssh/config"
      shift 2
      ;;
    --run-tests)
      RUN_TESTS=1
      shift
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

[[ -n "$REF" ]] || die "--ref es obligatorio; usa un tag o commit revisado."
[[ "$REF" =~ ^[A-Za-z0-9._/-]+$ ]] || die "REF contiene caracteres no permitidos."
[[ "$REF" != -* && "$REF" != *..* ]] || die "REF no puede empezar por '-' ni contener '..'."
[[ "$BASE_DIR" == /* && "$BASE_DIR" != "/" ]] || die "BASE_DIR no es segura."
[[ "$(id -u)" -ne 0 ]] || die "Ejecuta el despliegue como fire-risk, no como root."
[[ -r "$REPO_FILE" ]] || die "No existe $REPO_FILE; ejecuta el bootstrap."
[[ -r "$SSH_CONFIG" ]] || die "No existe $SSH_CONFIG; ejecuta el bootstrap."
[[ -d "$DATA_DIR" ]] || die "No existe el almacenamiento persistente $DATA_DIR."
[[ -x "$PYTHON_BIN" ]] || die "No existe el Python configurado: $PYTHON_BIN."
command -v git >/dev/null 2>&1 || die "No se encuentra git."

REPO_URL="$(tr -d '\r\n' < "$REPO_FILE")"
[[ -n "$REPO_URL" ]] || die "repository_url está vacío."
[[ "$REPO_URL" != *[[:space:]]* ]] || die "repository_url contiene espacios."

mkdir -p "$RELEASES_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_TMP="$RELEASES_DIR/.release-$timestamp-$$"

export GIT_SSH_COMMAND="ssh -F $SSH_CONFIG -o BatchMode=yes"
git init -q "$RELEASE_TMP"
git -C "$RELEASE_TMP" remote add origin "$REPO_URL"
echo "Descargando $REF desde GitHub..."
if [[ "$REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
  # GitHub no siempre expone un SHA como remote ref descargable. Primero se
  # trae la punta de main y sus tags; si el commit no está en ese shallow
  # history, se amplía el historial de main antes de resolverlo localmente.
  git -C "$RELEASE_TMP" fetch --depth=1 origin main --tags
  if ! git -C "$RELEASE_TMP" cat-file -e "$REF^{commit}" 2>/dev/null; then
    git -C "$RELEASE_TMP" fetch --unshallow origin main --tags
  fi
  git -C "$RELEASE_TMP" cat-file -e "$REF^{commit}" 2>/dev/null ||
    die "No se encontró el commit $REF en main ni en los tags descargados."
  git -C "$RELEASE_TMP" checkout --detach --quiet "$REF"
else
  git -C "$RELEASE_TMP" fetch --depth=1 origin "$REF"
  git -C "$RELEASE_TMP" checkout --detach --quiet FETCH_HEAD
fi

COMMIT="$(git -C "$RELEASE_TMP" rev-parse --verify HEAD)"
SHORT_COMMIT="$(git -C "$RELEASE_TMP" rev-parse --short=12 HEAD)"
RELEASE_DIR="$RELEASES_DIR/$timestamp"_"$SHORT_COMMIT"
[[ ! -e "$RELEASE_DIR" ]] || die "La release ya existe: $RELEASE_DIR"

for required_file in \
  "$RELEASE_TMP/environment.yml" \
  "$RELEASE_TMP/app.py" \
  "$RELEASE_TMP/scripts/run_daily_inference.py" \
  "$RELEASE_TMP/scripts/check_operational_run.py"
do
  [[ -f "$required_file" ]] || die "La release no contiene $required_file."
done

echo "Validando sintaxis Python..."
"$PYTHON_BIN" -m compileall -q \
  "$RELEASE_TMP/src" \
  "$RELEASE_TMP/scripts" \
  "$RELEASE_TMP/app.py"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "Ejecutando tests..."
  (
    cd "$RELEASE_TMP"
    PYTHONPATH=. "$PYTHON_BIN" -m pytest -q
  )
fi

# La release se ha descargado en una carpeta temporal nueva. El ``data`` que
# pueda traer Git (README, .gitkeep o metadatos pequeños) pertenece a la
# release, no al almacenamiento persistente. Se elimina sólo tras comprobar
# que la ruta sigue siendo una carpeta temporal bajo RELEASES_DIR; nunca se
# toca DATA_DIR.
if [[ -e "$RELEASE_TMP/data" || -L "$RELEASE_TMP/data" ]]; then
  [[ "$RELEASE_TMP" == "$RELEASES_DIR"/.release-* ]] ||
    die "Ruta temporal de release inesperada; se conserva data por seguridad."
  rm -rf -- "$RELEASE_TMP/data"
fi
ln -s "$DATA_DIR" "$RELEASE_TMP/data"

ENV_FILE="$CONFIG_DIR/.env"
if [[ -e "$ENV_FILE" ]]; then
  ln -s "$ENV_FILE" "$RELEASE_TMP/.env"
fi

mv "$RELEASE_TMP" "$RELEASE_DIR"
RELEASE_TMP=""

if [[ -e "$CURRENT_LINK" && ! -L "$CURRENT_LINK" ]]; then
  die "$CURRENT_LINK existe y no es un enlace simbólico; no se sobrescribe."
fi
NEXT_LINK="$BASE_DIR/.app.next.$$"
rm -f "$NEXT_LINK"
ln -s "$RELEASE_DIR" "$NEXT_LINK"
mv -Tf "$NEXT_LINK" "$CURRENT_LINK"

manifest_tmp="$CONFIG_DIR/current_release.json.tmp.$$"
{
  printf '%s\n' '{'
  printf '  "ref": "%s",\n' "$REF"
  printf '  "commit": "%s",\n' "$COMMIT"
  printf '  "release_dir": "%s",\n' "$RELEASE_DIR"
  printf '  "deployed_at": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' '}'
} > "$manifest_tmp"
mv -f "$manifest_tmp" "$CONFIG_DIR/current_release.json"

if [[ "$RESTART_DASHBOARD" -eq 1 ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    systemctl restart fire-risk-dashboard.service
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n systemctl restart fire-risk-dashboard.service
  else
    die "No se puede reiniciar Streamlit: falta sudo sin contraseña."
  fi
fi

echo "Release desplegada: $RELEASE_DIR"
echo "Commit: $COMMIT"
echo "Código activo: $CURRENT_LINK"
