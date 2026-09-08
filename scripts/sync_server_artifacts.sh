#!/usr/bin/env bash
# Sincroniza artefactos pesados desde la máquina de operación al servidor.
#
# La sincronización no usa la Deploy Key de GitHub. Usa una clave SSH separada
# para acceder al servidor destino.
#
# Ejemplo:
#   scripts/sync_server_artifacts.sh \
#     --host fire-risk.example.org \
#     --remote-user deploy \
#     --ssh-key ~/.ssh/fire-risk-server \
#     --sudo

set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_LIB="$SCRIPT_DIR/deploy_config.sh"
DEPLOY_ENV_FILE="$REPO_ROOT/deploy/deploy.env"
DEPLOY_HOST=""
DEPLOY_USER=""
DEPLOY_PORT=""
DEPLOY_SSH_KEY=""
DEPLOY_KNOWN_HOSTS=""
DEPLOY_REMOTE_BASE=""
DEPLOY_USE_SUDO=""
if [[ -f "$CONFIG_LIB" && -f "$DEPLOY_ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG_LIB"
  load_deploy_env "$DEPLOY_ENV_FILE"
fi
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_BASE="/srv/fire-risk/data"
SSH_KEY=""
KNOWN_HOSTS=""
SSH_PORT=""
REMOTE_SUDO=0
DATASET_DIR="$REPO_ROOT/data/external/egif"
MODELS_DIR="$REPO_ROOT/data/models"
GRID_PATH="$REPO_ROOT/data/processed/grid/galicia_grid_1km_egif.parquet"
SYNC_DATASET=1
SYNC_MODELS=1
SYNC_GRID=1
DRY_RUN=0
DELETE_REMOTE=0
CHECKSUM=0

REMOTE_HOST="$DEPLOY_HOST"
REMOTE_USER="$DEPLOY_USER"
[[ -n "$DEPLOY_REMOTE_BASE" ]] && REMOTE_BASE="$DEPLOY_REMOTE_BASE"
SSH_KEY="$DEPLOY_SSH_KEY"
KNOWN_HOSTS="$DEPLOY_KNOWN_HOSTS"
SSH_PORT="$DEPLOY_PORT"
[[ "$DEPLOY_USE_SUDO" == "true" ]] && REMOTE_SUDO=1

usage() {
  cat <<'EOF'
Uso:
  sync_server_artifacts.sh --host HOST --remote-user USER [opciones]

También puede leer los valores equivalentes de deploy/deploy.env. En ese caso
se puede ejecutar mediante make deploy-data.

Opciones:
  --host HOST              Servidor destino (obligatorio)
  --remote-user USER       Usuario SSH del servidor (obligatorio)
  --remote-base PATH       Raíz de datos (por defecto: /srv/fire-risk/data)
  --ssh-key PATH            Clave SSH máquina local -> servidor
  --known-hosts PATH        Fichero known_hosts explícito
  --port PORT               Puerto SSH
  --sudo                    Usar sudo -n en mkdir y rsync remotos
  --dataset-dir PATH        Directorio EGIF local
  --models-dir PATH         Directorio de modelos local
  --grid PATH               Rejilla EGIF local
  --no-dataset              No sincronizar datasets/datacubo
  --no-models               No sincronizar modelos
  --no-grid                 No sincronizar la rejilla
  --delete                  Reflejar borrados (requiere confirmación explícita)
  --checksum                Comparar por checksum, más lento
  --dry-run                 Mostrar cambios sin subir archivos
  -h, --help                Mostrar esta ayuda
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || die "Falta el valor de --host."
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-user)
      [[ $# -ge 2 ]] || die "Falta el valor de --remote-user."
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-base)
      [[ $# -ge 2 ]] || die "Falta el valor de --remote-base."
      REMOTE_BASE="$2"
      shift 2
      ;;
    --ssh-key)
      [[ $# -ge 2 ]] || die "Falta el valor de --ssh-key."
      SSH_KEY="$2"
      shift 2
      ;;
    --known-hosts)
      [[ $# -ge 2 ]] || die "Falta el valor de --known-hosts."
      KNOWN_HOSTS="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "Falta el valor de --port."
      SSH_PORT="$2"
      shift 2
      ;;
    --sudo)
      REMOTE_SUDO=1
      shift
      ;;
    --dataset-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --dataset-dir."
      DATASET_DIR="$2"
      shift 2
      ;;
    --models-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --models-dir."
      MODELS_DIR="$2"
      shift 2
      ;;
    --grid)
      [[ $# -ge 2 ]] || die "Falta el valor de --grid."
      GRID_PATH="$2"
      shift 2
      ;;
    --no-dataset)
      SYNC_DATASET=0
      shift
      ;;
    --no-models)
      SYNC_MODELS=0
      shift
      ;;
    --no-grid)
      SYNC_GRID=0
      shift
      ;;
    --delete)
      DELETE_REMOTE=1
      shift
      ;;
    --checksum)
      CHECKSUM=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

[[ -n "$REMOTE_HOST" ]] || die "--host es obligatorio."
[[ -n "$REMOTE_USER" ]] || die "--remote-user es obligatorio."
[[ "$REMOTE_HOST" =~ ^[A-Za-z0-9._:-]+$ ]] || die "Host no válido."
[[ "$REMOTE_USER" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || die "Usuario remoto no válido."
[[ "$REMOTE_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "Ruta remota no válida."
[[ "$REMOTE_BASE" != *..* && "$REMOTE_BASE" != "//"* ]] ||
  die "La ruta remota no puede contener '..' ni comenzar con '//'."
[[ "$SYNC_DATASET" -eq 1 || "$SYNC_MODELS" -eq 1 || "$SYNC_GRID" -eq 1 ]] ||
  die "No queda ningún componente que sincronizar."
[[ -z "$SSH_KEY" || -f "$SSH_KEY" ]] || die "No existe la clave SSH: $SSH_KEY."
[[ -z "$KNOWN_HOSTS" || -f "$KNOWN_HOSTS" ]] ||
  die "No existe known_hosts: $KNOWN_HOSTS."
[[ -z "$SSH_PORT" || "$SSH_PORT" =~ ^[0-9]+$ ]] || die "Puerto no válido."

if [[ "$SYNC_DATASET" -eq 1 ]]; then
  [[ -d "$DATASET_DIR" ]] || die "No existe --dataset-dir: $DATASET_DIR."
fi
if [[ "$SYNC_MODELS" -eq 1 ]]; then
  [[ -d "$MODELS_DIR" ]] || die "No existe --models-dir: $MODELS_DIR."
fi
if [[ "$SYNC_GRID" -eq 1 ]]; then
  [[ -f "$GRID_PATH" ]] || die "No existe --grid: $GRID_PATH."
fi

command -v ssh >/dev/null 2>&1 || die "No se encuentra ssh."
command -v rsync >/dev/null 2>&1 || die "No se encuentra rsync."

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=( -i "$SSH_KEY" -o IdentitiesOnly=yes )
[[ -n "$KNOWN_HOSTS" ]] && SSH_OPTS+=( -o UserKnownHostsFile="$KNOWN_HOSTS" )
[[ -n "$SSH_PORT" ]] && SSH_OPTS+=( -p "$SSH_PORT" )
TARGET="$REMOTE_USER@$REMOTE_HOST"
RSYNC_RSH="$(printf '%q ' ssh "${SSH_OPTS[@]}")"

if [[ "$REMOTE_SUDO" -eq 1 ]]; then
  REMOTE_MKDIR="sudo -n /usr/bin/mkdir -p"
  REMOTE_RSYNC_PATH="sudo -n /usr/bin/rsync"
else
  REMOTE_MKDIR="/usr/bin/mkdir -p"
  REMOTE_RSYNC_PATH="rsync"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  ssh "${SSH_OPTS[@]}" "$TARGET" \
    "$REMOTE_MKDIR '$REMOTE_BASE/external/egif' '$REMOTE_BASE/models' '$REMOTE_BASE/processed/grid'"
fi

# Mantener modos y timestamps, pero no propagar UID/GID del ordenador local.
# Es especialmente importante cuando el rsync remoto se ejecuta mediante sudo.
# Los parciales se guardan en un directorio oculto y las actualizaciones se
# publican al final. Así, una interrupción no deja un Parquet incompleto con
# el nombre definitivo que pudiera ser leído por la inferencia.
RSYNC_ARGS=(
  -a
  --no-owner
  --no-group
  --human-readable
  --partial
  --partial-dir=.rsync-partial
  --delay-updates
  --progress
  --itemize-changes
)
[[ "$DELETE_REMOTE" -eq 1 ]] && RSYNC_ARGS+=(--delete)
[[ "$CHECKSUM" -eq 1 ]] && RSYNC_ARGS+=(--checksum)
[[ "$DRY_RUN" -eq 1 ]] && RSYNC_ARGS+=(--dry-run)

sync_directory() {
  local source_dir="$1"
  local remote_dir="$2"
  echo "Sincronizando $source_dir -> $TARGET:$remote_dir"
  rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" \
    --rsync-path="$REMOTE_RSYNC_PATH" \
    "$source_dir/" "$TARGET:$remote_dir/"
}

sync_file() {
  local source_file="$1"
  local remote_dir="$2"
  echo "Sincronizando $source_file -> $TARGET:$remote_dir"
  rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" \
    --rsync-path="$REMOTE_RSYNC_PATH" \
    "$source_file" "$TARGET:$remote_dir/"
}

[[ "$SYNC_DATASET" -eq 0 ]] || \
  sync_directory "$DATASET_DIR" "$REMOTE_BASE/external/egif"
[[ "$SYNC_MODELS" -eq 0 ]] || \
  sync_directory "$MODELS_DIR" "$REMOTE_BASE/models"
[[ "$SYNC_GRID" -eq 0 ]] || \
  sync_file "$GRID_PATH" "$REMOTE_BASE/processed/grid"

if ls data/external/*.geojson >/dev/null 2>&1; then
  echo "Sincronizando capas vectoriales data/external/*.geojson -> $TARGET:$REMOTE_BASE/external/"
  rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" \
    --rsync-path="$REMOTE_RSYNC_PATH" \
    data/external/*.geojson "$TARGET:$REMOTE_BASE/external/"
fi

echo "Sincronización finalizada."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Modo dry-run: no se modificó el servidor."
fi
