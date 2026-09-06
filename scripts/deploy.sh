#!/usr/bin/env bash
# Promueve un tag o commit en el servidor mediante SSH.
#
# Uso:
#   cp deploy/deploy.env.example deploy/deploy.env
#   # editar deploy/deploy.env
#   make deploy
#
# El servidor debe haber sido inicializado y debe tener instalada la
# Deploy Key de GitHub junto con /srv/fire-risk/bin/deploy_code_server.sh.

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/deploy/deploy.env"
CONFIG_LIB="$SCRIPT_DIR/deploy_config.sh"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Uso:
  scripts/deploy.sh [opciones]

Opciones:
  --dry-run          Mostrar el comando remoto sin ejecutarlo
  --no-tests         No ejecutar pytest en la release remota
  --no-restart       No reiniciar Streamlit tras el despliegue
  --no-check         No ejecutar el smoke test remoto
  --require-output   Exigir que exista un output operativo válido
  -h, --help         Mostrar esta ayuda
EOF
}

[[ -f "$CONFIG_LIB" ]] || die "Falta $CONFIG_LIB."
# shellcheck source=/dev/null
source "$CONFIG_LIB"
load_deploy_env "$ENV_FILE"

: "${DEPLOY_HOST:=}"
: "${DEPLOY_PORT:=22}"
: "${DEPLOY_USER:=deploy}"
: "${DEPLOY_PATH:=/srv/fire-risk}"
: "${DEPLOY_SERVICE_USER:=fire-risk}"
: "${DEPLOY_REF:=}"
: "${DEPLOY_BRANCH:=}"
: "${DEPLOY_SSH_TARGET:=}"
: "${DEPLOY_SSH_KEY:=}"
: "${DEPLOY_KNOWN_HOSTS:=}"
: "${DEPLOY_RUN_TESTS:=true}"
: "${DEPLOY_RESTART_DASHBOARD:=true}"
: "${DEPLOY_RUN_CHECK:=true}"
: "${DEPLOY_REQUIRE_OUTPUT:=false}"
: "${DEPLOY_USE_SUDO:=true}"
: "${DEPLOY_DRY_RUN:=false}"

# Las opciones de línea de comandos prevalecen sobre deploy.env.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DEPLOY_DRY_RUN=true
      shift
      ;;
    --no-tests)
      DEPLOY_RUN_TESTS=false
      shift
      ;;
    --no-restart)
      DEPLOY_RESTART_DASHBOARD=false
      shift
      ;;
    --no-check)
      DEPLOY_RUN_CHECK=false
      shift
      ;;
    --require-output)
      DEPLOY_REQUIRE_OUTPUT=true
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

if [[ -z "$DEPLOY_REF" ]]; then
  DEPLOY_REF="$DEPLOY_BRANCH"
fi

[[ -n "$DEPLOY_REF" ]] || die "Define DEPLOY_REF o DEPLOY_BRANCH en $ENV_FILE."
[[ "$DEPLOY_REF" =~ ^[A-Za-z0-9._/-]+$ ]] ||
  die "DEPLOY_REF contiene caracteres no permitidos."
[[ "$DEPLOY_REF" != -* && "$DEPLOY_REF" != *..* ]] ||
  die "DEPLOY_REF no puede empezar por '-' ni contener '..'."
[[ "$DEPLOY_PORT" =~ ^[0-9]+$ ]] || die "DEPLOY_PORT no es válido."
[[ "$DEPLOY_PATH" =~ ^/[A-Za-z0-9._/-]+$ && "$DEPLOY_PATH" != *..* ]] ||
  die "DEPLOY_PATH no es una ruta absoluta segura."
[[ "$DEPLOY_SERVICE_USER" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] ||
  die "DEPLOY_SERVICE_USER no es válido."
[[ -z "$DEPLOY_HOST" || "$DEPLOY_HOST" =~ ^[A-Za-z0-9._:-]+$ ]] ||
  die "DEPLOY_HOST no es válido."
[[ -z "$DEPLOY_SSH_TARGET" || "$DEPLOY_SSH_TARGET" =~ ^[A-Za-z0-9._:@/-]+$ ]] ||
  die "DEPLOY_SSH_TARGET no es válido."
[[ -z "$DEPLOY_SSH_KEY" || -f "$DEPLOY_SSH_KEY" ]] ||
  die "No existe DEPLOY_SSH_KEY: $DEPLOY_SSH_KEY."
[[ -z "$DEPLOY_KNOWN_HOSTS" || -f "$DEPLOY_KNOWN_HOSTS" ]] ||
  die "No existe DEPLOY_KNOWN_HOSTS: $DEPLOY_KNOWN_HOSTS."

if [[ -n "$DEPLOY_SSH_TARGET" ]]; then
  SSH_TARGET="$DEPLOY_SSH_TARGET"
else
  [[ -n "$DEPLOY_HOST" ]] || die "Define DEPLOY_HOST o DEPLOY_SSH_TARGET."
  SSH_TARGET="$DEPLOY_USER@$DEPLOY_HOST"
fi

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
[[ -n "$DEPLOY_SSH_KEY" ]] &&
  SSH_OPTS+=( -i "$DEPLOY_SSH_KEY" -o IdentitiesOnly=yes )
[[ -n "$DEPLOY_KNOWN_HOSTS" ]] &&
  SSH_OPTS+=( -o UserKnownHostsFile="$DEPLOY_KNOWN_HOSTS" )
[[ -n "$DEPLOY_PORT" ]] && SSH_OPTS+=( -p "$DEPLOY_PORT" )

if [[ "$DEPLOY_USE_SUDO" == "true" ]]; then
  REMOTE_DEPLOY="sudo -n -u $DEPLOY_SERVICE_USER $DEPLOY_PATH/bin/deploy_code_server.sh"
else
  REMOTE_DEPLOY="$DEPLOY_PATH/bin/deploy_code_server.sh"
fi

REMOTE_COMMAND="$REMOTE_DEPLOY --base-dir $DEPLOY_PATH --ref $DEPLOY_REF"
[[ "$DEPLOY_RUN_TESTS" == "true" ]] && REMOTE_COMMAND="$REMOTE_COMMAND --run-tests"
if [[ "$DEPLOY_USE_SUDO" == "true" ]]; then
  # Las unidades deben instalarse desde la release recién activada. Esto evita
  # que el restart use una unidad inexistente o una plantilla de una release
  # anterior, y mantiene PYTHONPATH alineado con el código desplegado.
  REMOTE_UNITS="$DEPLOY_PATH/app/scripts/install_systemd_units.sh"
  REMOTE_COMMAND="$REMOTE_COMMAND && sudo -n bash $REMOTE_UNITS"
fi
if [[ "$DEPLOY_RESTART_DASHBOARD" == "true" ]]; then
  if [[ "$DEPLOY_USE_SUDO" == "true" ]]; then
    REMOTE_COMMAND="$REMOTE_COMMAND && sudo -n systemctl restart fire-risk-dashboard.service"
  else
    REMOTE_COMMAND="$REMOTE_COMMAND && systemctl restart fire-risk-dashboard.service"
  fi
fi

if [[ "$DEPLOY_RUN_CHECK" == "true" ]]; then
  REMOTE_CHECK="$DEPLOY_PATH/app/scripts/check_server_deployment.sh"
  if [[ "$DEPLOY_USE_SUDO" == "true" ]]; then
    REMOTE_CHECK="sudo -n -u $DEPLOY_SERVICE_USER $REMOTE_CHECK"
  fi
  [[ "$DEPLOY_REQUIRE_OUTPUT" == "true" ]] &&
    REMOTE_CHECK="$REMOTE_CHECK --require-output"
  REMOTE_COMMAND="$REMOTE_COMMAND && $REMOTE_CHECK"
fi

echo "Desplegando ref $DEPLOY_REF en $SSH_TARGET:$DEPLOY_PATH"
if [[ "$DEPLOY_DRY_RUN" == "true" ]]; then
  echo "DRY-RUN: no se ejecutará el servidor remoto."
  echo "DRY-RUN: $REMOTE_COMMAND"
  exit 0
fi

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$REMOTE_COMMAND"
echo "Deployment completado correctamente."
