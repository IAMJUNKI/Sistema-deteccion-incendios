#!/usr/bin/env bash
# Inicializa un servidor para despliegues pull desde GitHub.
#
# Ejecutar como root, una sola vez:
#   sudo bash scripts/bootstrap_server_deployment.sh \
#     --repo git@github.com:ORGANIZACION/REPOSITORIO.git
#
# Este script genera la Deploy Key en el servidor. Nunca imprime la clave
# privada; sólo muestra la clave pública que debe registrarse en GitHub.

set -Eeuo pipefail
umask 077

SERVICE_USER="fire-risk"
BASE_DIR="/srv/fire-risk"
GITHUB_HOST="github.com"
REPO_URL=""

usage() {
  cat <<'EOF'
Uso:
  bootstrap_server_deployment.sh --repo git@github.com:ORG/REPO.git [opciones]

Opciones:
  --repo URL             URL SSH del repositorio GitHub (obligatoria)
  --service-user NOMBRE  Usuario de servicio (por defecto: fire-risk)
  --base-dir PATH        Raíz persistente (por defecto: /srv/fire-risk)
  --github-host HOST     Host GitHub (por defecto: github.com)
  -h, --help             Mostrar esta ayuda
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || die "Falta el valor de --repo."
      REPO_URL="$2"
      shift 2
      ;;
    --service-user)
      [[ $# -ge 2 ]] || die "Falta el valor de --service-user."
      SERVICE_USER="$2"
      shift 2
      ;;
    --base-dir)
      [[ $# -ge 2 ]] || die "Falta el valor de --base-dir."
      BASE_DIR="$2"
      shift 2
      ;;
    --github-host)
      [[ $# -ge 2 ]] || die "Falta el valor de --github-host."
      GITHUB_HOST="$2"
      shift 2
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
[[ -n "$REPO_URL" ]] || die "--repo es obligatorio."
[[ "$BASE_DIR" == /* && "$BASE_DIR" != "/" ]] || die "--base-dir debe ser una ruta absoluta segura."
[[ "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "Nombre de usuario no válido."
[[ "$GITHUB_HOST" =~ ^[A-Za-z0-9.-]+$ ]] || die "Host GitHub no válido."
[[ "$REPO_URL" != *[[:space:]]* ]] || die "La URL del repositorio no puede contener espacios."
case "$REPO_URL" in
  git@"$GITHUB_HOST":*|ssh://git@"$GITHUB_HOST"/*) ;;
  *) die "La URL debe usar SSH contra $GITHUB_HOST, por ejemplo git@$GITHUB_HOST:ORG/REPO.git." ;;
esac

command -v git >/dev/null 2>&1 || die "No se encuentra git."
command -v ssh-keygen >/dev/null 2>&1 || die "No se encuentra ssh-keygen."
command -v ssh-keyscan >/dev/null 2>&1 || die "No se encuentra ssh-keyscan."
command -v runuser >/dev/null 2>&1 || die "No se encuentra runuser."

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "$BASE_DIR" \
    --shell /usr/sbin/nologin "$SERVICE_USER"
fi

SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$BASE_DIR"
for directory in \
  "$BASE_DIR/bin" \
  "$BASE_DIR/config" \
  "$BASE_DIR/releases" \
  "$BASE_DIR/incoming" \
  "$BASE_DIR/data/raw" \
  "$BASE_DIR/data/processed" \
  "$BASE_DIR/data/processed/grid" \
  "$BASE_DIR/data/processed/state" \
  "$BASE_DIR/data/processed/observations" \
  "$BASE_DIR/data/processed/tabular/egif" \
  "$BASE_DIR/data/models" \
  "$BASE_DIR/data/external/egif"
do
  install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$directory"
done

SSH_DIR="$BASE_DIR/.ssh"
SSH_KEY="$SSH_DIR/github_deploy_ed25519"
PUBLIC_KEY="$SSH_KEY.pub"
KNOWN_HOSTS="$SSH_DIR/known_hosts"
SSH_CONFIG="$SSH_DIR/config"
REPO_FILE="$BASE_DIR/config/repository_url"
ENV_FILE="$BASE_DIR/config/.env"

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0700 "$SSH_DIR"

if [[ ! -e "$SSH_KEY" ]]; then
  runuser -u "$SERVICE_USER" -- ssh-keygen -q -t ed25519 \
    -C "$SERVICE_USER@$GITHUB_HOST-deploy" \
    -f "$SSH_KEY" -N ""
elif [[ ! -s "$PUBLIC_KEY" ]]; then
  runuser -u "$SERVICE_USER" -- sh -c \
    "ssh-keygen -y -f '$SSH_KEY' > '$PUBLIC_KEY'"
fi
chown "$SERVICE_USER:$SERVICE_GROUP" "$SSH_KEY" "$PUBLIC_KEY"
chmod 0600 "$SSH_KEY"
chmod 0644 "$PUBLIC_KEY"

if [[ ! -s "$KNOWN_HOSTS" ]]; then
  known_hosts_tmp="$(mktemp)"
  trap 'rm -f "$known_hosts_tmp"' EXIT
  ssh-keyscan -T 10 -t ed25519 "$GITHUB_HOST" > "$known_hosts_tmp" 2>/dev/null \
    || die "No se pudo obtener la clave pública de $GITHUB_HOST."
  [[ -s "$known_hosts_tmp" ]] || die "La respuesta de ssh-keyscan está vacía."
  install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 \
    "$known_hosts_tmp" "$KNOWN_HOSTS"
  rm -f "$known_hosts_tmp"
  trap - EXIT
fi

ssh_config_tmp="$(mktemp)"
trap 'rm -f "$ssh_config_tmp"' EXIT
{
  printf '%s\n' "Host $GITHUB_HOST"
  printf '%s\n' "  HostName $GITHUB_HOST"
  printf '%s\n' "  User git"
  printf '%s\n' "  IdentityFile $SSH_KEY"
  printf '%s\n' "  IdentitiesOnly yes"
  printf '%s\n' "  UserKnownHostsFile $KNOWN_HOSTS"
  printf '%s\n' "  StrictHostKeyChecking yes"
  printf '%s\n' "  BatchMode yes"
} > "$ssh_config_tmp"
install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 \
  "$ssh_config_tmp" "$SSH_CONFIG"
rm -f "$ssh_config_tmp"
trap - EXIT

printf '%s\n' "$REPO_URL" > "$REPO_FILE"
chown "$SERVICE_USER:$SERVICE_GROUP" "$REPO_FILE"
chmod 0600 "$REPO_FILE"

if [[ ! -e "$ENV_FILE" ]]; then
  env_tmp="$(mktemp)"
  trap 'rm -f "$env_tmp"' EXIT
  cat > "$env_tmp" <<'EOF'
# Configuración de producción. Completar antes de activar los servicios.
PIPELINE_ENVIRONMENT=production
LOCAL_SIMULATION_MODE=false
FORECAST_PROVIDER=meteogalicia
# Añadir aquí las claves y rutas operativas.
EOF
  install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0600 \
    "$env_tmp" "$ENV_FILE"
  rm -f "$env_tmp"
  trap - EXIT
fi

echo "Servidor inicializado en: $BASE_DIR"
echo "Deploy Key pública (registrarla en GitHub como read-only):"
cat "$PUBLIC_KEY"
echo
echo "Siguiente paso: añadir esa clave en GitHub > Settings > Deploy keys."
echo "La clave privada queda en $SSH_KEY y no debe copiarse ni versionarse."
echo "Verifica también la huella de $GITHUB_HOST antes de confiar en known_hosts."

