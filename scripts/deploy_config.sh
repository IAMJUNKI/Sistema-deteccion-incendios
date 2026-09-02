#!/usr/bin/env bash
# Funciones compartidas para leer deploy/deploy.env sin eval.
#
# Formato admitido:
#   KEY=VALUE
#   export KEY=VALUE
#   # comentarios
#
# No se ejecutan sustituciones, comandos ni expresiones del fichero.

load_deploy_env() {
  local env_file="$1"
  local line clean_line key value

  [[ -f "$env_file" ]] || {
    echo "Error: no existe $env_file." >&2
    return 1
  }

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(printf '%s' "$line" | tr -d '\r')"
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    clean_line="$(printf '%s' "$line" | sed 's/^export //')"
    [[ "$clean_line" == *=* ]] || {
      echo "Error: línea no válida en $env_file." >&2
      return 1
    }

    key="$(printf '%s' "$clean_line" | cut -d= -f1)"
    value="$(printf '%s' "$clean_line" | cut -d= -f2-)"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
      echo "Error: nombre de variable no válido en $env_file: $key" >&2
      return 1
    }

    value="$(printf '%s' "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
}

