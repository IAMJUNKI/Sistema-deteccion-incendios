# Despliegue de código y datos pesados

## 1. Objetivo

Este documento define el procedimiento reproducible para desplegar el proyecto en
un servidor Linux sin guardar secretos, datasets de varios gigabytes ni modelos
serializados en GitHub.

El flujo separa dos operaciones:

~~~text
GitHub ──(Deploy Key read-only)──> servidor
máquina de operación/CI ──(SSH + rsync)──> servidor
servidor ──(systemd)──> ingesta, inferencia, health check y dashboard
~~~

La aplicación se despliega como releases inmutables. Los datos permanecen en un
volumen persistente fuera de cada checkout.

## 2. Qué se versiona y qué se sincroniza

### Se versiona en GitHub

- código Python y scripts de operación;
- documentación;
- plantillas systemd;
- environment.yml y .env.example;
- tests;
- metadatos pequeños cuando sea útil para reproducibilidad.

### No se versiona en GitHub

- galicia_1km.nc;
- dataset_2019.parquet a dataset_2023.parquet;
- forecasts brutos y observaciones descargadas;
- modelos .joblib y sus salidas;
- .env y claves privadas.

Los datasets y el NetCDF del modelo EGIF se encuentran localmente en
data/external/egif/ y se transfieren con rsync. En el servidor se conservan en
data/external/egif/ para entrenamiento o auditoría. La inferencia diaria utiliza
principalmente la rejilla, el estado meteorológico, los forecasts archivados y
los modelos.

## 3. Modelo de seguridad

Hay que mantener separadas estas credenciales:

| Credencial | Dirección | Ubicación | Uso |
|---|---|---|---|
| Deploy Key de GitHub | servidor → GitHub | /srv/fire-risk/.ssh/github_deploy_ed25519 | Leer el repositorio |
| Clave de transferencia | máquina local/CI → servidor | sólo en la máquina de operación | rsync de datos y modelos |
| Claves API | proceso del servidor → proveedores | /srv/fire-risk/config/.env | MeteoGalicia, AEMET u otras fuentes |

La Deploy Key de GitHub es de sólo lectura y no se copia a la máquina local.
La clave de transferencia no se registra en GitHub como Deploy Key del
repositorio. Las API keys sólo se almacenan en el fichero .env del servidor con
permisos 600.

El usuario fire-risk se crea con shell nologin. Para rsync se recomienda utilizar
un usuario administrativo o de despliegue separado, con una clave propia y
permisos limitados sobre el almacenamiento de datos. Si ese usuario necesita
sudo, debe configurarse una regla revisada por el administrador y utilizarse
sudo -n para que un job no pueda quedar esperando una contraseña.

## 4. Estructura del servidor

La estructura estándar es:

~~~text
/srv/fire-risk/
├── app -> releases/20260901T051500Z_ab12cd34ef56
├── bin/
│   ├── deploy_code_server.sh
│   ├── rollback_code_server.sh
│   └── check_server_deployment.sh
├── config/
│   ├── repository_url
│   ├── .env
│   └── current_release.json
├── releases/
│   ├── 20260901T051500Z_ab12cd34ef56/
│   └── 20260903T051500Z_7890abcd1234/
├── data/
│   ├── external/egif/
│   ├── models/
│   ├── raw/
│   └── processed/
└── .ssh/
    ├── config
    ├── github_deploy_ed25519
    ├── github_deploy_ed25519.pub
    └── known_hosts
~~~

El enlace app cambia de forma atómica entre releases. El enlace data dentro de
cada release apunta a /srv/fire-risk/data. Por ello, cambiar el código no borra
datasets, modelos, estado meteorológico ni outputs.

## 5. Preparación inicial del servidor

Los pasos siguientes se ejecutan una sola vez con una cuenta administrativa.

### 5.1 Copiar el bootstrap sin clonar todavía el repositorio

Desde la raíz local del proyecto:

~~~bash
scp scripts/bootstrap_server_deployment.sh ADMIN@SERVIDOR:/tmp/
ssh ADMIN@SERVIDOR \
  'sudo bash /tmp/bootstrap_server_deployment.sh \
  --repo git@github.com:ORGANIZACION/Sistema-deteccion-incendios.git'
~~~

El script crea el usuario fire-risk, la estructura persistente, la pareja de
claves de GitHub, known_hosts, la configuración SSH y un fichero .env inicial
vacío o mínimo.

Si el repositorio tiene otro nombre, sustituir la URL por la URL SSH exacta que
muestra GitHub en Code > SSH.

Si se desea crear la clave manualmente, el procedimiento equivalente, ejecutado
en el servidor, es:

~~~bash
sudo -u fire-risk mkdir -p /srv/fire-risk/.ssh
sudo -u fire-risk chmod 700 /srv/fire-risk/.ssh
sudo -u fire-risk ssh-keygen -t ed25519 \
  -C fire-risk@github-deploy \
  -f /srv/fire-risk/.ssh/github_deploy_ed25519
sudo -u fire-risk chmod 600 /srv/fire-risk/.ssh/github_deploy_ed25519
sudo -u fire-risk cat /srv/fire-risk/.ssh/github_deploy_ed25519.pub
~~~

Sólo la última salida, que termina en .pub, se copia a GitHub. Nunca se copia
github_deploy_ed25519 sin la extensión .pub. El bootstrap automatiza estos
mismos pasos y es idempotente: no reemplaza una clave ya existente.

### 5.2 Registrar la Deploy Key en GitHub

El bootstrap muestra la clave pública, pero nunca la privada. Registrar la
salida completa de la clave pública en:

~~~text
GitHub
  → repositorio
  → Settings
  → Deploy keys
  → Add deploy key
  → Allow write access: desactivado
~~~

No marcar Allow write access. La clave sólo tiene que leer tags, ramas y
commits.

Comprobar la huella de la clave del repositorio en el servidor:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk ssh-keygen -lf \
  /srv/fire-risk/.ssh/github_deploy_ed25519.pub'
~~~

El bootstrap obtiene la clave de host de GitHub con ssh-keyscan si known_hosts
está vacío. La primera instalación debe contrastar esa huella con una fuente
independiente y no aceptar cambios inesperados posteriormente.

### 5.3 Comprobar acceso al repositorio

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk env GIT_SSH_COMMAND="ssh -F /srv/fire-risk/.ssh/config" \
  git ls-remote "$(cat /srv/fire-risk/config/repository_url)" HEAD'
~~~

La respuesta debe mostrar el commit HEAD. El mensaje de GitHub indicando que no
proporciona una shell interactiva es normal; lo importante es que ls-remote
funcione.

### 5.4 Instalar los scripts de operación

Antes de desplegar la primera release, copiar los scripts que permiten obtener
el resto del código:

~~~bash
scp scripts/deploy_code_server.sh \
    scripts/rollback_code_server.sh \
    scripts/check_server_deployment.sh \
    ADMIN@SERVIDOR:/tmp/

ssh ADMIN@SERVIDOR '
  sudo install -o fire-risk -g fire-risk -m 0750 \
    /tmp/deploy_code_server.sh /srv/fire-risk/bin/deploy_code_server.sh
  sudo install -o fire-risk -g fire-risk -m 0750 \
    /tmp/rollback_code_server.sh /srv/fire-risk/bin/rollback_code_server.sh
  sudo install -o fire-risk -g fire-risk -m 0750 \
    /tmp/check_server_deployment.sh /srv/fire-risk/bin/check_server_deployment.sh
'
~~~

El script de bootstrap no modifica un fichero .env existente. Completar
/srv/fire-risk/config/.env con las claves reales y las rutas absolutas de
producción, y asegurar:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo chown fire-risk:fire-risk /srv/fire-risk/config/.env &&
   sudo chmod 600 /srv/fire-risk/config/.env'
~~~

## 6. Primer despliegue de código

Instalar previamente el entorno Conda y dependencias según
servidor_produccion.md. Después desplegar un tag o commit revisado:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk /srv/fire-risk/bin/deploy_code_server.sh \
   --ref v0.3.0 --run-tests'
~~~

La opción --ref es obligatoria para evitar actualizaciones accidentales. Se
recomienda desplegar tags o hashes de commit, no una rama mutable.

El script:

1. lee la URL almacenada por el bootstrap;
2. usa la configuración SSH de fire-risk;
3. descarga el ref en una carpeta temporal;
4. comprueba que contiene el pipeline y el dashboard;
5. ejecuta compileall;
6. ejecuta pytest si se solicita;
7. enlaza el .env y el almacenamiento data;
8. mueve la carpeta temporal a releases;
9. cambia app sólo después de superar las validaciones;
10. escribe config/current_release.json.

Si falla una validación, se conserva la release activa anterior y se elimina sólo
la carpeta temporal de la operación fallida.

## 6.1 Atajo local con Make

Este repositorio es Python y no necesita Node para funcionar. El `Makefile` sólo
ofrece atajos para los scripts Bash; no instala dependencias de JavaScript, no
ejecuta `npm` y no necesita PM2.

Configurar una vez:

~~~bash
cp deploy/deploy.env.example deploy/deploy.env
# editar deploy/deploy.env con el host, usuario, ref y clave SSH local
~~~

Probar sin tocar el servidor:

~~~bash
make deploy-dry-run
~~~

Desplegar código:

~~~bash
make deploy
~~~

Sincronizar sólo datasets, modelos y rejilla:

~~~bash
make deploy-data
~~~

Ejecutar ambos pasos:

~~~bash
make deploy-all
~~~

`make deploy-all` sincroniza primero los datos y modelos persistentes y sólo
después activa la release de código. Así, una release que dependa de un modelo
o de una rejilla nueva no se publica antes de que sus artefactos estén en el
servidor.

La configuración real de `deploy/deploy.env` está excluida de Git. El lector
acepta únicamente líneas simples `KEY=VALUE` y no utiliza `eval`. También se
pueden pasar opciones directamente al wrapper, por ejemplo
`bash scripts/deploy.sh --dry-run --no-restart`.

## 7. Sincronización de datos pesados con rsync

### 7.1 Preparar el acceso SSH de transferencia

La clave utilizada aquí es distinta de la Deploy Key de GitHub. En la máquina
de operación se puede crear:

~~~bash
ssh-keygen -t ed25519 \
  -f ~/.ssh/fire-risk-server \
  -C fire-risk-server-transfer
~~~

Instalar la clave pública en el usuario de despliegue del servidor según la
política de acceso de la organización. Si el usuario fire-risk conserva shell
nologin, utilizar un usuario administrativo/de despliegue y la opción --sudo,
o configurar un usuario de transferencia con permisos sólo sobre
/srv/fire-risk/data.

Antes de utilizar rsync, probar:

~~~bash
ssh -i ~/.ssh/fire-risk-server \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=yes \
  DEPLOY_USER@SERVIDOR 'id'
~~~

Se debe tener la entrada del servidor en known_hosts. No utilizar
StrictHostKeyChecking=no en producción.

### 7.2 Dry-run obligatorio

Desde la raíz del repositorio local:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR \
  --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server \
  --sudo \
  --dry-run
~~~

El dry-run lista los archivos que cambiarían y no sube datos. La opción --sudo
hace que mkdir y rsync se ejecuten remotamente mediante sudo -n. Si el usuario
remoto ya tiene permisos sobre los directorios, omitir --sudo.

### 7.3 Transferencia normal

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR \
  --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server \
  --sudo
~~~

Por defecto sincroniza:

| Origen local | Destino remoto | Uso |
|---|---|---|
| data/external/egif/ | data/external/egif/ | datacubo y datasets EGIF |
| data/models/ | data/models/ | modelos serializados y metadatos |
| data/processed/grid/galicia_grid_1km_egif.parquet | processed/grid/ | rejilla canónica |

Rsync es incremental: una segunda ejecución no vuelve a copiar archivos cuyo
tamaño y fecha no han cambiado. --partial permite continuar una transferencia
interrumpida. El script no propaga UID/GID del ordenador local; la propiedad
queda controlada por el servidor. No se utiliza --delete por defecto; así, un
error local no borra un artefacto válido del servidor.

Utilizar --checksum sólo para verificaciones periódicas o después de una copia
de seguridad. En datasets de varios gigabytes obliga a leer todos los archivos
en ambos extremos y puede tardar mucho.

Utilizar --delete únicamente después de revisar el dry-run:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR \
  --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server \
  --sudo --dry-run --delete
~~~

Nunca sincronizar desde el ordenador local las carpetas
data/processed/state, data/processed/observations, data/raw de producción ni
predicciones_operativas.parquet. Esas carpetas son propiedad del servidor y
contienen el estado generado por los jobs.

### 7.4 Sincronizar sólo un componente

Si sólo cambian los modelos:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server --sudo \
  --no-dataset --no-grid
~~~

Si sólo cambia el datacubo o los datasets:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server --sudo \
  --no-models --no-grid
~~~

Si sólo cambia la rejilla:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server --sudo \
  --no-dataset --no-models
~~~

## 8. Publicar rejilla y modelos

Preparar la rejilla en una máquina con el NetCDF:

~~~bash
PYTHONPATH=. python scripts/prepare_operational_grid.py \
  --cube data/external/egif/galicia_1km.nc \
  --output data/processed/grid/galicia_grid_1km_egif.parquet
~~~

Después transferirla con el comando de sólo rejilla. Entrenar los modelos EGIF
en una máquina con suficiente memoria y transferirlos con el comando de sólo
modelos. No entrenar en el servidor de inferencia durante una ejecución diaria.

En el servidor:

~~~bash
ssh ADMIN@SERVIDOR '
  sudo -u fire-risk /srv/fire-risk/app/scripts/check_server_deployment.sh
'
~~~

La inferencia sólo debe activarse cuando están presentes:

- la rejilla de 29.601 celdas;
- los tres artefactos T+1/T+2/T+3 compatibles con egif-2d-v1;
- el estado meteorológico mínimo;
- las claves del proveedor;
- el entorno Python.

## 9. Instalar y activar systemd

Tras el primer despliegue de código, instalar las plantillas:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo /srv/fire-risk/app/scripts/install_systemd_units.sh --enable --start'
~~~

Las unidades versionadas son:

- fire-risk-dashboard.service;
- fire-risk-inference.service;
- fire-risk-inference.timer;
- fire-risk-health.service;
- fire-risk-health.timer.

La inferencia se programa a las 05:15 y 10:15 para permitir reintentos o una
actualización posterior del forecast. El health check se ejecuta cada 30
minutos. El servidor debe tener la zona horaria Europe/Madrid y NTP activo.

Comprobar:

~~~bash
ssh ADMIN@SERVIDOR '
  sudo systemctl status fire-risk-dashboard.service
  sudo systemctl list-timers fire-risk-inference.timer fire-risk-health.timer
'
~~~

El dashboard escucha en 127.0.0.1:8501. Nginx o una VPN debe ser la única
entrada pública.

## 10. Actualizar código

Procedimiento recomendado:

1. crear y revisar un tag o commit;
2. ejecutar tests en local;
3. sincronizar modelos o rejilla si también cambiaron;
4. desplegar la release;
5. revisar la salida de compileall y pytest;
6. reiniciar Streamlit;
7. ejecutar el smoke test;
8. revisar el health check y el manifest.

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk /srv/fire-risk/bin/deploy_code_server.sh \
   --ref v0.3.1 --run-tests'

ssh ADMIN@SERVIDOR \
  'sudo systemctl restart fire-risk-dashboard.service'

ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk /srv/fire-risk/app/scripts/check_server_deployment.sh \
   --require-output'
~~~

No hacer git pull dentro del timer de inferencia. El código sólo cambia por un
despliegue explícito y auditable.

## 11. Actualizar modelos sin cambiar código

Los modelos serializados deben incluir horizonte, versión de contrato, versión
de dataset, semilla, hash y métricas. Transferir los tres artefactos compatibles
en una operación controlada y comprobar que no se mezclan releases:

~~~bash
PYTHONPATH=. scripts/sync_server_artifacts.sh \
  --host SERVIDOR --remote-user DEPLOY_USER \
  --ssh-key ~/.ssh/fire-risk-server --sudo \
  --no-dataset --no-grid
~~~

La escritura de modelos debería hacerse a una carpeta temporal o de staging y
promoverse cuando los tres ficheros estén presentes. Si se actualizan
manualmente, no borrar primero los modelos activos: conservar una copia de
rollback y comprobar el manifest de cada artefacto.

## 12. Rollback

Listar las releases existentes:

~~~bash
ssh ADMIN@SERVIDOR 'sudo -u fire-risk ls -1 /srv/fire-risk/releases'
~~~

Cambiar a una release conocida:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk /srv/fire-risk/bin/rollback_code_server.sh \
   --release 20260901T051500Z_ab12cd34ef56'

ssh ADMIN@SERVIDOR \
  'sudo systemctl restart fire-risk-dashboard.service'
~~~

El rollback cambia sólo el código. Los datos y modelos permanecen en el volumen
persistente. Si el cambio de modelo también fue la causa del problema, restaurar
el conjunto de modelos compatible desde el backup o desde el staging anterior.

## 13. Health check y operación diaria

Comandos de diagnóstico:

~~~bash
ssh ADMIN@SERVIDOR \
  'sudo -u fire-risk /srv/fire-risk/app/scripts/check_server_deployment.sh \
   --require-output'

ssh ADMIN@SERVIDOR 'sudo journalctl -u fire-risk-inference.service -n 100 --no-pager'
ssh ADMIN@SERVIDOR 'sudo journalctl -u fire-risk-health.service -n 100 --no-pager'
ssh ADMIN@SERVIDOR 'sudo journalctl -u fire-risk-dashboard.service -n 100 --no-pager'
~~~

El health check debe fallar si el output está desactualizado, incompleto, no
coincide con sus checksums o utiliza un forecast stale no permitido. Esa
condición debe conectarse a una alerta externa; un timer que falla no garantiza
por sí mismo que alguien reciba una notificación.

El servidor no debe considerar un proceso activo como prueba de que el producto
es válido. Hay que comprobar también el manifest, la antigüedad del forecast,
la cobertura, la versión del modelo y la calidad meteorológica.

## 14. Retención, backups y limpieza

Mantener al menos:

- releases de código suficientes para un rollback;
- los tres modelos actuales y la versión anterior;
- forecasts brutos y manifests;
- features operativas y predicciones publicadas;
- estado meteorológico;
- configuración sin secretos o un backup cifrado del .env según la política.

No borrar releases automáticamente desde el despliegue hasta definir una política
de retención. Cuando se automatice, limitar la limpieza a subdirectorios
concretos de /srv/fire-risk/releases y conservar siempre la release activa y
la anterior.

Los backups deben probarse restaurando en un servidor de prueba. Un backup que
no se ha restaurado no debe considerarse validado.

## 15. Checklist de puesta en producción

### Código y acceso

- [ ] Deploy Key registrada como read-only en el repositorio correcto.
- [ ] La clave privada sólo existe en el servidor y tiene permisos 600.
- [ ] known_hosts está poblado y verificado.
- [ ] El bootstrap no ha sobrescrito claves o .env existentes.
- [ ] El despliegue usa un tag o commit explícito.
- [ ] La release activa es un enlace simbólico.
- [ ] data está fuera del checkout y enlazado desde la release.

### Datos y artefactos

- [ ] rsync funciona con dry-run.
- [ ] Datasets y NetCDF aparecen en data/external/egif.
- [ ] La rejilla tiene 29.601 celdas.
- [ ] T+1, T+2 y T+3 están presentes y son compatibles.
- [ ] No se han sincronizado estados de producción desde el portátil.
- [ ] El servidor tiene espacio suficiente para datos, forecasts y backups.

### Servicios

- [ ] PIPELINE_ENVIRONMENT=production.
- [ ] LOCAL_SIMULATION_MODE=false.
- [ ] .env tiene permisos 600.
- [ ] Dashboard sólo escucha en localhost.
- [ ] systemd timers están habilitados.
- [ ] health check se ejecuta periódicamente.
- [ ] logs y alertas externas están configurados.
- [ ] backup y rollback se han probado.

## 16. Relación con el flujo científico

Este despliegue no cambia la interpretación del modelo. EGIF, ERA5-Land,
MeteoGalicia WRF y AEMET tienen roles diferentes:

- ERA5-Land y datasets históricos: entrenamiento y benchmark.
- MeteoGalicia WRF 1 km: fuente operativa preferente.
- MeteoGalicia WRF 4 km: fallback espacial etiquetado.
- AEMET: contingencia degradada o pruebas.
- El modelo 2D EGIF: estima riesgo por celda y horizonte.

La transferencia de archivos pesados sólo resuelve la distribución de artefactos.
La validez operativa sigue dependiendo de la calidad del forecast, ausencia de
fuga temporal, compatibilidad del contrato y evaluación independiente de T+1,
T+2 y T+3.
