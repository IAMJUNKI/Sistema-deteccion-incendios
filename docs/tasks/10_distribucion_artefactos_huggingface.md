# 10. Distribución de Modelos y Artefactos Operativos vía Hugging Face Hub

---

## 1. ¿Qué se ha hecho?

- **Script de sincronización y subida a Hugging Face (`scripts/publish_to_huggingface.py`):**
  - Implementación de un cliente CLI basado en `huggingface_hub` con soporte para dos modalidades de operación:
    - Modo completo (`--all`): Sube artefactos inmutables (modelos calibrados `.joblib`, contratos `.json`, manifiesto activo `active_model_manifest.json` y rejilla base `galicia_grid_1km_egif.parquet`) más los datos operativos dinámicos.
    - Modo diario operativo (por defecto): Sincroniza exclusivamente el pronóstico del día (`predicciones_operativas.parquet`), su manifiesto de trazabilidad y la ventana rodante de 30 días de estado meteorológico (`weather_daily_state.parquet`).
  - Creación automática y configurable del repositorio remoto como modelo privado o público.
- **Script de aprovisionamiento inmediato para usuarios (`scripts/download_artifacts.py`):**
  - Descargador desasistido que coloca los artefactos en sus rutas exactas del proyecto (`data/models/`, `data/processed/grid/`, `data/processed/state/`, `data/processed/`).
  - Soporte para modo ligero (`--only-dashboard`, ~25 MB) para visualizar mapas y explicabilidad SHAP al instante, y modo completo (~170 MB) para habilitar también la inferencia de nuevos días.
- **Configuración de variables de entorno y dependencias:**
  - Actualización de `.env.example` con `HF_REPO_ID` y `HF_TOKEN`.
  - Inclusión de `huggingface_hub>=0.20.0` en `environment.yml`.

## 2. ¿Por qué se ha hecho?

- **Resolución del problema de peso en Git sin comprometer la usabilidad:** Los modelos serializados (~8 MB), la rejilla geoespacial (~3.5 MB), las predicciones operativas (~17 MB) y el estado acumulado (~145 MB) están ignorados en `.gitignore` para no saturar el repositorio de código. Sin embargo, un nuevo colaborador o evaluador del TFM no podía ejecutar el sistema tras hacer `git clone` sin pasar por un proceso de entrenamiento y preprocesamiento de más de 20 GB de datos.
- **Solución al problema del "Cold Start" en inferencia:** Para estimar las 48 variables del modelo (sequedad acumulada, anomalías térmicas y déficit de vapor), se necesitan 30 días de observaciones previas. Descargarlas de cero de las APIs meteorológicas en un entorno nuevo llevaría minutos y saturaría cuotas de servicio; suministrar un estado reciente como punto de partida permite hacer un "Warm Start" y ponerse al día en segundos.

## 3. ¿Cómo se ha hecho?

- **Segmentación de artefactos entre estáticos y dinámicos:**
  - *Artefactos estáticos (inmutables):* Modelos `forecast_risk_egif_48_t{1,2,3}.joblib`, metadatos y rejilla base de 29.601 celdas. Se versionan y suben de manera puntual tras reentrenamientos.
  - *Artefactos dinámicos (ventana deslizante):* `predicciones_operativas.parquet` y `weather_daily_state.parquet`. Se actualizan en producción cada mañana tras la corrida del modelo.
- **Integración con Hugging Face Hub:**
  - Uso de `HfApi.upload_file` para transferencias atómicas archivo a archivo con mensajes de commit descriptivos.
  - Uso de `snapshot_download` filtrado mediante `allow_patterns` para descargar únicamente los archivos pertinentes de forma resumable y paralelizada.
- **Cadena de producción en servidor:**
  - Al concluir la inferencia diaria matinal en el servidor Linux, el pipeline invoca `publish_to_huggingface.py`, manteniendo el repositorio de Hugging Face actualizado para cualquier cliente externo.

## 4. ¿Por qué se han elegido estas tecnologías?

- **Hugging Face Hub frente a Git-LFS / almacenamiento propio:**
  - Hugging Face es el estándar abierto de la comunidad de Machine Learning, con soporte nativo para `Model Cards`, versionado de artefactos y descargas resumables con caché local.
  - Elimina el coste y la complejidad de configurar servidores S3/FTP con credenciales compartidas o saturar la cuota de ancho de banda de Git-LFS de GitHub.
- **Biblioteca `huggingface_hub`:**
  - Proporciona autenticación limpia vía token de entorno (`HF_TOKEN`), verificación de integridad por hash SHA-256 y compatibilidad multiplataforma sin depender de herramientas de terminal como `git-lfs` o `curl`.

## 5. ¿Qué conseguimos con ello?

- **Experiencia de usuario y tribunal "Zero-Retrain / Zero-Config":** Un evaluador del TFM o nuevo desarrollador puede clonar el repositorio, ejecutar `python scripts/download_artifacts.py` y abrir inmediatamente el panel de control (`streamlit run app.py`) con datos actualizados de Galicia en menos de un minuto.
- **Desacoplamiento arquitectónico MLOps:** El entrenamiento pesado (20+ GB de NetCDF ERA5 y tabulares anuales) queda completamente aislado del ciclo de vida operativo e inferencia diaria (170 MB), garantizando reproducibilidad y escalabilidad.
