# 12. Documentación Integral y Sencilla del Pipeline Operativo en Producción para la Memoria del TFM

---

## 1. ¿Qué se ha hecho?
* **Redacción de la Arquitectura Operativa End-to-End:** Se ha elaborado una descripción completa, clara y pedagógica del funcionamiento del pipeline en producción, estructurada cronológicamente en 7 fases consecutivas (desde la activación por temporizador hasta la visualización táctica en el Centro de Mando).
* **Integración en la Memoria del TFM:** Se ha incorporado la nueva subsección `7.3` en [docs/tfm_borrador_memoria.md](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/docs/tfm_borrador_memoria.md), incluyendo un diagrama de flujo en Mermaid y un apartado de síntesis no técnica orientado a negocio y tribunal.
* **Alineación con el Estado del Arte:** Se han contextualizado formalmente las ventajas de nuestra arquitectura frente a referencias canónicas: IberFire (Ercibengoa et al., 2025), IPIF de AEMET (2026) e índices continentales europeos de EFFIS/GEFF.

## 2. ¿Por qué se ha hecho?
* **Requisito Académico Central del TFM:** Para la redacción de la memoria y la defensa del Trabajo de Fin de Máster, es indispensable contar con una explicación nítida que sintetice cómo la investigación de modelos predictivos se traslada a un servicio de producción real y desasistido.
* **Claridad Conceptual para Perfiles No Técnicos:** Se requería una explicación accesible que permita tanto a evaluadores académicos como a mandos operativos de protección civil comprender por qué cada etapa del procesamiento es necesaria sin perderse en tecnicismos irrelevantes.

## 3. ¿Cómo se ha hecho?
* **Estructuración en 7 Etapas Consecutivas:**
  1. *Fase 1 (Control de Concurrencia):* Exclusión mutua mediante lock de fichero (`.daily_inference.lock`).
  2. *Fase 2 (Memoria Ambiental Antecedente):* Fusión multi-fuente de AEMET (< D-4) y la red EMA de MeteoGalicia (D-4 a D-1) vía IDW de 4 vecinos para cerrar los 30 días móviles sin data leakage.
  3. *Fase 3 (Pronóstico Futuro a 72h):* Cadena de fallback adaptativa (MeteoGalicia WRF 1km $\rightarrow$ WRF 4km $\rightarrow$ AEMET municipal $\rightarrow$ stale) con validación física rigurosa.
  4. *Fase 4 (Ingeniería de Variables):* Agregación en la ventana crítica de máxima inflamabilidad (12:00 a 18:00 h local) y extracción del contrato `egif-2d-48-v1`.
  5. *Fase 5 (Inferencia y Calibración):* Tres modelos LightGBM serializados para $T+1, T+2, T+3$ y calibración sigmoide de Platt.
  6. *Fase 6 (Estratificación Táctica):* Doble escala: Severidad física absoluta PLADIGA (1 a 5) y Priorización relativa de despacho por percentiles (Top 1%).
  7. *Fase 7 (Publicación y Consumo):* Escritura atómica (`os.replace`), manifiesto con firmas criptográficas SHA-256, visualización desacoplada en Streamlit y sincronización en Hugging Face Hub.

## 4. ¿Por qué se han elegido estas tecnologías?
* **Diagramas Mermaid en Markdown:** Facilitan la comprensión visual del flujo de datos de extremo a extremo directamente integrable en la memoria del TFM.
* **Muestreo por Puntos Representativos (Sampling cada 4 km):** Reduce 29.601 peticiones potenciales a bloques de 20 puntos consultados a la API MeteoSIX v5, mitigando latencia y respetando los límites del proveedor.
* **Escritura Atómica (`os.replace`) y Manifiestos SHA-256:** Evitan que el dashboard visualice ficheros Parquet incompletos o corruptos y proporcionan trazabilidad reproducible completa.

## 5. ¿Qué conseguimos con ello?
* **Texto Listo para la Memoria Oficial:** El autor dispone de una descripción pulida y estructurada para su inserción directa en el Capítulo 7 de la memoria del TFM.
* **Defensa Sólida ante el Tribunal:** Proporciona argumentos contundentes que justifican las decisiones de ingeniería de datos (cierre de latencia observacional, ventanas críticas y prevención estricta de fugas temporales).
