# Reglas del Espacio de Trabajo — Sistema de Documentación Obligatoria

Este archivo define las directrices y restricciones de comportamiento para cualquier Agente de Inteligencia Artificial que colabore en este repositorio.

---

## 📋 Regla General de Documentación

**Cada vez que se complete una fase, hito o cambio de código significativo, es OBLIGATORIO y AUTOMÁTICO generar o actualizar un archivo de registro en el directorio `docs/tasks/`.**

Además del registro de tarea, el agente debe **recolectar las explicaciones conceptuales, decisiones metodológicas y justificaciones de negocio** que se definan en la conversación e integrarlas/actualizarlas en el borrador de la memoria [docs/tfm_borrador_memoria.md](file:///Users/junki/Desktop/proyectos/Sistema-deteccion-incendios/docs/tfm_borrador_memoria.md) en su capítulo correspondiente.

El agente no debe dar una tarea por completada sin antes haber verificado y actualizado ambos documentos.

---

## 📁 Convención de Nombres y Estructura en `docs/`

La documentación del repositorio debe estructurarse estrictamente de la siguiente manera:

1. **`docs/tasks/`** (Histórico cronológico de desarrollos):
   - Los archivos deben llevar el prefijo secuencial de dos dígitos correspondiente al orden del roadmap de fases del TFM (ej: `01_infraestructura_geoespacial.md`, `02_ingesta_datos.md`).
2. **`docs/technical/`** (Referencias de arquitectura y código):
   - Estructura de bases de datos, diagramas de flujo de datos, manuales de API de MeteoGalicia/AEMET, etc.
3. **`docs/explanations/`** (Justificaciones conceptuales y de negocio):
   - Modelos de negocio, justificaciones de la región del MVP, discusiones científicas (ej. causa humana vs. predicción natural), y análisis de degradación.

---

## 📄 Plantilla Estricta para `docs/tasks/`

El archivo de documentación de tareas debe estructurarse obligatoriamente bajo los siguientes cinco apartados:

```markdown
# [Nombre Secuencial de la Tarea]

---

## 1. ¿Qué se ha hecho?
[Resumen conciso y en viñetas de las características técnicas implementadas, los archivos creados y los outputs generados]

## 2. ¿Por qué se ha hecho?
[Justificación del valor de esta tarea para el proyecto o por qué es un paso necesario dentro de las fases del TFM]

## 3. ¿Cómo se ha hecho?
[Descripción detallada de la lógica algorítmica y el flujo de ejecución del código]

## 4. ¿Por qué se han elegido estas tecnologías?
[Justificación técnica de la selección de librerías, algoritmos y herramientas específicas utilizadas]

## 5. ¿Qué conseguimos con ello?
[Beneficios inmediatos para el equipo, impacto en el TFM y valor de negocio/académico obtenido]
```

---

## 📚 Directrices de Estilo Académico y Citas del TFM

Al generar explicaciones conceptuales o documentar decisiones del modelo en `docs/explanations/` o en los informes, la IA debe seguir las siguientes pautas:

1. **Alineación con el Estado del Arte (Citas Obligatorias):**
   - **IberFire** (Ercibengoa et al., 2025): Posicionar nuestro desarrollo como un pipeline operativo *end-to-end* en tiempo real con MeteoGalicia, tomando la estructura de datos de IberFire como validación y benchmark metodológico de 1km x 1km.
   - **IPIF de AEMET** (2026): Justificar que nuestro proyecto sigue la misma tendencia institucional de integrar variables multimodales de vegetación y suelo más allá del FWI clásico, pero mediante un enfoque de Machine Learning supervisado y calibrado.
   - **EFFIS / GEFF**: Contrastar nuestra resolución local de 1km frente a los modelos continentales europeos de 10-25km de resolución.
2. **Tono "Negocio-Académico":**
   - Las explicaciones técnicas deben incluir un párrafo resumen orientado a perfiles no técnicos (Negocio/Tribunal) explicando el impacto físico real de las variables (ej: por qué la orientación solana o la pendiente multiplican la velocidad de propagación).

---

## ⚠️ Restricciones adicionales para agentes IA

- **Anti-Data-Leakage:** Garantizar en cada script de features que no se utiliza información del día T para predecir el riesgo del día T. Además, respetar la **estrategia de rejilla multi-temporal** (Opción A): cruzar datos históricos de 2013-2018 con la rejilla basada en CORINE 2012, datos de 2019-2024 con la rejilla basada en CORINE 2018, y datos operativos de producción (2025+) con CORINE 2024 para evitar la fuga de datos espaciotemporales.
- **Ruta de datos:** No escribir nunca archivos de datos pesados (.nc, .parquet, .geojson grandes) directamente en Git. Verificar que están en `.gitignore`.
- **Modo Desarrollo:** Los scripts y módulos de Python en `src/` deben ser testeables con pytest y formateados con Ruff.
