---
name: conceptual-documentation
description: >
  Guía para la redacción de explicaciones conceptuales y documentación de marco teórico 
  orientada a la memoria del TFM. Define el tono académico y de negocio para describir 
  conceptos geoespaciales, meteorológicos y físicos del proyecto.
---

# Skill: Redacción Conceptual y Marco Teórico TFM

Este skill guía a los agentes de IA sobre cómo documentar los conceptos científicos, físicos y geográficos del proyecto para que las explicaciones puedan ser integradas directamente en la memoria del TFM y entendidas tanto por técnicos como por perfiles de negocio.

---

## 1. Tono y Enfoque de Redacción

- **Riguroso pero legible:** Evitar el exceso de tecnicismos incomprensibles para personas sin formación en GIS o Data Science. Usar analogías claras (ej: la relación entre ignición humana y meteorología explicada como "trigger vs. amplificador" o "accidentes de tráfico").
- **Orientación a Negocio/Gestión:** Explicar siempre el **por qué** y el **valor operativo**. ¿De qué le sirve a un gestor de incendios saber que una ladera está orientada al Sur? (Solana, evaporación de humedad, precalentamiento del combustible).
- **Consistencia científica:** Respetar los términos estándar de ecología del fuego (combustibles, ignición, propagación, interfaz urbano-forestal) y teledetección (anomalías térmicas, firmas infrarrojas, resolución espacial, reanálisis).

---

## 2. Estructura de Explicación de Fuentes Geográficas

Cada vez que el usuario o el flujo del TFM requiera documentar una nueva capa de datos (como Sentinel-2, Copernicus EMS, etc.), la explicación debe seguir la siguiente estructura:

1. **¿Qué es la fuente?** (Definición formal, resolución espacial y temporal, procedencia).
2. **¿Cómo funciona?** (Física o captura del dato de forma simplificada).
3. **¿Por qué importa para los incendios?** (Efecto físico real sobre la ignición o la propagación).
4. **¿Cómo se procesa en el pipeline?** (Mapeo de códigos, interpolación, agregación zonal).

---

## 3. Glosario de Equivalencias Clave (Garantizar consistencia)

Al documentar, utilizar y mantener siempre las siguientes equivalencias:
- **Vegetación ➔ Combustible:** La vegetación se trata cuantitativamente como combustible vegetal (biomasa fina/gruesa, viva/muerta).
- **Anomalía Térmica ➔ Incendio:** NASA FIRMS no ve "árboles quemados", ve firmas térmicas infrarrojas en superficie (anomalías térmicas).
- **ERA5-Land ➔ Reanálisis:** ERA5 no es predicción meteorológica, es una reconstrucción histórica (reanálisis asimilado).
- **MeteoGalicia ➔ Red Física de Producción:** La fuente que resuelve el gap temporal y la baja resolución espacial de las redes globales en la fase operativa.

---

## 4. Estructura de Documentos en `docs/explanations/`

Los archivos en esta carpeta deben estructurarse con títulos claros, tablas resumen de variables y justificaciones de comportamiento físico. Siempre se debe incluir una sección final que resuma cómo el concepto explicado impacta directamente en las decisiones de negocio o en la toma de decisiones operativas de un cuerpo de bomberos forestales o protección civil.
