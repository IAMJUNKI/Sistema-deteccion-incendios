# Módulo: webapp — Dashboard e Integración Operativa

**Fase 5 del proyecto.** Dashboard interactivo en Streamlit y pipeline de inferencia diaria alimentado con previsiones AEMET.

## Responsabilidad

- Construir el dashboard Streamlit con mapas de riesgo interactivos (Folium/PyDeck).
- Implementar filtros por provincia, municipio y horizonte temporal (24h/48h/72h).
- Mostrar panel de interpretabilidad con valores SHAP simplificados.
- Programar el script de inferencia diaria (cron job 05:00 AM) que descarga previsiones AEMET y genera el mapa de riesgo del día siguiente.
- Documentar y cuantificar la degradación de rendimiento (ERA5 real vs. previsión AEMET).

## Entregable

- WebApp desplegada en local o Streamlit Cloud.
- Script de producción para el pipeline de inferencia diaria.
- Análisis comparativo de rendimiento (datos perfectos vs. previsión).

## Dependencias externas

- `AEMET_API_KEY` en `.env` (ver `.env.example`).
- Modelo serializado disponible en `data/models/`.
