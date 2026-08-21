# 03. Ingeniería de Características (Features) y Desfase Anti-Leakage (T-1)

---

## 1. ¿Qué se ha hecho?
- Implementation del módulo de transformaciones temporales en `src/features/temporal_shift.py`.
- Cálculo de variables de **memoria climática acumulada**: precipitación acumulada a 7 días (`prec_acum_7d`), precipitación acumulada a 30 días (`prec_acum_30d`) y temperatura máxima media a 7 días (`tmax_media_7d`).
- Aplicación estricta del **desfase temporal de 24 horas ($T-1$)** sobre la matriz de predictores meteorológicos (`tmax_t1`, `tmean_t1`, `rh_min_t1`, `wind_speed_t1`, `precip_t1`).
- Encodificación de **variables cíclicas estacionales**: seno y coseno del día del año (`dia_año_sin`, `dia_año_cos`), mes, día de la semana y flag de fin de semana (`es_finde`).
- Implementación de la regla de negocio experta `alerta_30_30` (activada cuando $T_{max} \ge 30^\circ\text{C}$ y $RH_{min} \le 30\%$).

## 2. ¿Por qué se ha hecho?
- **Erradicar la Fuga de Datos (Data Leakage):** Para que el sistema sea capaz de generar alertas de riesgo operativas por la mañana antes de que comiencen los fuegos, el modelo debe aprender obligatoriamente las relaciones entre la ignición de hoy ($Y_T$) y la meteorología observada hasta ayer ($X_{T-1}$).
- **Capturar el Secado Progresivo del Combustible:** La vegetación no se vuelve inflamable por un único día caluroso, sino por la acumulación continuada de días secos y altas temperaturas. Las ventanas móviles de 7 y 30 días simulan físicamente el estrés hídrico de la biomasa viva y muerta.
- **Evitar Discontinuidades en Fechas:** Las variables cíclicas sinusoidales garantizan que el día 365 (31 de diciembre) se considere contiguo al día 1 (1 de enero) sin saltos numéricos artificiales.

## 3. ¿Cómo se ha hecho?
1. Se ordena el dataset por celda (`cell_id`) y fecha ascendente (`fecha`).
2. Se calculan las ventanas móviles agrupadas por `cell_id` mediante `.shift(1).rolling(w).sum()` o `.mean()`.
3. Se desplazan las columnas explicativas un intervalo `.shift(1)`.
4. Se concatenan las columnas de características transformadas al DataFrame maestro tabular.

## 4. ¿Por qué se han elegido estas tecnologías?
- **Pandas & NumPy Vectorizado:** Permite calcular transformaciones sobre millones de filas de forma rápida sin bucles explícitos en Python.
- **Operadores de Ventana Móvil de Pandas (`rolling`):** Garantizan la preservación del orden temporal sin invadir valores futuros.

## 5. ¿Qué conseguimos con ello?
- **Matriz de Características Robusta:** Dataset de entrada para Machine Learning totalmente alineado con las leyes físicas del combustible forestal.
- **Garantía Académica para el TFM:** Demostración explícita ante el tribunal de que el modelo predice el riesgo futuro sin hacer "trampas" con datos del mismo día.
