# Por qué predecir incendios tiene sentido aunque el 96% sean de causa humana

## TFM: Sistema Predictivo de Anticipación de Incendios Forestales

---

## El dato que parece invalidar el proyecto

El 96% de los incendios forestales en España tienen causa humana — accidental o intencionada. En Galicia, la cifra es aún más extrema: estudios regionales estiman que entre el 60% y el 70% son provocados deliberadamente (quemas agrícolas ilegales, piromanía, conflictos territoriales), y el porcentaje de causa humana total supera el 95%.

A primera vista, esto parece romper la premisa del proyecto: si los incendios los provocan personas, ¿cómo puede un modelo meteorológico y ambiental predecirlos?

La respuesta está en entender exactamente qué predice el sistema — y qué no.

---

## La distinción fundamental: trigger vs. amplificador

```
Lo que NO predecimos:
  "¿Decidirá alguien provocar un incendio mañana?"
  → Comportamiento humano. Aleatorio e impredecible. Correcto.

Lo que SÍ predecimos:
  "Si mañana se produce una ignición (por cualquier causa),
   ¿harán las condiciones que se propague y se vuelva catastrófica?"
  → Determinado por física atmosférica y ambiental. Muy predecible.
```

La ignición humana es el **detonante**. Las condiciones ambientales son el **amplificador**. El detonante ocurre casi todos los días en algún punto de Galicia. El amplificador es lo que determina si ese detonante produce un fuego que se apaga solo en 20 minutos o uno que arrasa 10.000 hectáreas.

**La misma acción humana, dos resultados radicalmente distintos:**

| Escenario | Condiciones | Resultado |
|---|---|---|
| Alguien quema rastrojos en enero | 12°C, humedad 80%, sin viento, lluvia hace 2 días | Fuego controlado, se extingue en horas |
| Alguien quema rastrojos en agosto | 38°C, humedad 18%, viento 60 km/h del este, 22 días sin lluvia | Incendio catastrófico, miles de hectáreas |

El modelo no predice al humano. Predice cuándo las condiciones convierten cualquier ignición en catástrofe.

---

## Por qué la causa humana hace el modelo más predecible, no menos

Paradójicamente, que el 96% de los incendios sean de causa humana **ayuda** al modelo. El comportamiento humano no es uniformemente aleatorio — correlaciona con las mismas condiciones que hacen el fuego peligroso:

### 1. Las personas que queman intencionalmente eligen los días de riesgo

Quien quema para limpiar terreno o por otros motivos elige días secos y con viento: necesita que el fuego se propague. Esto significa que los días de mayor ignición humana intencionada **coinciden** con los días de mayor peligro meteorológico. El modelo captura exactamente esa correlación.

### 2. Las igniciones accidentales se vuelven visibles solo en condiciones extremas

Una chispa de maquinaria agrícola, una colilla mal apagada o una línea eléctrica caída ocurren cualquier día del año. La inmensa mayoría se extingue sola sin llegar a registrarse como incendio forestal. Solo en condiciones de extrema sequedad, calor y viento esas igniciones se propagan lo suficiente para ser detectadas por el satélite y entrar en el registro de FIRMS. El modelo aprende esas condiciones umbrales.

### 3. Los patrones espaciales humanos son estables y predecibles

Los incendios no ocurren en celdas al azar del territorio — se concentran sistemáticamente:
- **Cerca de carreteras y núcleos urbanos** → donde hay presencia humana constante
- **En determinados tipos de vegetación** → eucaliptal y pinar de Galicia, con alta carga de combustible
- **En laderas de orientación sur** → más secas, más expuestas a la radiación solar
- **En zonas con historial de incendios previos** → la recurrencia es un patrón documentado en Galicia

Todos estos factores están capturados por las features del modelo (`dist_carretera_m`, `combustible_clase`, `orientacion`, `n_incendios_celda_3años`).

---

## La analogía de los accidentes de tráfico

El 94% de los accidentes de tráfico son causados por error humano. Sin embargo, los sistemas de predicción de accidentalidad vial no intentan predecir si alguien va a cometer un error — predicen qué condiciones (lluvia, visibilidad, densidad de tráfico, tipo de vía) hacen que los errores humanos resulten en accidente grave.

El resultado es perfectamente útil: permite reforzar la vigilancia en los tramos y momentos de mayor riesgo.

Nuestro sistema funciona exactamente igual para incendios forestales.

---

## El valor operativo real

El usuario final del sistema — un gestor de la Consellería de Medio Ambiente de la Xunta de Galicia o un jefe de brigada de extinción — no espera que el modelo le diga quién va a provocar un incendio. Lo que necesita es:

> *"Mañana, en la zona de A Fonsagrada y el entorno del Courel, las condiciones son de riesgo extremo. Pre-posicionar medios aéreos. Activar protocolo de vigilancia en carreteras forestales. Emitir alerta preventiva a los ayuntamientos."*

Para esa decisión, el modelo es completamente útil — independientemente de que el incendio que se produzca lo haya iniciado una persona.

---

## Precedentes operativos que validan el enfoque

Este no es un enfoque nuevo. Los sistemas de predicción de peligro de incendio basados en condiciones ambientales llevan décadas en uso operativo a nivel internacional:

| Sistema | País/Región | Base | En uso desde |
|---|---|---|---|
| **Canadian FWI** (Fire Weather Index) | Canadá / Global | Temperatura, humedad, viento, precipitación | 1970s |
| **NFDRS** (National Fire Danger Rating System) | Estados Unidos | Meteorología + vegetación | 1972 |
| **McArthur FFDI** | Australia | Temperatura, humedad, viento, sequedad | 1960s |
| **EFFIS / GEFF** | Europa (Copernicus) | ERA5 + FWI | 2000s |
| **IPIF** | España (AEMET) | Meteorología + humedad suelo + satélite | 2026 |

Todos estos sistemas predicen el peligro de incendio en territorios donde la inmensa mayoría de los incendios tienen causa humana. Todos son operativamente útiles. Ninguno intenta predecir el comportamiento humano.

Nuestro TFM adopta el mismo principio, pero con Machine Learning supervisado entrenado sobre datos reales de ignición histórica en lugar de índices físicos empíricos — lo que nos permite capturar patrones más complejos y no lineales entre las variables.

---

## Cómo articular esto en la memoria del TFM

> *"El 96% de los incendios forestales en España tienen causa humana, ya sea accidental o intencionada. Lejos de invalidar el enfoque predictivo, este hecho subraya su valor: el sistema no predice la intención humana —impredecible por definición— sino la vulnerabilidad ambiental del territorio en un momento dado. Los datos históricos de ignición muestran que los incendios, independientemente de su causa, se concentran sistemáticamente en condiciones de temperatura extrema, baja humedad, viento intenso y sequedad acumulada, y en zonas con alta accesibilidad humana y determinados tipos de combustible. El modelo aprende estas condiciones de riesgo, que son las mismas que transforman cualquier ignición —accidental o deliberada— en un incendio de propagación incontrolada. La variable objetivo no es '¿quemará alguien hoy?' sino '¿si hoy se produce una ignición, las condiciones del territorio favorecerán su propagación catastrófica?'"*
