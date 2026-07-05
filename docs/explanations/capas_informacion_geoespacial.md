# Capas de Información Geoespacial y Modelos de Combustible

> **Sección recomendada para el TFM:** *Fuentes de Datos y Marco Teórico de Ingeniería Ambiental.*
> Este documento detalla la base científica y la procedencia de cada variable geográfica utilizada en el sistema predictivo de incendios para Galicia.

---

## 1. Cobertura del Suelo: CORINE Land Cover y Modelos de Combustible

### ¿Qué es CORINE Land Cover?
**CORINE Land Cover (CLC)** es un inventario biofísico de la ocupación del suelo y del uso del territorio en Europa, gestionado por la Agencia Europea de Medio Ambiente (AEMA) y el programa Copernicus.
- **Resolución original:** Raster de 100 metros o vectorial con unidad cartográfica mínima de 25 hectáreas.
- **Taxonomía:** Utiliza una clasificación jerárquica de 3 niveles con **44 clases de uso de suelo** (desde zonas industriales hasta glaciares).

### ¿Por qué lo usamos? (El concepto de "Combustible Forestal")
En la ciencia de los incendios forestales, la vegetación no se analiza por su valor botánico, sino por sus propiedades físicas y químicas como **combustible**. El combustible forestal es cualquier materia orgánica (viva o muerta) capaz de arder.

No toda la vegetación arde igual. Para que el modelo de Machine Learning (XGBoost) entienda esto, mapeamos las 44 clases originales de CORINE a **8 macro-clases de comportamiento frente al fuego**:

| Macro-clase de Combustible | Comportamiento frente al fuego | Riesgo de Inicio | Velocidad de Propagación |
|---|---|---|---|
| **`bosque_coniferas`** | Contiene resinas y aceites inflamables (pinos). Carga de combustible vertical que facilita que el fuego pase de la superficie a las copas. | **Extremo** | Muy Alta |
| **`bosque_frondosas`** | Hojas anchas caducas o perennes (robles, castaños). Retienen más humedad. Arden a temperaturas más altas y propagan despacio. | **Bajo** | Baja |
| **`bosque_mixto`** | Mezcla de coníferas y frondosas. Comportamiento intermedio según la especie dominante. | **Moderado** | Media |
| **`matorral`** | Arbustos bajos y secos (toxo, brezo en Galicia). Combustible fino con baja humedad en verano. Prende con extrema facilidad. | **Extremo** | Alta |
| **`pastizal`** | Hierba y vegetación herbácea. Se seca rápido pero tiene poca biomasa. El fuego pasa rápido pero con poca intensidad. | **Moderado** | Alta |
| **`agricola`** | Cultivos. Actúa como cortafuegos en invierno y primavera, pero en verano (cereales secos) puede propagar el fuego. | **Bajo** | Media (verano) |
| **`urbano`** | Pavimento, hormigón y zonas edificadas. Combustible nulo, pero zona de interfaz urbano-forestal de alta vulnerabilidad humana. | **Nulo** (Estructura) | Nula |
| **`agua_humedal`** | Ríos, embalses, rías y turberas. Humedad saturada que actúa como barrera física absoluta. | **Nulo** | Nula |

### Dinámica Temporal del Suelo y Prevención de Fuga de Datos (Data Leakage)

A diferencia de la topografía, que permanece estática a escala humana, la vegetación y el uso del suelo sufren **dinámicas temporales lentas** (tala de bosques, crecimiento urbano, conversión agrícola, degradación tras incendios previos). 

El proyecto CORINE se actualiza en ciclos plurianuales de 6 años (2000, 2006, 2012, 2018 y el nuevo ciclo 2024, estimado para el tercer trimestre de 2026). Para gestionar este factor en la modelización y prevenir problemas metodológicos, el TFM adopta el siguiente diseño de **rejilla multi-temporal**:

1. **Grid Histórico 2012 (basado en CLC 2012):**
   - Utilizado para cruzar y entrenar sobre el histórico de incendios del periodo **2013-2018**.
2. **Grid Histórico 2018 (basado en CLC 2018):**
   - Utilizado para el entrenamiento y validación de los años **2019-2024**.
   - **Prevención de Temporal Data Leakage:** Cruzar un incendio ocurrido en 2019 con el mapa de combustibles de 2024 introduciría información del futuro en el entrenamiento. Por ejemplo, si una zona de bosque denso en 2019 fue talada o convertida en pastizal en 2023, figurará como pastizal en 2024. Entrenar el modelo con esta etiqueta errónea le enseñaría relaciones falsas que no se corresponden con el estado del combustible en el momento real de la ignición.
3. **Grid de Producción 2024 (basado en CLC 2024):**
   - Utilizado en el pipeline de inferencia operativa (producción 2026) para asegurar que el modelo trabaje sobre el estado más actualizado de la vegetación en la webapp.

### Flexibilidad y Reemplazo "Drop-in" en el Pipeline

Gracias a la arquitectura modular del código desarrollado en la Fase 1, la generación de cualquiera de las tres rejillas se realiza de forma paramétrica en segundos sin alterar el código de modelado ni el de la WebApp:

```bash
# Para generar el grid de 2012:
python -m src.geospatial.pipeline --corine data/raw/corine/clc_galicia_2012.tif --output data/processed/grid/galicia_grid_1km_2012.parquet

# Para generar el grid de 2018:
python -m src.geospatial.pipeline --corine data/raw/corine/clc_galicia.tif --output data/processed/grid/galicia_grid_1km_2018.parquet

# Para generar el grid de 2024:
python -m src.geospatial.pipeline --corine data/raw/corine/clc_galicia_2024.tif --output data/processed/grid/galicia_grid_1km_2024.parquet
```

Al heredar este diseño, las fases posteriores del proyecto asocian de forma dinámica la fila temporal del incendio con el parquet correspondiente a su año, eliminando por completo la fuga de datos espaciotemporales.

---


## 2. Topografía: Copernicus DEM (Modelo Digital de Elevaciones)

### ¿Qué es el Copernicus DEM GLO-30?
El **Copernicus DEM** es un Modelo Digital de Elevaciones (DEM) que representa la superficie terrestre global libre de edificios y vegetación alta (modelo digital de superficie suavizado).
- **Resolución:** GLO-30 ofrece una resolución espacial de aproximadamente **30 metros** por píxel.
- **Precisión:** Altamente preciso, ideal para análisis de micro-relieve regional.

### ¿Por qué importa la topografía en un incendio?
El relieve altera la física de la propagación del fuego y las condiciones locales de la atmósfera. El pipeline calcula tres variables clave:

1. **Altitud Media (`altitud_media`):**
   - **Efecto físico:** A mayor altitud, la temperatura media del aire disminuye (gradiente térmico adiabático de ~0.65°C por cada 100m) y la humedad relativa suele ser mayor. Las zonas de alta montaña arden menos por condiciones climáticas basales.
2. **Pendiente Media (`pendiente_media`):**
   - **Efecto físico (Efecto Chimenea):** El fuego se propaga cuesta arriba mucho más rápido porque las llamas precalientan por radiación y convección el combustible (ramas, hojas) que está ladera arriba antes de que llegue el frente de fuego. Una pendiente de 15° dobla la velocidad de propagación respecto al plano.
3. **Orientación de la Ladera (`orientacion_media` / `orientacion_clase`):**
   - **Efecto físico (Solana vs. Umbría):** En el hemisferio norte, las laderas orientadas al **Sur (Solana)** reciben mucha más radiación solar directa, lo que eleva la temperatura del suelo y evapora la humedad de la vegetación fina. Las laderas **Norte (Umbría)** son más frías y húmedas, presentando un riesgo basal menor de ignición.

---

## 3. Variable Objetivo: NASA FIRMS (Anomalías Térmicas)

### ¿Qué es NASA FIRMS?
El **Fire Information for Resource Management System (FIRMS)** de la NASA distribuye datos de fuego activo en tiempo casi real (NRT) detectados por los sensores espaciales **MODIS** (satélites Terra y Aqua) y **VIIRS** (satélites Suomi NPP y NOAA-20).

### Sensores y resolución detectada
- **MODIS:** Resolución de pixel de 1 km. Detecta anomalías desde el año 2000.
- **VIIRS:** Resolución de pixel de **375 metros**. Detecta fuegos más pequeños y ofrece mejor delimitación nocturna. Es nuestro sensor prioritario en el MVP.

### ¿Cómo funciona la detección térmica?
El satélite mide la radiación infrarroja de onda media (~3.9 µm), que es la longitud de onda donde la emisión de cuerpos calientes (como frentes de llama a 600°C - 1000°C) se dispara exponencialmente en comparación con la reflectancia solar normal del suelo circundante.
- **Confidence (Confianza):** Clasifica las detecciones en baja, nominal o alta. Para el TFM, filtramos y usamos únicamente **nominal y alta** para descartar reflejos solares en tejados o quemas industriales estables.
- **FRP (Fire Radiative Power):** Mide la tasa de energía liberada por el fuego en Megavatios (MW), dándonos una estimación indirecta del tamaño e intensidad del frente térmico.

---

## 4. Clima Histórico: Copernicus ERA5-Land (Reanálisis)

### ¿Qué es ERA5-Land?
**ERA5-Land** es un dataset de reanálisis global producido por el Centro Europeo de Previsiones Meteorológicas a Plazo Medio (ECMWF). Combina observaciones meteorológicas reales de globos, estaciones y satélites con ecuaciones físicas del clima mediante asimilación de datos.
- **Resolución:** Rejilla regular de **9 km** con frecuencia **horaria** desde 1950 hasta el presente.
- **Ventaja en el TFM:** Proporciona un registro histórico continuo e idéntico para cada coordenada de Galicia, sin sufrir los problemas de falta de datos ("missing values") que plagan a las estaciones meteorológicas terrestres individuales.

### ¿Por qué lo usamos para entrenar el modelo?
Para que el modelo aprenda la relación causa-efecto real entre meteorología e incendios, necesita datos climáticos estables y precisos de lo que **ocurrió de verdad** el día del incendio. ERA5-Land es la fuente de verdad histórica del clima terrestre.

---

## 5. Inferencia Operativa: MeteoGalicia (Predicción y Red Física)

### ¿Por qué cambiar de ERA5 a MeteoGalicia en producción?
ERA5-Land tiene dos limitaciones para usarse operativamente en el día a día:
1. **Latencia:** Los datos consolidados tardan meses en publicarse. La versión rápida (ERA5T) tiene una latencia de 5 días. No sirve para predecir "mañana".
2. **Previsión:** ERA5-Land solo describe el pasado. No produce pronósticos para las próximas 24, 48 o 72 horas.

Para el pipeline de inferencia diaria en tiempo real, el sistema cambia a **MeteoGalicia**:
- **Densidad:** Dispone de **170 estaciones automáticas** distribuidas por Galicia (frente a las ~35 de la red estatal AEMET en el territorio), cubriendo valles y zonas de monte.
- **Previsión local (WRF):** Corre un modelo de predicción meteorológica numérico regional a **4 km** de resolución espacial ajustado a la orografía gallega.
- **Variables de suelo:** Mide humedad y temperatura del suelo en tiempo real, lo que permite refinar los índices de sequedad acumulada.
