from src.features.tabular import obtener_columnas_predictoras


def test_year_es_columna_de_particion_no_predictor() -> None:
    columns = obtener_columnas_predictoras(
        {
            "year",
            "month",
            "temperature_mean",
            "precipitation_sum",
            "precipitation_sum_1d",
            "aspect_no_data_fraction",
            "target_ignicion",
            "burned_area_ha",
            "large_fire_500ha",
            "is_galicia",
            "cell_id",
        }
    )

    assert columns == ["month", "precipitation_sum", "temperature_mean"]
