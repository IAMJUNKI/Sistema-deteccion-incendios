# =============================================================================
# Atajo de conveniencia para lanzar el entrenamiento sin escribir la ruta de
# Python completa cada vez.
#
# Uso:
#   .\run.ps1 lightgbm xgboost              -> pasa todo lo que le sigue tal cual
#                                                a `python -m scripts.train_model`
#   .\run.ps1 --evaluate-test               -> test ciego
#
# No sustituye a un alias de perfil (que cada persona puede crear en su propio
# $PROFILE si quiere); esto simplemente evita repetir la ruta del intérprete.
# =============================================================================

$py = "$env:USERPROFILE\anaconda3\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "No se encuentra el intérprete en $py. Ajusta la ruta en run.ps1."
    exit 1
}

Set-Location $PSScriptRoot
& $py -m scripts.train_model @args
