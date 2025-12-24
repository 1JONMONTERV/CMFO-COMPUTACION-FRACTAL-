#!/bin/bash
# ================================================================
# CMFO - Script de Ejecución Completa (Linux/Mac)
# Ejecuta todo el sistema y guarda resultados en carpeta 'resultados'
# ================================================================

echo ""
echo "================================================================"
echo "  CMFO - COMPUTACION FRACTAL DEL UNIVERSO"
echo "  Sistema Completo de Ejecucion"
echo "================================================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 no encontrado. Por favor instala Python 3.8 o superior."
    exit 1
fi

echo "[1/3] Verificando dependencias..."
python3 -c "import torch; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Instalando dependencias..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudieron instalar las dependencias."
        exit 1
    fi
fi

echo "[2/3] Creando carpeta de resultados..."
mkdir -p resultados

echo "[3/3] Ejecutando sistema CMFO completo..."
echo ""
python3 run_cmfo_complete.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: La ejecucion fallo. Revisa los mensajes anteriores."
    exit 1
fi

echo ""
echo "================================================================"
echo "  EJECUCION COMPLETADA"
echo "  Resultados guardados en: resultados/"
echo "================================================================"
echo ""

