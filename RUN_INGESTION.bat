@echo off
title CMFO INDUSTRIAL INGESTION (PARALLEL)
color 0A
echo ==================================================
echo   CMFO D9 MASS INGESTION ENGINE
echo   Source: D:\CMFO_DATA\ontology\raw_source
echo   Mode:   ALL CORES (FULL THROTTLE)
echo ==================================================
echo.
echo Starting Ingestion... The usage of D: drive will increase.
echo You can minimize this window.
echo.

cd /d "c:\Users\UNSC\Documents\GitHub\CMFO-COMPUTACION-FRACTAL-"
python cmfo\ingest\parallel_ingestor.py

echo.
echo ==================================================
echo   INGESTION COMPLETE
echo ==================================================
pause
