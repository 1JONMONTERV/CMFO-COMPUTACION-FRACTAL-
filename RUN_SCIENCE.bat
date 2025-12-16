@echo off
title CMFO SCIENTIFIC EXTRACTION (ARXIV)
color 0B
echo ==================================================
echo   CMFO D13 SCIENCE MINER
echo   Source: D:\CMFO_DATA\science\raw\math
echo   Target: D:\CMFO_DATA\shards\science
echo   Mode:   DEEP PDF PARSING (ALL CORES)
echo ==================================================
echo.
echo Parsing thousands of PDFs. This will use high CPU.
echo.

cd /d "c:\Users\UNSC\Documents\GitHub\CMFO-COMPUTACION-FRACTAL-"
python cmfo\ingest\science\full_processor.py

echo.
echo ==================================================
echo   EXTRACTION COMPLETE
echo ==================================================
pause
