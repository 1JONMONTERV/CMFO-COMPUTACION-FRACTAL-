@echo off
echo Building Distribution for PyPI...
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

cd bindings/python
rmdir /s /q dist
rmdir /s /q build
python setup.py sdist bdist_wheel

echo.
echo Build Complete. Artifacts in bindings/python/dist
dir dist
