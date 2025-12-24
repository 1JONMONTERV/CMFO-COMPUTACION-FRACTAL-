"""
Script de diagnóstico y reparación de CUDA para numba
"""

import sys
import subprocess

def check_cuda():
    """Verifica estado de CUDA"""
    print("=" * 80)
    print("  DIAGNÓSTICO DE CUDA")
    print("=" * 80)
    
    # 1. Verificar numba
    try:
        import numba
        print(f"\n✅ Numba instalado: {numba.__version__}")
    except ImportError:
        print("\n❌ Numba NO instalado")
        return False
    
    # 2. Verificar CUDA en numba
    try:
        from numba import cuda
        is_available = cuda.is_available()
        print(f"✅ numba.cuda importado")
        print(f"   CUDA disponible: {'✅ SÍ' if is_available else '❌ NO'}")
        
        if is_available:
            # Información de GPU
            gpu = cuda.get_current_device()
            print(f"\n📊 Información de GPU:")
            print(f"   Nombre: {gpu.name}")
            print(f"   Compute Capability: {gpu.compute_capability}")
            print(f"   Memoria Total: {gpu.total_memory / (1024**3):.2f} GB")
            return True
        else:
            print("\n⚠️  CUDA no disponible en numba")
            print("   Posibles causas:")
            print("   1. cudatoolkit no instalado")
            print("   2. Versión incompatible de CUDA")
            print("   3. Variables de entorno no configuradas")
            return False
            
    except Exception as e:
        print(f"\n❌ Error al importar numba.cuda: {e}")
        return False


def suggest_fix():
    """Sugiere cómo reparar CUDA"""
    print("\n" + "=" * 80)
    print("  SOLUCIONES SUGERIDAS")
    print("=" * 80)
    
    print("""
1. Instalar cudatoolkit (conda):
   conda install -y cudatoolkit=11.2

2. O reinstalar numba con CUDA:
   pip uninstall numba
   conda install -y numba cudatoolkit=11.2

3. Verificar variables de entorno:
   CUDA_HOME debe apuntar a instalación de CUDA
   PATH debe incluir CUDA bin

4. Verificar nvidia-smi:
   nvidia-smi
   
   Si nvidia-smi funciona, la GPU está OK.
   El problema es la conexión numba ↔ CUDA.
    """)


def main():
    """Función principal"""
    cuda_ok = check_cuda()
    
    if not cuda_ok:
        suggest_fix()
        print("\n" + "=" * 80)
        print("  ⚠️  CUDA NO DISPONIBLE - Usando fallback CPU")
        print("=" * 80 + "\n")
        return 1
    else:
        print("\n" + "=" * 80)
        print("  ✅ CUDA COMPLETAMENTE FUNCIONAL")
        print("=" * 80 + "\n")
        return 0


if __name__ == '__main__':
    sys.exit(main())
