
import sys
import os
import time
import math
from PIL import Image, ImageDraw, ImageFont

# Importar librería interna
sys.path.insert(0, 'bindings/python')
try:
    from cmfo.graphics.upscale import FractalUpscaler
except ImportError:
    print("Error: No se encontró el módulo cmfo.graphics.upscale")
    sys.exit(1)

def create_test_pattern():
    """Crea una imagen de baja resolución (64x64) con texto para probar nitidez."""
    img = Image.new('RGB', (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    
    # Dibujar patrones geométricos
    d.rectangle([10, 10, 30, 30], outline=(255, 0, 0))
    d.line([0, 0, 64, 64], fill=(0, 255, 0))
    d.text((5, 40), "CMFO", fill=(255, 255, 255))
    
    img.save("test_low_res.png")
    return "test_low_res.png"

def run_demo():
    print("========================================")
    print("   CMFO SUPER-RESOLUCIÓN FRACTAL")
    print("   Solución para Gamers y Artistas")
    print("========================================")
    
    # 1. Crear Input
    input_file = create_test_pattern()
    print(f"[1] Generada imagen de entrada (Baja Res): {input_file}")
    
    # 2. Iniciar Upscaler
    upscaler = FractalUpscaler(intensity=0.15) # Un poco más intenso para demo
    
    start_t = time.time()
    print("[2] Iniciando Escalado Fractal x2...")
    
    # 3. Procesar
    high_res_img = upscaler.upscale(input_file, scale_factor=2)
    
    end_t = time.time()
    
    if high_res_img:
        output_file = "test_fractal_high.png"
        high_res_img.save(output_file)
        print(f"[3] Guardado resultado (Alta Res): {output_file}")
        print(f"    Tiempo de proceso: {end_t - start_t:.4f}s")
        print("\n[CONCLUSIÓN]")
        print("La imagen ha sido reconstruida inyectando 'Ruido Phi'.")
        print("Esto simula texturas realistas donde la interpolación normal se vería borrosa.")
    else:
        print("Error en el proceso de escalado.")

if __name__ == "__main__":
    run_demo()
