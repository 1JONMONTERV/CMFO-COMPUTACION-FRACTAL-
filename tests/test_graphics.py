
import pytest
import os
from PIL import Image
import sys

# Ensure bindings are in path
sys.path.insert(0, os.path.abspath('bindings/python'))

try:
    from cmfo.graphics.upscale import FractalUpscaler
except ImportError:
    pytest.skip("CMFO Graphics module not installed", allow_module_level=True)

class TestFractalGraphics:
    
    @pytest.fixture
    def upscaler(self):
        return FractalUpscaler(intensity=0.1)

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Creates a temporary 32x32 red image"""
        img_path = tmp_path / "test_img.png"
        img = Image.new('RGB', (32, 32), color='red')
        img.save(img_path)
        return str(img_path)

    def test_initialization(self, upscaler):
        assert upscaler.intensity == 0.1
        
    def test_upscale_dimensions(self, upscaler, temp_image):
        """Output should be exactly 2x the input dimensions"""
        res_img = upscaler.upscale(temp_image, scale_factor=2)
        assert res_img is not None
        assert res_img.size == (64, 64)
        
    def test_upscale_missing_file(self, upscaler):
        """Should return None or handle error gracefully for missing file"""
        res = upscaler.upscale("non_existent_ghost_file.png")
        assert res is None, "Should fail gracefully on missing file"

    def test_upscale_integrity(self, upscaler, temp_image):
        """Output should still be an RGB image"""
        res_img = upscaler.upscale(temp_image)
        assert res_img.mode == 'RGB'
        
    def test_unsupported_scale(self, upscaler, temp_image, capsys):
        """Should warn about non-x2 scaling but currently proceed/handle it"""
        # Note: Current implementation might just print a warning and likely fail or fallback
        # based on my code reading. Let's check if it returns something or fails.
        # Actually in the code: it creates x_map based on new_w which IS scaled.
        # So x4 might fail if dimensions don't align with tile logic?
        # Let's just create the test to ensure it runs without crashing.
        res_img = upscaler.upscale(temp_image, scale_factor=4)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
