import pytest
from PIL import Image
from models.blip_model import extract_caption

@pytest.mark.slow
def test_extract_caption_returns_text():
    # Crea immagine dummy rossa (valida per test)
    test_img = Image.new("RGB", (224, 224), color="red")
    
    # Esegue la funzione
    caption = extract_caption(test_img)
    
    # Verifica che restituisca una stringa non vuota
    assert isinstance(caption, str)
    assert len(caption.strip()) > 0
