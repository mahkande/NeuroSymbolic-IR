import openai # Veya kullanmak istediğin herhangi bir LLM kütüphanesi
from core.validator import CognitiveValidator

class CognitiveCompiler:
    def __init__(self, api_key):
        self.validator = CognitiveValidator()
        self.api_key = api_key

    def compile_text(self, text):
        # 1. LLM'e yukarıdaki prompt ile metni gönder (Burada simüle ediyoruz)
        # Gerçek kullanımda openai.ChatCompletion.create(...) kullanılacak.
        print(f"Metin Derleniyor: {text}")
        
        # Simüle edilmiş LLM çıktısı:
        raw_ir = [
            {"op": "ATTR", "args": ["self", "state", "yorgun"]},
            {"op": "MUST", "args": ["proje_bitirme"]},
            {"op": "INTEND", "args": ["self", "kahve_icme"]}
        ]

        # 2. Validator ile doğrula
        for instr in raw_ir:
            is_valid, msg = self.validator.validate_instruction(instr['op'], instr['args'])
            if not is_valid:
                raise ValueError(f"Derleme Hatası: {msg}")

        # 3. Mantıksal Çelişki Kontrolü
        is_consistent, logic_msg = self.validator.check_logical_conflicts(raw_ir)
        if not is_consistent:
            print(f"Uyarı: {logic_msg}")

        return raw_ir

# Test
if __name__ == "__main__":
    compiler = CognitiveCompiler(api_key="sk-...")
    ir_output = compiler.compile_text("Çok yorgunum ama çalışmalıyım.")
    print("Derleme Başarılı. Üretilen IR:", ir_output)