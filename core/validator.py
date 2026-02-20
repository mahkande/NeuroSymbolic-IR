import json
import os
from core.isa_versioning import get_active_isa_path, get_active_version

class CognitiveValidator:
    def __init__(self, isa_path=None):
        self.isa_path = isa_path or get_active_isa_path()
        self.isa_version = get_active_version()
        self.isa = self.load_isa()
        self.opcodes = self.flatten_opcodes()

    def load_isa(self):
        """JSON dosyasını yükler."""
        if not os.path.exists(self.isa_path):
            raise FileNotFoundError(f"Hata: {self.isa_path} bulunamadı!")
        with open(self.isa_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

    def flatten_opcodes(self):
        """Kategoriler altındaki opcode'ları hızlı erişim için düz bir sözlüğe çevirir."""
        flat_list = {}
        for cat, content in self.isa['categories'].items():
            for op, details in content['opcodes'].items():
                flat_list[op] = details
        return flat_list

    def validate_instruction(self, op, args):
        """Tek bir IR komutunun sözdizimini kontrol eder."""
        if op not in self.opcodes:
            return False, f"Geçersiz Opcode: {op}"
        
        expected_args_count = len(self.opcodes[op]['args'])
        if len(args) != expected_args_count:
            return False, f"{op} için beklenen argüman sayısı {expected_args_count}, alınan {len(args)}"
        
        return True, "Geçerli"

    def check_logical_conflicts(self, ir_chain):
        """
        IR zinciri içindeki temel mantıksal çelişkileri denetler.
        Örnek: Aynı anda hem CAUSE hem OPPOSE bağı varsa hata verir.
        """
        causes = set()
        opposes = set()

        for instr in ir_chain:
            op = instr.get("op")
            args = instr.get("args")

            if op == "CAUSE":
                # (A, B) çiftini kaydet
                causes.add(tuple(args))
            elif op == "OPPOSE":
                # (A, B) çiftini kaydet (sırasız karşılaştırma için sıralayalım)
                opposes.add(tuple(sorted(args)))

        # Çelişki kontrolü
        for cause in causes:
            if tuple(sorted(cause)) in opposes:
                return False, f"Mantıksal Çelişki: {cause[0]} hem {cause[1]}'e sebep oluyor hem de onunla çelişiyor!"

        return True, "Mantıksal tutarlılık onaylandı."

# --- Test Alanı ---
if __name__ == "__main__":
    validator = CognitiveValidator()
    
    # Örnek bir IR zinciri (LLM'den gelmiş gibi simüle ediyoruz)
    sample_ir = [
        {"op": "DEF_ENTITY", "args": ["yagmur", "hava_durumu"]},
        {"op": "DEF_ENTITY", "args": ["piknik", "aktivite"]},
        {"op": "CAUSE", "args": ["yagmur", "piknik"]},
        {"op": "OPPOSE", "args": ["yagmur", "piknik"]}  # Bu bir çelişki yaratacak
    ]

    print("--- Sözdizimi Kontrolü ---")
    for instr in sample_ir:
        valid, msg = validator.validate_instruction(instr['op'], instr['args'])
        print(f"{instr['op']}: {valid} ({msg})")

    print("\n--- Mantıksal Denetim ---")
    consistent, logic_msg = validator.check_logical_conflicts(sample_ir)
    print(f"Sonuç: {consistent} | Mesaj: {logic_msg}")
