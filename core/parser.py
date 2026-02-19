import json
import re

class IRParser:
    def __init__(self):
        # JSON bloklarını metin içinden çekmek için Regex
        self.json_pattern = re.compile(r'\[\s*\{.*\}\s*\]', re.DOTALL)

    def parse_raw_output(self, llm_output):
        """
        LLM'den gelen ham metni alır, içindeki ilk geçerli JSON dizi bloğunu bulur
        ve Python listesine çevirir. Fazla veya bozuk veri varsa ilkini döndürür.
        """
        import re
        if isinstance(llm_output, list):
            return llm_output
        if isinstance(llm_output, dict):
            return llm_output
        # Agresif regex: ilk [ ve son ] arası
        match = re.search(r'(\[.*?\])', llm_output, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return {"error": f"JSON Çözümleme Hatası: {str(e)}", "raw": json_str}
        # JSON bulunamazsa boş dict döndür
        return {"error": "Metin içerisinde geçerli bir IR yapısı bulunamadı.", "raw": llm_output}

    def stringify_ir(self, ir_chain):
        """IR zincirini okunabilir bir string formatına çevirir (Debugging için)."""
        lines = []
        for i, instr in enumerate(ir_chain):
            op = instr.get("op", "UNKNOWN")
            args = ", ".join(instr.get("args", []))
            lines.append(f"{i+1}. {op}({args})")
        return "\n".join(lines)

# --- Test ---
if __name__ == "__main__":
    parser = IRParser()
    
    # LLM'in bazen yaptığı gibi 'gevezelik' içeren bir çıktı simülasyonu
    dirty_output = """
    İşte istediğin derleme sonuçları:
    ```json
    [
      {"op": "DEF_ENTITY", "args": ["kahve", "icecek"]},
      {"op": "WANT", "args": ["user", "kahve"]}
    ]
    ```
    Umarım bu işine yarar!
    """
    
    clean_ir = parser.parse_raw_output(dirty_output)
    print("Temizlenmiş IR Dizisi:")
    print(clean_ir)
    print("\nOkunabilir Format:")
    print(parser.stringify_ir(clean_ir))