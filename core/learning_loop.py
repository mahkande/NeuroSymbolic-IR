import json

class LearningLoop:
    def __init__(self, memory, intuition, compiler):
        self.memory = memory
        self.intuition = intuition
        self.compiler = compiler

    def run_cycle(self):
        """Bir öğrenme döngüsü çalıştırır."""
        # 1. En çok parlayan düşünceleri al
        bright_thoughts = self.intuition.get_brightest_thoughts(threshold=0.3)
        if len(bright_thoughts) < 2:
            return "Sentez için yeterli aktivasyon yok."

        # 2. Bu düğümler arasında yeni bir ilişki kurulabilir mi diye LLM'e sor (Synthesis)
        thought_names = [t[0] for t in bright_thoughts]
        synthesis_prompt = f"Şu kavramlar arasında yeni bir mantıksal IR bağı kur: {', '.join(thought_names)}"
        
        # Compiler aracılığıyla yeni IR'lar üret
        new_synthetic_ir = self.compiler.compile_text(synthesis_prompt)
        
        # 3. Yeni üretilen IR'ları 'Validation'dan geçir ve hafızaya ekle
        conflicts, msg = self.memory.find_conflicts_with_history(new_synthetic_ir)
        if not conflicts:
            self.memory.add_ir_chain(new_synthetic_ir)
            return f"Yeni bilgi sentezlendi ve hafızaya eklendi: {new_synthetic_ir}"
        else:
            return f"Sentez reddedildi: {msg}"