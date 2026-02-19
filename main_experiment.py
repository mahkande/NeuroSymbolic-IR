# main_experiment.py
from core.ollama_compiler import OllamaCompiler
from memory.knowledge_graph import CognitiveMemory
from memory.probabilistic_layer import IntuitiveLayer

# 1. Belleği Başlat ve Bilgileri Yükle
mem = CognitiveMemory()
mem.add_ir_chain([
    {"op": "CAUSE", "args": ["magnezyum_eksikligi", "kas_krampi"]},
    {"op": "ATTR", "args": ["ispanak", "icerik", "magnezyum"]}
])

# 2. Sezgisel Katmanı Çalıştır
intuition = IntuitiveLayer(mem.graph)
intuition.inject_energy("ispanak", 1.0)
intuition.inject_energy("kas_krampi", 1.0)
intuition.spread_activation()

# 3. Parlayan düğümleri yakala ve Sentezle
bright = intuition.get_brightest_thoughts()
if len(bright) >= 2:
    compiler = OllamaCompiler()
    # LLM'den yeni IR sentezlemesi iste
    new_ir_json = compiler.synthesize(bright[0][0], bright[1][0], "isa_v1.json")
    print(f"Sentezlenen Yeni IR: {new_ir_json}")

    # 4. Z3 ile Doğrula ve Hafızaya Kaydet (Learning Loop Tamamlandı)