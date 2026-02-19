import networkx as nx

class IntuitiveLayer:
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        # Her düğümün başlangıç aktivasyon enerjisi 0.0
        self.activations = {node: 0.0 for node in self.memory_graph.nodes()}

    def inject_energy(self, node_id, energy=1.0):
        """Kullanıcının odaklandığı kavrama enerji verir."""
        if node_id in self.activations:
            self.activations[node_id] = energy

    def spread_activation(self, decay=0.5, iterations=2):
        """
        Enerjiyi komşu düğümlere yayar. 
        decay: Enerjinin her adımda ne kadar sönümleneceği.
        """
        for _ in range(iterations):
            new_activations = self.activations.copy()
            for node in self.memory_graph.nodes():
                if self.activations[node] > 0:
                    # Komşuları bul (CAUSE, RELY gibi bağlar üzerinden)
                    neighbors = list(self.memory_graph.neighbors(node))
                    if neighbors:
                        # Enerjiyi komşulara bölerek dağıt
                        spread_value = (self.activations[node] * decay) / len(neighbors)
                        for neighbor in neighbors:
                            new_activations[neighbor] += spread_value
            
            # Enerjiyi 1.0 ile sınırla (Saturation)
            self.activations = {n: min(1.0, v) for n, v in new_activations.items()}

    def get_brightest_thoughts(self, threshold=0.1):
        """En çok aktive olan (parlayan) düşünceleri döndürür."""
        sorted_thoughts = sorted(self.activations.items(), key=lambda x: x[1], reverse=True)
        return [t for t in sorted_thoughts if t[1] >= threshold]

# --- Test ---
if __name__ == "__main__":
    from memory.knowledge_graph import CognitiveMemory
    
    # Hafıza oluştur ve bazı bağlar ekle
    mem = CognitiveMemory()
    mem.add_ir_chain([
        {"op": "DEF_ENTITY", "args": ["kahve", "icecek"]},
        {"op": "DEF_ENTITY", "args": ["uyaniklik", "durum"]},
        {"op": "DEF_ENTITY", "args": ["odaklanma", "durum"]},
        {"op": "CAUSE", "args": ["kahve", "uyaniklik"]},
        {"op": "CAUSE", "args": ["uyaniklik", "odaklanma"]}
    ])

    # Sezgisel katmanı başlat
    intuition = IntuitiveLayer(mem.graph)
    
    # Kullanıcı "kahve" dediği an:
    print("Kullanıcı 'kahve' kavramını tetikledi...")
    intuition.inject_energy("kahve", 1.0)
    
    # Enerjiyi yay (Sezgi çalışıyor)
    intuition.spread_activation(decay=0.6, iterations=2)
    
    # Sonuçları gör
    print("Sistemin parlayan düşünceleri (Sezgisel Tahmin):")
    for thought, energy in intuition.get_brightest_thoughts():
        print(f"Düğüm: {thought:12} | Aktivasyon Skoru: {energy:.4f}")