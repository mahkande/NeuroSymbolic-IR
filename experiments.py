import argparse
import networkx as nx
import matplotlib.pyplot as plt
from core.parser import IRParser
from core.validator import CognitiveValidator
from memory.knowledge_graph import CognitiveMemory
from core.logic_engine import LogicEngine
from core.model_bridge import NanbeigeBridge
from memory.conceptnet_service import ConceptNetService
import datetime
import os

class CognitiveLab:
    def __init__(self):
        self.parser = IRParser()
        self.validator = CognitiveValidator()
        self.memory = CognitiveMemory()
        self.z3_engine = LogicEngine()
        self.bridge = NanbeigeBridge()
        self.conceptnet = ConceptNetService(language="tr")
        self.log_lines = []

    def log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.log_lines.append(line)

    def save_log(self, scenario_name):
        with open(f"cognitive_process_{scenario_name}.log", "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines))

    def run_scenario(self, name, steps):
        self.log(f"--- Senaryo: {name} ---")
        for idx, sentence in enumerate(steps):
            self.log(f"Girdi: {sentence}")
            main_entity = sentence.split()[0]
            context = self.memory.get_relevant_context(main_entity)
            # Hızlı yol: ConceptNet
            concept_data = self.conceptnet.query(main_entity)
            ir_chain = self.conceptnet.extract_facts(concept_data)
            if ir_chain:
                used_path = "ConceptNet"
                self.conceptnet.add_to_graph(self.memory.graph, ir_chain)
            else:
                used_path = "Nanbeige"
                isa_schema = self.validator.isa
                llm_raw_response = self.bridge.compile_to_ir(sentence, isa_schema)
                ir_chain = self.parser.parse_raw_output(llm_raw_response)
                if isinstance(ir_chain, dict) and 'error' in ir_chain:
                    self.log(f"LLM Çıktı Hatası: {ir_chain['error']}")
                    continue
            # Z3 ile mantıksal çelişki kontrolü
            is_consistent, logic_msg = self.z3_engine.verify_consistency(ir_chain)
            z3_status = "Tutarlı" if is_consistent else f"Çelişki: {logic_msg}"
            self.log(f"Kullanılan Yol: {used_path}")
            self.log(f"Z3 Durumu: {z3_status}")
            self.log(f"Yeni Eklenen IR'lar: {ir_chain}")
            if not is_consistent:
                self.log(f"[Z3_LOG] UNSAT detected! {logic_msg}")
            self.memory.add_ir_chain(ir_chain)
        # Görselleştirme
        self.visualize_graph(name)
        self.save_log(name)

    def visualize_graph(self, scenario_name):
        plt.figure(figsize=(10, 7))
        G = self.memory.graph
        pos = nx.spring_layout(G, seed=42)
        labels = nx.get_edge_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200, font_size=10, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
        plt.title(f"Cognitive Graph: {scenario_name}")
        plt.tight_layout()
        plt.savefig(f"test_results_{scenario_name}.png")
        plt.close()

# --- Test Senaryoları ---
def sokratik_celiski():
    return [
        "Penguen bir kuştur.",
        "Kuşlar uçar.",
        "Penguen uçar."
    ]

def gizli_baglanti():
    return [
        "Limon bir meyvedir.",
        "Meyveler bitkiseldir.",
        "Limon sarıdır.",
        "Limon ekşidir."
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="sokratik_celiski", help="Çalıştırılacak senaryo adı")
    args = parser.parse_args()
    lab = CognitiveLab()
    scenarios = {
        "sokratik_celiski": sokratik_celiski(),
        "gizli_baglanti": gizli_baglanti(),
    }
    if args.scenario not in scenarios:
        print(f"Bilinmeyen senaryo: {args.scenario}")
        print(f"Mevcut senaryolar: {list(scenarios.keys())}")
    else:
        lab.run_scenario(args.scenario, scenarios[args.scenario])
        print(f"Senaryo tamamlandı. Log ve grafik dosyaları kaydedildi.")
