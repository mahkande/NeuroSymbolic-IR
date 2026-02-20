# Cognitive OS - Task List (Inference Engine Roadmap)

Bu liste, sistemi tam bir cikarim motoruna donusturmek icin teknik sirayi ve uygulama durumunu izler.

## Faz 1 - Altyapi ve Veri Katmani
- [x] `Graph DB secimi`: JSON yerine birincil graph backend olarak **Neo4j** secildi.
- [x] `Persistence soyutlama`: `memory/graph_store.py` ile backend-agnostic store katmani eklendi.
- [x] `Geri uyum`: Neo4j baglantisi yoksa JSON fallback korunuyor.
- [x] `Pipeline baglantisi`: `main.py` load/save/clear akisi yeni store ile calisiyor.
- [x] `Operasyonel kurulum`: Docker/servis seviyesinde Neo4j ortami standardize edilecek.
- [x] `Veri migrasyonu`: Mevcut `global_graph.json` verisinin Neo4j'e tek seferlik aktarim scripti.

## Faz 2 - Cikarim Motoru Cekirdegi
- [x] `Rule execution engine`: ISA/ATTR/WANT/BELIEVE/GOAL/TEMPORAL kurallari icin deterministic kural yurutucu.
- [x] `Truth maintenance`: Celisen bilgide zaman damgasi + confidence + kaynak onceligiyle karar mekanizmasi.
- [x] `Abductive layer`: Sonuctan muhtemel neden uretimi ve puanlama.
- [x] `Bidirectional reasoning`: Forward + inverse reasoning zincirlerinin kaydi.

## Faz 3 - Hibrit Semantik Siniflandirma
- [x] `Rule-based extractor v2`: Dil kaliplari ve baglacsal sinyallerle opcode kapsami arttirma.
- [x] `Classifier layer`: Relation multi-label siniflandirici (opcode olasiliklari).
- [x] `LLM arbitration`: Belirsiz baglarda LLM sadece hakem olarak cagirilsin.
- [x] `Confidence fusion`: Rule + classifier + LLM skorlarinin birlestirilmesi.

## Faz 4 - Kalite, Olcum ve Izlenebilirlik
- [x] `Edge diversity metric`: entropy, coverage, dominance ratio metrikleri.
- [x] `Per-opcode kalite`: opcode bazli precision/recall izleme.
- [x] `Drift alarmi`: `CAUSE` asiri baskinliginda otomatik uyari.
- [x] `Provenance`: Her edge icin kaynak, zaman, confidence, inference_rule kaydi.

## Faz 5 - Vektor Katmani ve LLM Dogrulama
- [x] `Vector DB secimi`: Qdrant/Weaviate/PgVector arasindan secim ve PoC.
- [x] `Graph->embedding`: node/edge/subgraph embedding pipeline'i.
- [x] `Retriever`: LLM'in graph + vector hafizadan birlikte kanit cekmesi.
- [x] `Deterministic verifier gate`: LLM sonucunu graph kurallari ve SMT ile final dogrulama.

## Faz 6 - Uretim Hazirlik
- [x] `Load testing`: Uzun metin, paralel istek, listener senaryolari.
- [x] `Backfill jobs`: Arka plan inference batch isleri.
- [x] `Versioning`: ISA versiyonlama + migration politikalari.
- [x] `Release checklist`: test, benchmark, rollback planlari.

## Faz 7 - Reasoning Kalite Sertlestirme (Adim Adim)
### Adim 1 - Inference Noise Kontrolu
- [x] `Transitive guard`: `INFERRED` edge uretiminde opcode whitelist + semantic type check + max branching limiti.
- [x] `Low-value pruning`: self-loop, anlamsiz token ve tekrarli edge baglari icin prune kurallari.
- [x] `Confidence policy`: inference edge'lerinde min confidence threshold ve dynamic penalty.

### Adim 2 - Gold Dataset ve Gercek Degerlendirme
- [x] `Gold set v1`: en az 300-500 etiketli cumle (opcode + arguman truth).
- [x] `Eval pipeline`: dataset uzerinde opcode precision/recall/F1 olcumu.
- [x] `Regression suite`: her degisiklikte otomatik kalite karsilastirma raporu.

### Adim 3 - Verifier Gate Sertlestirme
- [x] `Strict gate mode`: low-confidence IR reddi + provenance zorunlulugu.
- [x] `Conflict risk score`: yazimdan once celiski olasiligi puani ve reject/escalate politikasi.
- [x] `Policy profiles`: `safe`, `balanced`, `aggressive` verifier profilleri.

### Adim 4 - Retriever Ablation ve Etki Analizi
- [x] `Ablation runner`: `vector-only`, `graph-only`, `hybrid` modlarini ayni setle test et.
- [x] `KPI raporu`: kalite (P/R/F1) + latency + token maliyeti karsilastirma.
- [x] `Default strategy`: sonuc metriklerine gore production varsayilan retrieval modunu sec.

### Adim 5 - CI/CD Release Gate
- [x] `CI workflow`: `compile + release_check + load_test_smoke + eval_pipeline` zorunlu.
- [x] `Threshold gates`: belirlenen kalite/latency esiklerinin altinda release bloklansin.
- [x] `Canary rollout`: sinirli trafik + otomatik rollback kosullari.

## Bu Turde Yapilanlar
- `Tamamlandi`: Faz 2 cekirdegi (rule engine + truth maintenance + abductive + bidirectional trace) tamamlandi.
- `Tamamlandi`: Faz 3 hibrit semantik katmani (rule-based extractor v2 + classifier + LLM arbitration + confidence fusion) entegre edildi.
- `Tamamlandi`: Faz 4 kalite/olcum/izlenebilirlik (diversity metrikleri + opcode kalite proxy + drift alarm + provenance) entegre edildi.
- `Tamamlandi`: Faz 5 vector layer + graph->embedding + hybrid retriever + deterministic verifier gate entegre edildi.
- `Tamamlandi`: Faz 6 uretim hazirlik (load test + backfill batch jobs + ISA versioning/migration + release checklist) tamamlandi.
- `Siradaki is`: Faz 7 tamamlandi; yeni faz planlanacak.
