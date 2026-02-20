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

## Faz 8 - Fallback Graph Quality Hardening
### Adim 1 - Merkezi Semantik Fallback
- [x] `Semantic fallback motoru`: `core/fallback_rules.py` ile ISA/ATTR/CAUSE/GOAL/BEFORE/OPPOSE/PREVENT oncelikli fallback.
- [x] `DO son care`: fallback'te DO'nun sadece yuksek degerli iliski cikmiyorsa uretilmesi.
- [x] `Deterministik merge`: fallback ciktilarinda duplicate temizligi (`merge_fallback`).

### Adim 2 - Pipeline Entegrasyonu
- [x] `Model bridge`: `template_fallback` merkezi fallback motoruna baglandi.
- [x] `Grammar filter`: POS blok ve unknown-opcode akislari semantik fallback ile guclendirildi.
- [x] `Shadow listener`: rule-based extractor + semantic fallback birlesik akis.
- [x] `Main pipeline`: compile/parse hatalarinda chunk bazli fallback ve tum metin fallback.
- [x] `File listener`: compile/parse/empty IR durumlarinda semantik fallback.

### Adim 3 - Kalite Dengeleme
- [x] `Classifier fallback revizyonu`: sinyal yoksa otomatik DO puani kaldirildi.
- [x] `Fallback KPI`: fallback opcode dagilimi ve faydali-edge orani dashboard metrikleri.
- [x] `Gold eval`: fallback-only senaryolari icin precision/recall olcumu.

### Adim 4 - Semantic Validation ve Rule Bootstrap
- [x] `ISA validation constraints`: `spec/isa_v1_1.json` icine opcode bazli `arg_types` ve `distinct_pairs` kurallari eklendi.
- [x] `Validator hardening`: `core/validator.py` tip/range/coreference kontrolleri ile guclendirildi.
- [x] `Template rule engine`: `args_template` ile regex capture-group tabanli kural uygulama destegi eklendi.
- [x] `Human-in-the-loop`: LLM uretilen rule adaylari `require_human` ile kuyruga alinip auto-review disina cikarildi.
- [x] `DO-only bootstrap`: Listener/main/file-listener akisinda DO-only durumda LLM rule adayi + clarifying question uretimi eklendi.

## Faz 9 - Intent Graph + Backward Verifier (Evidence-First)
### Adim 1 - Kanit Odakli Cikti Sozlesmesi
- [x] `Strict JSON schema`: LLM ciktilari tek bir katki semasina zorlanacak (opcode, args, confidence, source_span, evidence_ids).
- [x] `Schema reject`: semaya uymayan cikti parser'da ayiklanmak yerine direkt reddedilecek.
- [x] `Unsupported-claim blokaji`: evidence bagi olmayan claim yazimi engellenecek.

### Adim 2 - Kanit Grafigi ve Geriye Dogru Saglama
- [x] `Proof object`: her claim icin `claim -> evidence -> rule -> verdict` zinciri tutulacak.
- [x] `Backward verifier`: yeni edge eklenmeden once claim'den geriye dogru kanit yurutumu zorunlu olacak.
- [x] `Trace persistence`: reasoning trace kayitlari edge-level provenance ile birlestirilecek.

### Adim 3 - Retriever Kalite Artisi (Jina)
- [x] `Embedding upgrade`: `hash_embed` yerine `jina-embeddings-v3` tabanli semantik embedding pipeline'i.
- [x] `Reranker`: retrieval sonuclarinda `jina-reranker` ile yeniden siralama.
- [x] `Retriever v2`: sabit graph score yerine edge confidence/provenance/path-depth tabanli dinamik puanlama.
- [x] `Evidence recall eval`: kanit geri cagirimi icin top-k recall ve MRR metriklerinin eklenmesi.

### Adim 4 - Mantik Motoru Sertlestirme
- [x] `Native Z3 profile`: `COGNITIVE_USE_Z3=1` icin uretim profili, timeout ve watchdog guardlari.
- [x] `Constraint expansion`: temporal/deontic/causal celiskiler icin opcode-bazli SMT kisitlari.
- [x] `Counterexample raporu`: verifier reddinde minimal celiski cekirdegi + duzeltme onerisi uretimi.

### Adim 5 - Endustri Seviyesi Guvence ve Olcum
- [x] `Quality gates`: unsupported-claim rate, contradiction rate, evidence coverage esikleri.
- [x] `Offline benchmark`: fallback-only, llm-only, hybrid ve backward-verifier modlarinin karsilastirmasi.
- [x] `Release policy`: esik alti performansta otomatik rollback/escalation kurallari.

## Bu Turde Yapilanlar
- `Tamamlandi`: Faz 2 cekirdegi (rule engine + truth maintenance + abductive + bidirectional trace) tamamlandi.
- `Tamamlandi`: Faz 3 hibrit semantik katmani (rule-based extractor v2 + classifier + LLM arbitration + confidence fusion) entegre edildi.
- `Tamamlandi`: Faz 4 kalite/olcum/izlenebilirlik (diversity metrikleri + opcode kalite proxy + drift alarm + provenance) entegre edildi.
- `Tamamlandi`: Faz 5 vector layer + graph->embedding + hybrid retriever + deterministic verifier gate entegre edildi.
- `Tamamlandi`: Faz 6 uretim hazirlik (load test + backfill batch jobs + ISA versioning/migration + release checklist) tamamlandi.
- `Tamamlandi`: Faz 8 fallback graph quality hardening (KPI + gold eval adimlari tamamlandi).
- `Backlog eklendi`: Faz 9 intent-graph + backward verifier yol haritasi (Jina v3 embedding + Jina reranker dahil).
- `Tamamlandi`: Faz 9 tamamlandi (strict schema + backward verifier + Jina retrieval + evidence eval + Z3 profile + quality/release gates).
