# NeuroGraph OS

ISA Tabanli Bilissel IR (Intermediate Representation) derleyicisi ve canli bilgi grafigi sistemi.

Bu proje, dogal dil metnini opcode-temelli bir bilissel temsile cevirir ve sonuc IR zincirini node+edge grafigine isler.
Amaç; sadece nedensellik (`CAUSE`) degil, ontolojik/epistemik/teleolojik/zamansal baglari da birlikte modellemektir.

## Ne Yapar?
- Metni ISA semasina uygun IR komutlarina derler.
- IR komutlarini kalici graph hafizasina yazar.
- Tutarlilik kontrolu uygular (guvenli mod + opsiyonel native Z3).
- Dashboard uzerinden canli graph, opcode dagilimi ve sistem olaylarini gosterir.
- Shadow/listener boru hattiyla dosya/oturum tabanli veri beslemesini destekler.

## IR Kategorileri
- `ONTOLOGICAL`: `DEF_ENTITY`, `DEF_CONCEPT`, `ISA`, `EQUIV`, `ATTR`
- `EPISTEMIC`: `KNOW`, `BELIEVE`, `DOUBT`, `WONDER`, `ASSUME`
- `TELEOLOGICAL`: `WANT`, `AVOID`, `GOAL`, `INTEND`, `EVAL`, `DO`
- `CAUSAL_LOGIC`: `CAUSE`, `PREVENT`, `IMPLY`, `OPPOSE`, `TRIGGER`
- `DEONTIC`: `MUST`, `MAY`, `FORBID`, `CAN`
- `TEMPORAL`: `BEFORE`, `WHILE`, `START`, `END`
- `META_COGNITIVE`: `REFLECT`, `CORRECT`, `ANALOGY`

ISA semasi: `spec/isa_v1.json`

## Mimari Ozet
- `core/model_bridge.py`: LLM istemcisi, prompt, fallback derleme
- `core/nlp_utils.py`: normalizasyon, grammar filter, relation rebalance
- `core/logic_engine.py`: tutarlilik kontrolu
- `memory/knowledge_graph.py`: graph yazim/okuma kurallari
- `main.py`: ana boru hatti (chunking, cache, validation, persistence)
- `ui/dashboard.py`: canli kontrol paneli

## Kurulum (Lokal)
1. Python 3.11+ ortami hazirla.
2. Sanal ortam olustur ve aktif et:
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
3. Bagimliliklari yukle:
```bash
pip install -U networkx streamlit streamlit-agraph watchdog groq z3-solver zeyrek nltk
```
4. Ortam degiskenlerini ayarla:
```bash
set GROQ_API_KEY=YOUR_KEY
set GROQ_MODEL=llama-3.3-70b-versatile
```

## Dashboard Calistirma
```bash
streamlit run ui/dashboard.py
```

Dashboard acildiginda:
- Sol panelden `Clear Memory` ve `Clear IR Cache` ile temiz baslangic yapabilirsin.
- Metni input alanina yapistirip Enter ile IR->Graph akisina girebilirsin.

## CLI Calistirma (Alternatif)
```bash
python main.py
```

## GitHub'a Yukleme
Bu proje klasorunden:
```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: NeuroGraph OS"
git remote add origin https://github.com/mahkande/NeuroSymbolic-IR.git
git push -u origin main
```

Eger remote zaten varsa:
```bash
git remote set-url origin https://github.com/mahkande/NeuroSymbolic-IR.git
git push -u origin main
```

## Tasarim Notlari
- Uzun metinler chunk edilerek islenir; tek parca sinirlari azaltilmistir.
- Graph, ayni node cifti icin coklu opcode saklayacak sekilde yapilandirilmistir.
- Runtime gurdultusunu azaltmak icin shadow/bridge kanallarinda filtreleme vardir.

## Guvenlik ve Gizlilik
- API anahtarlarini kodda hardcode etmek yerine ortam degiskeni kullanin.
- Runtime graph/log/cache dosyalari `.gitignore` ile repo disinda tutulmalidir.

## Yol Haritasi
- Opcode basina precision/recall metrik dashboard'u
- Alt-graf kalite skorlari ve otomatik alarm esikleri
- Dataset tabanli relation classifier egitimi (hybrid pipeline)
