import importlib
import json
import os
import re
import subprocess
import sys
import urllib.request

from core.nlp_utils import normalize_and_match
from core.fallback_rules import semantic_fallback_ir


PROVIDER_SPECS = {
    "local": {"label": "Local (Ollama)", "package": None, "env_key": None},
    "groq": {"label": "Groq", "package": "groq", "env_key": "GROQ_API_KEY"},
    "openai": {"label": "OpenAI", "package": "openai", "env_key": "OPENAI_API_KEY"},
    "anthropic": {"label": "Anthropic", "package": "anthropic", "env_key": "ANTHROPIC_API_KEY"},
    "google": {"label": "Google Gemini", "package": "google-generativeai", "env_key": "GEMINI_API_KEY"},
    "mistral": {"label": "Mistral", "package": "mistralai", "env_key": "MISTRAL_API_KEY"},
    "together": {"label": "Together", "package": "together", "env_key": "TOGETHER_API_KEY"},
}


def ensure_provider_client(provider: str, auto_install: bool = True):
    provider = (provider or "").strip().lower()
    spec = PROVIDER_SPECS.get(provider, {})
    pkg = spec.get("package")
    if not pkg:
        return True, "ok"
    try:
        importlib.import_module("google.generativeai" if pkg == "google-generativeai" else pkg)
        return True, "ok"
    except Exception:
        if not auto_install:
            return False, f"{pkg} missing"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "installed"
    except Exception as exc:
        return False, str(exc)


class NanbeigeBridge:
    # Backward compatibility: older modules reference DEFAULT_MODEL directly.
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    # Backward compatibility: older modules reference DEFAULT_API_KEY directly.
    DEFAULT_API_KEY = ""
    DEFAULT_MODELS = {
        "local": "llama3.1",
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "google": "gemini-1.5-flash",
        "mistral": "mistral-large-latest",
        "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    }
    FALLBACK_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model_name=None, api_key=None, provider=None):
        self.provider = (provider or os.getenv("COGNITIVE_LLM_PROVIDER", "groq")).strip().lower()
        if self.provider not in PROVIDER_SPECS:
            self.provider = "groq"
        self.model_name = model_name or os.getenv("COGNITIVE_LLM_MODEL") or self.DEFAULT_MODELS.get(self.provider, self.DEFAULT_MODELS["groq"])
        spec = PROVIDER_SPECS[self.provider]
        env_key = spec.get("env_key")
        self.api_key = api_key or os.getenv("COGNITIVE_LLM_API_KEY") or (os.getenv(env_key) if env_key else "")
        self.client = None
        self._init_error = ""

        auto_install = os.getenv("COGNITIVE_AUTO_INSTALL_CLIENTS", "1").strip().lower() in {"1", "true", "yes", "on"}
        ok, msg = ensure_provider_client(self.provider, auto_install=auto_install)
        if not ok:
            self._init_error = f"Client init failed: {msg}"
            return
        if self.provider != "local" and not self.api_key:
            self._init_error = f"{env_key or 'API_KEY'} tanimli degil."
            return
        self._init_client()

    def _init_client(self):
        try:
            if self.provider == "groq":
                from groq import Groq

                self.client = Groq(api_key=self.api_key)
            elif self.provider == "openai":
                from openai import OpenAI

                base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
                self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            elif self.provider == "anthropic":
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            elif self.provider == "google":
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self.client = genai
            elif self.provider == "mistral":
                from mistralai import Mistral

                self.client = Mistral(api_key=self.api_key)
            elif self.provider == "together":
                from together import Together

                self.client = Together(api_key=self.api_key)
            else:
                self.client = "local"
        except Exception as exc:
            self._init_error = str(exc)

    def _build_prompt(self, user_text, isa_schema, hints=None, error_msg=None, normalization_note=None):
        isa_str = json.dumps(isa_schema, ensure_ascii=False, indent=2)
        examples = [
            ("Kedi bir memelidir", '[{"op": "ISA", "args": ["kedi", "memeli"]}]'),
            ("Elif cesurdur", '[{"op": "ATTR", "args": ["elif", "ozellik", "cesur"]}]'),
            ("Elif kuleye girmek istiyor", '[{"op": "WANT", "args": ["elif", "kuleye_girmek"]}]'),
            ("Elif once kapiyi acti sonra iceri girdi", '[{"op": "BEFORE", "args": ["kapiyi_acmak", "iceri_girmek"]}]'),
        ]

        prompt = "### Sistem Mesaji:\n"
        prompt += "Sen bir Bilissel IR Derleyicisisin.\n"
        prompt += "Once cumleyi duzgun Turkceye normalize et, sonra kok (lemma/stem) hallerini kullan.\n"
        prompt += "Noktalama ve gereksiz ekleri temizleyerek yalnizca anlamli kok kavramlarla IR uret.\n"
        prompt += "Asagidaki ISA semasi, gecerli opcode ve argumanlari tanimlar.\n"
        prompt += f"ISA Semasi:\n{isa_str}\n\n"
        if hints:
            prompt += f"HINTS: {hints}\n"
        if normalization_note:
            prompt += f"NORMALIZATION: {normalization_note}\n"

        prompt += "### Ornekler (sadece format icin):\n"
        for ex_in, ex_out in examples:
            prompt += f"Girdi: {ex_in}\nCikti: {ex_out}\n"

        prompt += (
            "\nDikkat: Ornekler sadece format icindir.\n"
            "Sadece kullanicinin girdisinden gelen kavramlari kullan.\n"
            "Ciktida sadece JSON formatinda bir IR listesi uret.\n"
            "Her satir su semaya uymali: {\"op\": \"OPCODE\", \"args\": [\"a\", \"b\"], "
            "\"confidence\": 0.0-1.0(optional), \"source_span\": \"metin parcasi\"(optional), "
            "\"evidence_ids\": [\"ev::0\"](optional)}.\n"
            "Schema disi ekstra alan ekleme.\n"
            "Her arguman girdi metnindeki bir kavramdan gelmeli.\n"
            "Gereksiz kelimeleri (bir, ve, ile, icin vb.) kullanma.\n"
            "CAUSE yalnizca acik nedensellik sinyali varsa (cunku, dolayisiyla, -se/-sa) kullan.\n"
        )
        prompt += f"\n### Komut:\nGirdi: {user_text}\nCikti: "
        prompt += "\nDIKKAT: Sadece JSON listesini dondur. Aciklama yapma. Yanit '[' ile baslamali ve ']' ile bitmeli."
        if error_msg:
            prompt += f"\nHata: {error_msg}\nLutfen semaya sadik kalarak tekrar derle."
        return prompt

    @staticmethod
    def _extract_json_array(text):
        match = re.search(r"(\[.*?\])", text or "", re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def _token_budget(text: str, min_tokens: int = 220, max_tokens: int = 1200) -> int:
        wc = len((text or "").split())
        budget = min_tokens + int(wc * 1.5)
        return max(min_tokens, min(max_tokens, budget))

    @staticmethod
    def _dashboard_alert(message):
        return f"[DASHBOARD_ALERT] {message}"

    def _map_api_error(self, exc):
        msg = str(exc).lower()
        if "quota" in msg or "rate limit" in msg or "insufficient_quota" in msg:
            return self._dashboard_alert("API kotasi/hiz limiti doldu.")
        if any(token in msg for token in ["connection", "timeout", "network", "dns", "unavailable", "reset", "refused"]):
            return self._dashboard_alert("API baglantisi koptu veya servis gecici olarak kullanilamaz.")
        return None

    def _chat_local_ollama(self, prompt, max_tokens=256):
        base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"
        payload = {
            "model": self.model_name or self.DEFAULT_MODELS["local"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            row = json.loads(raw or "{}")
            return row.get("response", "") or ""

    def _chat(self, prompt, max_tokens=160):
        if self._init_error:
            raise RuntimeError(self._init_error)
        if self.provider == "local":
            return self._chat_local_ollama(prompt, max_tokens=max_tokens)
        if self.client is None:
            raise RuntimeError("API client initialized degil.")

        if self.provider == "groq":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content if completion.choices else ""
        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content if completion.choices else ""
        if self.provider == "anthropic":
            completion = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = completion.content or []
            return content[0].text if content else ""
        if self.provider == "google":
            model = self.client.GenerativeModel(self.model_name)
            completion = model.generate_content(prompt)
            return getattr(completion, "text", "") or ""
        if self.provider == "mistral":
            completion = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content if completion.choices else ""
        if self.provider == "together":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content if completion.choices else ""
        raise RuntimeError(f"Desteklenmeyen provider: {self.provider}")

    def _normalize_user_text(self, user_text, memory_terms=None):
        normalized = normalize_and_match(user_text, memory_terms=memory_terms, threshold=0.90)
        stopwords = {"bir", "ve", "ile", "icin", "ama", "fakat", "de", "da", "ki", "bu", "su", "o", "cok", "az", "en", "gibi", "olan", "olanlar"}
        filtered_tokens = [t for t in normalized["corrected_tokens"] if t not in stopwords]
        normalized["filtered_tokens"] = filtered_tokens
        normalized["filtered_text"] = " ".join(filtered_tokens).strip()
        return normalized

    def compile_to_ir(self, user_text, isa_schema, conceptnet_hints=None, max_retries=2, memory_terms=None):
        norm = self._normalize_user_text(user_text, memory_terms=memory_terms)
        normalized_text = norm["filtered_text"] or norm["corrected_text"] or norm["cleaned"]
        correction_note = None
        if norm["corrections"]:
            parts = [f"{c['from']} -> {c['to']} ({int(c['confidence'] * 100)}%)" for c in norm["corrections"]]
            correction_note = "; ".join(parts)
        prompt = self._build_prompt(normalized_text, isa_schema, hints=conceptnet_hints, normalization_note=correction_note)

        allowed_words = set(norm["tokens"]) | set(norm["lemmas"]) | set(norm["corrected_tokens"]) | set(norm["filtered_tokens"])
        retries = 0
        while retries < max_retries:
            try:
                result = self._chat(prompt, max_tokens=self._token_budget(normalized_text))
                json_str = self._extract_json_array(result)
                if not json_str:
                    retries += 1
                    continue
                try:
                    ir_chain = json.loads(json_str)
                    for instr in ir_chain:
                        for arg in instr.get("args", []):
                            if isinstance(arg, str) and arg.lower() not in allowed_words:
                                raise ValueError(f"IR icindeki '{arg}' girdi kavramlariyla eslesmiyor.")
                    return json_str
                except Exception:
                    retries += 1
            except Exception as exc:
                mapped = self._map_api_error(exc)
                if mapped:
                    return {"error": mapped}
                retries += 1

        ir_templates = self.template_fallback(normalized_text)
        if ir_templates:
            return ir_templates
        return {"error": "Gecersiz JSON formati. LLM'den IR alinamadi."}

    def template_fallback(self, user_text):
        irs = semantic_fallback_ir(user_text or "", include_do=True)
        return irs if irs else None

    def feedback_correction(self, user_text, isa_schema, error_msg, max_retries=2, memory_terms=None):
        norm = self._normalize_user_text(user_text, memory_terms=memory_terms)
        normalized_text = norm["filtered_text"] or norm["corrected_text"] or norm["cleaned"]
        prompt = self._build_prompt(normalized_text, isa_schema, error_msg=error_msg)
        retries = 0
        while retries < max_retries:
            try:
                result = self._chat(prompt, max_tokens=self._token_budget(normalized_text, min_tokens=180, max_tokens=600))
                json_str = self._extract_json_array(result)
                if not json_str:
                    retries += 1
                    continue
                return json.loads(json_str)
            except Exception as exc:
                mapped = self._map_api_error(exc)
                if mapped:
                    return {"error": mapped}
                retries += 1
        ir_templates = self.template_fallback(normalized_text)
        if ir_templates:
            return ir_templates
        return {"error": "Gecersiz JSON formati. LLM'den IR alinamadi."}

    def request_revision(self, new_ir, old_ir, isa_schema):
        new_str = json.dumps(new_ir, ensure_ascii=False)
        old_str = json.dumps(old_ir, ensure_ascii=False)
        prompt = (
            f"Hata: Yeni gelen {new_str} bilgisi, mevcut {old_str} bilgisiyle celisiyor.\n"
            "Gorev: Bu iki bilgiyi analiz et. Eger eski bilgi yanlissa 'UPDATE', "
            "eger yeni bilgi yanlissa 'REJECT', eger ust kural ile birlesebiliyorsa 'SYNTHESIZE' ver."
        )
        try:
            return self._chat(prompt, max_tokens=120)
        except Exception as exc:
            mapped = self._map_api_error(exc)
            if mapped:
                return json.dumps({"error": mapped}, ensure_ascii=False)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

