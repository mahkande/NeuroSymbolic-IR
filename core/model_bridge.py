import json
import os
import re

from core.nlp_utils import normalize_and_match


class NanbeigeBridge:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODEL = "llama-3.1-8b-instant"
    DEFAULT_API_KEY = ""

    def __init__(self, model_name=None, api_key=None):
        key = api_key or os.getenv("GROQ_API_KEY") or self.DEFAULT_API_KEY
        self.model_name = model_name or os.getenv("GROQ_MODEL") or self.DEFAULT_MODEL
        self.client = None
        self._init_error = ""
        if not key:
            self._init_error = "GROQ_API_KEY tanimli degil."
            return
        try:
            from groq import Groq

            self.client = Groq(api_key=key)
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
            "Her arguman girdi metnindeki bir kavramdan gelmeli.\n"
            "Gereksiz kelimeleri (bir, ve, ile, icin vb.) kullanma.\n"
            "CAUSE yalnizca acik nedensellik sinyali varsa (cunku, dolayisiyla, -se/-sa) kullan.\n"
        )
        prompt += f"\n### Komut:\nGirdi: {user_text}\nCikti: "
        prompt += (
            "\nDIKKAT: Sadece JSON listesini dondur. Aciklama yapma. "
            "Yanit '[' ile baslamali ve ']' ile bitmeli."
        )
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
        status = getattr(exc, "status_code", None)
        if status == 429 or "quota" in msg or "rate limit" in msg or "insufficient_quota" in msg:
            return self._dashboard_alert(
                "Groq kotasi/hiz limiti doldu. Lutfen Dashboard'u kontrol edin: https://console.groq.com"
            )

        network_tokens = [
            "connection",
            "timeout",
            "timed out",
            "network",
            "dns",
            "service unavailable",
            "temporarily unavailable",
            "reset",
            "refused",
        ]
        if any(token in msg for token in network_tokens):
            return self._dashboard_alert(
                "Groq baglantisi koptu. Ag baglantisini ve Dashboard'u kontrol edin: https://console.groq.com"
            )
        return None

    def _chat(self, prompt, max_tokens=160):
        if self.client is None:
            raise RuntimeError(
                "[DASHBOARD_ALERT] groq paketi kurulu degil veya yuklenemedi. "
                "Lutfen aktif ortamda 'pip install groq' calistirin."
            )
        last_error = None
        models = [self.model_name]
        if self.model_name != self.FALLBACK_MODEL:
            models.append(self.FALLBACK_MODEL)

        for model in models:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                if content:
                    return content
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in (400, 404):
                    continue
                raise

        if last_error:
            raise last_error
        return ""

    def _normalize_user_text(self, user_text, memory_terms=None):
        normalized = normalize_and_match(user_text, memory_terms=memory_terms, threshold=0.90)
        stopwords = {
            "bir", "ve", "ile", "icin", "ama", "fakat", "de", "da", "ki", "bu", "su", "o",
            "cok", "az", "en", "gibi", "olan", "olanlar"
        }
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

        prompt = self._build_prompt(
            normalized_text,
            isa_schema,
            hints=conceptnet_hints,
            normalization_note=correction_note,
        )

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
                except json.JSONDecodeError:
                    retries += 1
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
        text = (user_text or "").lower().strip()
        irs = []

        m = re.findall(r"([\wçğıöşü]+)\s+bir\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", text)
        for x, y in m:
            irs.append({"op": "ISA", "args": [x, y]})

        m = re.findall(r"([\wçğıöşü]+)\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", text)
        for x, y in m:
            if y not in {"bir"}:
                irs.append({"op": "ATTR", "args": [x, "ozellik", y]})

        m = re.search(r"^([\wçğıöşü]+)(?:\s+[\wçğıöşü]+)*\s+([\wçğıöşü]+(?:mak|mek))\s+ist(?:iyor|edi|er)\b", text)
        if m:
            irs.append({"op": "WANT", "args": [m.group(1), m.group(2)]})

        m = re.search(r"([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+inan(?:iyor|ıyordu|di|dı|ir)\b", text)
        if m:
            irs.append({"op": "BELIEVE", "args": [m.group(1), m.group(2), "0.7"]})

        m = re.search(r"([\wçğıöşü]+)\s+icin\s+([\wçğıöşü]+)", text)
        if m:
            irs.append({"op": "GOAL", "args": [m.group(2), m.group(1), "medium"]})

        m = re.search(r"once\s+([\wçğıöşü]+)\s+sonra\s+([\wçğıöşü]+)", text)
        if m:
            irs.append({"op": "BEFORE", "args": [m.group(1), m.group(2)]})

        m = re.search(r"([\wçğıöşü]+)\s+(iyi|kotu|guzel|cirkin)\b", text)
        if m:
            score = "1.0" if m.group(2) in {"iyi", "guzel"} else "-1.0"
            irs.append({"op": "EVAL", "args": [m.group(1), score]})

        m = re.search(r"([\wçğıöşü]+)\s+(?:ama|fakat|ancak)\s+([\wçğıöşü]+)", text)
        if m:
            irs.append({"op": "OPPOSE", "args": [m.group(1), m.group(2)]})

        m = re.search(r"([\wçğıöşü]+)\s+(?:cunku|çünkü|dolayisiyla|dolayısıyla)\s+([\wçğıöşü]+)", text)
        if m:
            irs.append({"op": "CAUSE", "args": [m.group(1), m.group(2)]})

        if not irs:
            toks = re.findall(r"[\wçğıöşü]+", text)
            if len(toks) >= 2:
                irs.append({"op": "DO", "args": [toks[0], toks[1]]})

        return irs if irs else None

    def feedback_correction(self, user_text, isa_schema, error_msg, max_retries=2, memory_terms=None):
        norm = self._normalize_user_text(user_text, memory_terms=memory_terms)
        normalized_text = norm["filtered_text"] or norm["corrected_text"] or norm["cleaned"]
        prompt = self._build_prompt(normalized_text, isa_schema, error_msg=error_msg)

        allowed_words = set(norm["tokens"]) | set(norm["lemmas"]) | set(norm["corrected_tokens"]) | set(norm["filtered_tokens"])
        retries = 0
        while retries < max_retries:
            try:
                result = self._chat(prompt, max_tokens=self._token_budget(normalized_text, min_tokens=180, max_tokens=600))
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
                    return ir_chain
                except json.JSONDecodeError:
                    retries += 1
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

