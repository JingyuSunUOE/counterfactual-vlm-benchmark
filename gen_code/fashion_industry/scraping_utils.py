"""
VLM Bias Dataset — 共享工具模块
================================
Pexels/SerpAPI 图片搜索, 下载验证, Gemini Review, Metadata 管理
"""

import os, json, time, io, hashlib, base64, requests
from pathlib import Path
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

# ── .env 加载 (复用 gen_originals_final.py 模式) ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

def load_env(env_path=None):
    if env_path is None:
        env_path = REPO_ROOT / ".env"
    env_path = Path(env_path)
    load_dotenv(env_path, override=True)

load_env()

# ── 限流器 ────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()

# ── 图片下载 + 验证 ──────────────────────────────────────────────
def download_image(url, output_path, min_size=512, timeout=30):
    """下载图片，验证最小尺寸，转为 RGB JPEG。返回 (success, width, height)"""
    try:
        headers = {"User-Agent": "VLMBiasResearch/1.0 (academic research)"}
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        w, h = img.size
        if w < min_size or h < min_size:
            return False, w, h
        img = img.convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), "JPEG", quality=92)
        return True, w, h
    except Exception as e:
        print(f"    [DL FAIL] {str(e)[:80]}")
        return False, 0, 0

def image_hash(path):
    """简单 MD5 去重"""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ── Pexels API 客户端 ────────────────────────────────────────────
class PexelsClient:
    BASE_URL = "https://api.pexels.com/v1"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY", "")
        self.limiter = RateLimiter(calls_per_minute=25)
        if not self.api_key:
            print("[WARN] PEXELS_API_KEY not set")

    def search(self, query, per_page=15, orientation="landscape"):
        self.limiter.wait()
        try:
            r = requests.get(
                f"{self.BASE_URL}/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": per_page, "orientation": orientation},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            results = []
            for p in data.get("photos", []):
                results.append({
                    "id": p["id"],
                    "url": p["src"]["original"],
                    "url_large": p["src"]["large2x"],
                    "photographer": p["photographer"],
                    "width": p["width"],
                    "height": p["height"],
                    "pexels_url": p["url"],
                    "license": "Pexels License (free for research)",
                })
            return results
        except Exception as e:
            print(f"  [Pexels ERROR] {e}")
            return []

    def download(self, photo, output_path, min_size=512):
        url = photo.get("url_large", photo["url"])
        return download_image(url, output_path, min_size=min_size)

# ── SerpAPI Google Images 客户端 ─────────────────────────────────
class SerpAPIClient:
    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("SERPAPI_KEY", "")
        self.limiter = RateLimiter(calls_per_minute=10)
        if not self.api_key:
            print("[WARN] SERPAPI_KEY not set")

    def search_images(self, query, num=20):
        self.limiter.wait()
        try:
            r = requests.get(
                self.BASE_URL,
                params={
                    "engine": "google_images",
                    "q": query,
                    "num": num,
                    "api_key": self.api_key,
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            results = []
            for img in data.get("images_results", []):
                source = img.get("source", "")
                if any(x in source.lower() for x in ["pinterest", "instagram", "facebook", "tiktok"]):
                    continue
                results.append({
                    "title": img.get("title", ""),
                    "url": img.get("original", ""),
                    "thumbnail": img.get("thumbnail", ""),
                    "source": source,
                    "width": img.get("original_width", 0),
                    "height": img.get("original_height", 0),
                    "license": "Google Images (fair use for research)",
                })
            return results
        except Exception as e:
            print(f"  [SerpAPI ERROR] {e}")
            return []

# ── Gemini 3.1 Pro 质量审核 ──────────────────────────────────────
class GeminiReviewer:
    MODEL = "gemini-2.5-pro"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.limiter = RateLimiter(calls_per_minute=10)

    def _encode_image(self, path):
        with open(path, "rb") as f:
            data = f.read()
        ext = Path(path).suffix.lower()
        mime = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
        if ext == ".jpg":
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        return base64.b64encode(data).decode(), mime

    def _call_gemini(self, parts, max_retries=2):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent?key={self.api_key}"
        for attempt in range(max_retries + 1):
            self.limiter.wait()
            try:
                r = requests.post(url, json={"contents": [{"parts": parts}]}, timeout=60)
                if r.status_code == 429 and attempt < max_retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                text = ""
                for c in r.json().get("candidates", []):
                    for p in c.get("content", {}).get("parts", []):
                        if "text" in p:
                            text += p["text"]
                return text
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return f"ERROR: {e}"
        return "ERROR: max retries"

    def review_counterfactual(self, cf_path, orig_path, mod_desc, ground_truth):
        b64_orig, mime_orig = self._encode_image(orig_path)
        b64_cf, mime_cf = self._encode_image(cf_path)
        prompt = (
            f"You are a quality reviewer for a VLM benchmark dataset.\n"
            f"Image 1 is the ORIGINAL image. Image 2 is the COUNTERFACTUAL (edited) image.\n"
            f"The intended modification: {mod_desc}\n"
            f"Ground truth for the edited image: {json.dumps(ground_truth)}\n\n"
            f"Evaluate:\n"
            f"1. Does Image 2 correctly show the intended modification?\n"
            f"2. Is the edit realistic and not obviously AI-generated (no artifacts)?\n"
            f"3. Is the rest of the image preserved (not distorted)?\n\n"
            f"Respond in JSON: {{\"passed\": true/false, \"score\": 0.0-1.0, \"issues\": [\"...\"], \"explanation\": \"...\"}}"
        )
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": mime_orig, "data": b64_orig}},
            {"inlineData": {"mimeType": mime_cf, "data": b64_cf}},
        ]
        text = self._call_gemini(parts)
        return self._parse_review(text)

    def review_logo(self, image_path, brand, expected_features):
        b64, mime = self._encode_image(image_path)
        prompt = (
            f"You are reviewing images for a fashion brand logo dataset.\n"
            f"Brand: {brand}\n"
            f"Expected logo features: {expected_features}\n\n"
            f"Check this image:\n"
            f"1. Is the brand logo clearly visible?\n"
            f"2. Is the image high quality (not blurry, not a screenshot)?\n"
            f"3. Does the logo match the expected brand?\n\n"
            f"Respond in JSON: {{\"passed\": true/false, \"score\": 0.0-1.0, \"issues\": [\"...\"], \"explanation\": \"...\"}}"
        )
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": mime, "data": b64}},
        ]
        text = self._call_gemini(parts)
        return self._parse_review(text)

    def _parse_review(self, text):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"passed": False, "score": 0.0, "issues": ["Failed to parse review"], "explanation": text[:200]}

# ── Gemini 图像编辑 (image-to-image) ────────────────────────────
class GeminiImageEditor:
    MODEL = "gemini-3.1-flash-image-preview"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.limiter = RateLimiter(calls_per_minute=10)

    def edit(self, input_path, prompt, output_path, max_retries=2):
        """传入原图 + editing prompt，输出编辑后的图像。返回 (success, info)"""
        with open(input_path, "rb") as f:
            img_data = f.read()
        ext = Path(input_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        b64 = base64.b64encode(img_data).decode()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime, "data": b64}},
            ]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }

        for attempt in range(max_retries + 1):
            self.limiter.wait()
            try:
                r = requests.post(url, json=payload, timeout=120)
                if r.status_code == 429 and attempt < max_retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    if attempt < max_retries:
                        time.sleep(3)
                        continue
                    return False, f"HTTP {r.status_code}"
                for c in r.json().get("candidates", []):
                    for p in c.get("content", {}).get("parts", []):
                        if "inlineData" in p:
                            img_bytes = base64.b64decode(p["inlineData"]["data"])
                            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(img_bytes)
                            return True, "gemini-edit"
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return False, "no image in response"
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return False, str(e)[:120]
        return False, "max retries"

# ── GPT 图像生成 ────────────────────────────────────────────────
def gen_gpt(prompt, path):
    """复用 gen_originals_final.py 的 GPT 生成模式"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return False, "OPENAI_API_KEY not set"
    from openai import OpenAI
    c = OpenAI(api_key=api_key)
    for m in ["gpt-image-1.5", "gpt-image-1"]:
        try:
            r = c.images.generate(model=m, prompt=prompt, size="1024x1024", quality="high", output_format="png", n=1)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(base64.b64decode(r.data[0].b64_json))
            return True, m
        except Exception:
            continue
    return False, "all gpt models failed"

# ── Gemini 图像生成 (text-to-image) ─────────────────────────────
def gen_gemini(prompt, path):
    """复用 gen_originals_final.py 的 Gemini 生成模式"""
    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={api_key}"
    try:
        r = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }, timeout=120)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        for c in r.json().get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if "inlineData" in p:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(p["inlineData"]["data"]))
                    return True, "gemini-3.1"
        return False, "no image"
    except Exception as e:
        return False, str(e)[:120]

# ── Metadata 工具 ────────────────────────────────────────────────
def save_metadata(entries, output_path):
    """追加 metadata entries 到 JSON 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    existing.extend(entries)
    output_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"  Metadata → {output_path} ({len(existing)} entries)")

def make_metadata_entry(*, filename, category, brand_key, source_type, source_url="",
                        license_info="", model="", prompt="", success=True,
                        review=None, **extra):
    entry = {
        "filename": filename,
        "category": category,
        "brand_key": brand_key,
        "source_type": source_type,
        "source_url": source_url,
        "license": license_info,
        "model": model,
        "prompt": prompt,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
    if review:
        entry["review"] = review
    entry.update(extra)
    return entry
