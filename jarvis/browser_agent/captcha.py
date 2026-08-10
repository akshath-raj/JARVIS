"""CAPTCHA solving with a LOCAL open-source vision model (Qwen2.5-VL via Ollama).

Runs UNDER .venv-browser only. Exposes a browser-use custom action `solve_captcha`
that screenshots the captcha image on the page, has a local vision-language model
read the characters (OCR), and returns the text for the agent to type. Fully local,
free, and open-source — no third-party captcha service, no API key. Best for
distorted-text image captchas (e.g. VTOP); general enough for other image captchas.

Gated on a configured vision model (JARVIS_CAPTCHA_MODEL); wired in only then.
"""
from __future__ import annotations

import asyncio
import re

# Find the most likely captcha <img> and return its box + src.
_JS_FIND = """
(function(){
  var imgs = Array.from(document.querySelectorAll('img'));
  function score(im){
    var s=((im.id||'')+' '+(im.className||'')+' '+(im.alt||'')+' '+(im.src||'')).toLowerCase();
    return s.indexOf('captcha')>=0 ? 2 : (s.indexOf('verif')>=0?1:0);
  }
  var cand = imgs.filter(function(im){return im.width>=30 && im.width<=500 && im.height>=12 && im.height<=220;});
  cand.sort(function(a,b){return score(b)-score(a);});
  var im = cand.filter(function(i){return score(i)>0;})[0] || cand[0];
  if(!im) return null;
  var r = im.getBoundingClientRect();
  return {found:true, x:r.x, y:r.y, w:r.width, h:r.height, src:(im.src||'')};
})()
"""

_OCR_PROMPT = (
    "This image is a CAPTCHA. Transcribe the characters exactly as they appear, left "
    "to right. Characters may be rotated, overlapping, or crossed by noise lines. Read "
    "every character. Reply with ONLY the characters (letters and digits) — no spaces, "
    "no punctuation, no quotes, no explanation."
)


def _upscale_b64_png(img_b64: str, factor: int = 3) -> str:
    """Upscale the captcha image — small distorted text reads far better enlarged."""
    import base64
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    if max(im.size) < 600:
        im = im.resize((im.width * factor, im.height * factor), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _eval_js(browser_session, expression: str):
    cdp = await browser_session.get_or_create_cdp_session()
    res = await cdp.cdp_client.send.Runtime.evaluate(
        params={"expression": expression, "returnByValue": True, "awaitPromise": True},
        session_id=cdp.session_id,
    )
    return ((res or {}).get("result") or {}).get("value")


async def _capture_captcha(browser_session, info: dict) -> str:
    """Return the captcha image as base64 PNG (from a data: URL or a clipped screenshot)."""
    src = info.get("src") or ""
    if src.startswith("data:image") and "," in src:
        return src.split(",", 1)[1]
    cdp = await browser_session.get_or_create_cdp_session()
    shot = await cdp.cdp_client.send.Page.captureScreenshot(
        params={
            "format": "png",
            "clip": {"x": info["x"], "y": info["y"], "width": info["w"], "height": info["h"], "scale": 1},
            "captureBeyondViewport": True,
        },
        session_id=cdp.session_id,
    )
    return (shot or {}).get("data") or ""


def _ocr(img_b64: str, model: str, host: str) -> str:
    from ollama import Client

    try:
        img_b64 = _upscale_b64_png(img_b64)
    except Exception:  # noqa: BLE001 - upscale is best-effort
        pass
    client = Client(host=host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": _OCR_PROMPT, "images": [img_b64]}],
        options={"temperature": 0},
    )
    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    return re.sub(r"[^A-Za-z0-9]", "", raw)[:12]


def build_captcha_tools(vision_model: str, ollama_host: str):
    """browser-use Tools (defaults + a local-vision solve_captcha action)."""
    from browser_use import ActionResult, Tools

    tools = Tools()

    @tools.action(
        "Read a distorted-text or image CAPTCHA on the current page using a local vision "
        "model, and get the characters to type. Use this ONLY when a captcha is blocking you; "
        "after it returns, type the characters into the captcha field and submit."
    )
    async def solve_captcha(browser_session) -> ActionResult:  # noqa: ANN001
        info = await _eval_js(browser_session, _JS_FIND)
        if not info or not info.get("found"):
            return ActionResult(extracted_content="No captcha image found on this page.")
        try:
            img = await _capture_captcha(browser_session, info)
            if not img:
                return ActionResult(extracted_content="Couldn't capture the captcha image.")
            text = await asyncio.to_thread(_ocr, img, vision_model, ollama_host)
        except Exception as e:  # noqa: BLE001
            return ActionResult(extracted_content=f"Captcha reading failed: {e}")
        if not text:
            return ActionResult(extracted_content="The vision model couldn't read the captcha.")
        return ActionResult(
            extracted_content=(
                f"The captcha reads '{text}'. Type exactly '{text}' into the captcha text field, "
                "then submit. If it's rejected, call solve_captcha again to retry."
            )
        )

    return tools
