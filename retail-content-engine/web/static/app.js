"use strict";

const form = document.getElementById("gen-form");
const submitBtn = document.getElementById("submit-btn");
const ctaLabel = submitBtn.querySelector(".cta-label");
const spinner = submitBtn.querySelector(".spinner");
const errorMsg = document.getElementById("error-msg");
const emptyState = document.getElementById("empty-state");
const resultCards = document.getElementById("result-cards");

let lastResult = null;

// 预填默认营销上下文
fetch("/api/context")
  .then((r) => (r.ok ? r.json() : null))
  .then((ctx) => {
    if (!ctx) return;
    setPlaceholder("season", ctx.season);
    setPlaceholder("current_trend", ctx.current_trend);
    setPlaceholder("brand_voice", ctx.brand_voice);
  })
  .catch(() => {});

function setPlaceholder(name, value) {
  const el = form.elements[name];
  if (el && value) el.placeholder = value;
}

function splitFeatures(raw) {
  return raw
    .split(/[、,;；\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function setLoading(on) {
  submitBtn.disabled = on;
  spinner.hidden = !on;
  ctaLabel.textContent = on ? "AI 创作中…" : "✨ 生成全平台文案";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const data = Object.fromEntries(new FormData(form).entries());
  const features = splitFeatures(data.features || "");
  if (features.length === 0) {
    showError("请至少填写一个卖点。");
    return;
  }

  const payload = {
    sku_id: data.sku_id || "WEB-001",
    product_name: data.product_name,
    category: data.category,
    target_audience: data.target_audience,
    features,
    season: data.season || null,
    current_trend: data.current_trend || null,
    brand_voice: data.brand_voice || null,
  };

  setLoading(true);
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) {
      throw new Error(body.detail || `请求失败 (${res.status})`);
    }
    lastResult = body;
    render(body);
  } catch (err) {
    showError(err.message || "生成失败，请稍后重试。");
  } finally {
    setLoading(false);
  }
});

function render(out) {
  emptyState.hidden = true;
  resultCards.hidden = false;

  set("xhs_title", out.xiaohongshu.title);
  set("xhs_content", out.xiaohongshu.content);
  renderTags("xhs_tags", out.xiaohongshu.tags);

  set("tt_hook", out.tiktok_script.hook);
  set("tt_body", out.tiktok_script.body);
  set("tt_cta", out.tiktok_script.cta);

  renderBullets("ec_bullets", out.ecommerce_details.bullet_points);

  resultCards.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function set(field, value) {
  const el = document.querySelector(`[data-field="${field}"]`);
  if (el) el.textContent = value || "";
}

function renderTags(field, tags) {
  const el = document.querySelector(`[data-field="${field}"]`);
  el.innerHTML = "";
  (tags || []).forEach((t) => {
    const span = document.createElement("span");
    span.textContent = t.startsWith("#") ? t : `#${t}`;
    el.appendChild(span);
  });
}

function renderBullets(field, items) {
  const el = document.querySelector(`[data-field="${field}"]`);
  el.innerHTML = "";
  (items || []).forEach((b) => {
    const li = document.createElement("li");
    li.textContent = b;
    el.appendChild(li);
  });
}

// 复制功能
document.querySelectorAll(".copy-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!lastResult) return;
    const text = buildCopyText(btn.dataset.copy, lastResult);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      fallbackCopy(text);
    }
    btn.classList.add("done");
    btn.textContent = "已复制 ✓";
    setTimeout(() => {
      btn.classList.remove("done");
      btn.textContent = "复制";
    }, 1600);
  });
});

function buildCopyText(kind, out) {
  if (kind === "xhs") {
    const x = out.xiaohongshu;
    return `${x.title}\n\n${x.content}\n\n${(x.tags || []).join(" ")}`;
  }
  if (kind === "tt") {
    const t = out.tiktok_script;
    return `【钩子】${t.hook}\n【正文】${t.body}\n【CTA】${t.cta}`;
  }
  if (kind === "ec") {
    return (out.ecommerce_details.bullet_points || [])
      .map((b) => `• ${b}`)
      .join("\n");
  }
  return "";
}

function fallbackCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
  } catch {}
  document.body.removeChild(ta);
}
