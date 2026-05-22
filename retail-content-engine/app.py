"""Web 应用入口：FastAPI 后端，包住核心生成器，对外提供生成接口与前端页面。

启动：
    uvicorn app:app --reload
或：
    python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.context_manager import load_context
from src.generator import generate_content
from src.schema import ContentOutput, MarketingContext, ProductInput

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"

app = FastAPI(title="Retail-Content-Engine", version="1.0.0")


class GenerateRequest(BaseModel):
    """前端生成请求：商品信息 + 可选营销上下文（缺省时读取 config/context.json）。"""

    sku_id: str = Field(default="WEB-001")
    product_name: str
    category: str
    features: List[str]
    target_audience: str

    season: Optional[str] = None
    current_trend: Optional[str] = None
    brand_voice: Optional[str] = None

    def to_product(self) -> ProductInput:
        return ProductInput(
            sku_id=self.sku_id.strip() or "WEB-001",
            product_name=self.product_name.strip(),
            category=self.category.strip(),
            features=[f.strip() for f in self.features if f.strip()],
            target_audience=self.target_audience.strip(),
        )

    def to_context(self) -> MarketingContext:
        base = load_context()
        return MarketingContext(
            season=self.season or base.season,
            current_trend=self.current_trend or base.current_trend,
            brand_voice=self.brand_voice or base.brand_voice,
        )


@app.get("/api/context")
def get_context() -> MarketingContext:
    """返回默认营销上下文，供前端预填。"""
    return load_context()


@app.post("/api/generate", response_model=ContentOutput)
def api_generate(req: GenerateRequest) -> ContentOutput:
    """为单个商品生成多平台营销文案。"""
    product = req.to_product()
    if not product.product_name:
        raise HTTPException(status_code=422, detail="商品名称不能为空。")
    if not product.features:
        raise HTTPException(status_code=422, detail="请至少填写一个卖点。")

    try:
        return generate_content(product, req.to_context())
    except RuntimeError as exc:  # 缺少 API Key 等配置问题
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 统一兜底，避免 500 暴露堆栈
        raise HTTPException(status_code=502, detail=f"生成失败：{exc}") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
