"""
SmartLease AI — API de Análise de Contratos Imobiliários
Deploy gratuito no Render.com
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
import json
import os
import re

app = FastAPI(title="SmartLease AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

PROMPT_SISTEMA = """Voce e um especialista em direito imobiliario brasileiro com 20 anos de experiencia analisando contratos de locacao, compra e venda, e permuta.

Analise o contrato abaixo e retorne EXATAMENTE um JSON valido com os 5 pontos mais criticos para o CORRETOR DE IMOVEIS se atentar antes de apresentar ao cliente.

Formato obrigatorio (apenas JSON, sem markdown, sem texto fora do JSON):
{
  "tipo_contrato": "string",
  "nivel_risco": "BAIXO | MEDIO | ALTO",
  "pontos_criticos": [
    {"numero": 1, "titulo": "string", "descricao": "string", "acao_recomendada": "string"},
    {"numero": 2, "titulo": "...", "descricao": "...", "acao_recomendada": "..."},
    {"numero": 3, "titulo": "...", "descricao": "...", "acao_recomendada": "..."},
    {"numero": 4, "titulo": "...", "descricao": "...", "acao_recomendada": "..."},
    {"numero": 5, "titulo": "...", "descricao": "...", "acao_recomendada": "..."}
  ],
  "resumo_executivo": "string (3-4 frases)"
}

Priorize: multas, prazos, responsabilidades, reajuste, garantias.

CONTRATO:
"""


def extrair_texto_pdf(conteudo_bytes: bytes) -> str:
    doc = fitz.open(stream=conteudo_bytes, filetype="pdf")
    texto = "".join(pagina.get_text() for pagina in doc)
    doc.close()
    return texto.strip()


def formatar_analise(analise: dict) -> str:
    risco = analise.get("nivel_risco", "N/D")
    emoji = {"BAIXO": "verde", "MEDIO": "amarelo", "ALTO": "vermelho"}.get(risco, "")
    linhas = [
        f"Contrato: {analise.get('tipo_contrato', '')}",
        f"Risco: {risco} {emoji}", "",
        "===== 5 PONTOS CRITICOS =====",
    ]
    for p in analise.get("pontos_criticos", []):
        linhas += ["", f"{p['numero']}. {p['titulo']}", p['descricao'], f"Acao: {p['acao_recomendada']}"]
    linhas += ["", "===== RESUMO =====", analise.get("resumo_executivo", "")]
    return "\n".join(linhas)


async def chamar_gemini(texto_contrato: str) -> dict:
    payload = {
        "contents": [{"parts": [{"text": PROMPT_SISTEMA + texto_contrato[:40000]}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.8, "maxOutputTokens": 2048},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(GEMINI_URL, json=payload)
        resp.raise_for_status()
    texto = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    texto = re.sub(r"```json\s*|```\s*", "", texto).strip()
    return json.loads(texto)


async def processar_pdf_bytes(conteudo: bytes) -> dict:
    texto = extrair_texto_pdf(conteudo)
    if len(texto) < 100:
        raise HTTPException(422, "PDF parece vazio ou e imagem sem OCR.")
    analise = await chamar_gemini(texto)
    analise["texto_formatado"] = formatar_analise(analise)
    return analise


@app.post("/analisar")
async def analisar_arquivo(arquivo: UploadFile = File(...)):
    if not arquivo.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Envie um arquivo PDF.")
    conteudo = await arquivo.read()
    if len(conteudo) > 10 * 1024 * 1024:
        raise HTTPException(400, "PDF muito grande. Maximo 10 MB.")
    return await processar_pdf_bytes(conteudo)


class UrlPayload(BaseModel):
    url: str

@app.post("/analisar-url")
async def analisar_url(payload: UrlPayload):
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(payload.url)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(400, f"Nao foi possivel baixar o arquivo: {e}")
    return await processar_pdf_bytes(resp.content)


@app.get("/")
def health():
    return {"status": "SmartLease AI online", "version": "1.0.0"}
