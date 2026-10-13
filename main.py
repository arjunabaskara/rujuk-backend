from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Rujuk API", description="Backend untuk cek referensi dan validasi")

# Mengizinkan frontend (rujuk.html) mengakses API ini dari domain manapun (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Data Input
class URLRequest(BaseModel):
    url: str

class DOIRequest(BaseModel):
    doi: str

class TextCompareRequest(BaseModel):
    text1: str
    text2: str

@app.post("/api/check_url")
async def check_url(req: URLRequest):
    """Mengecek status URL dan menelusuri redirect berantai (Final URL)"""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.head(req.url)
            # Jika HEAD tidak diizinkan, gunakan GET
            if response.status_code == 405:
                response = await client.get(req.url)
            
            return {
                "original_url": req.url,
                "final_url": str(response.url),
                "status_code": response.status_code,
                "is_active": response.status_code == 200
            }
    except Exception as e:
        return {"error": f"Tidak dapat menjangkau URL: {str(e)}", "is_active": False}

@app.post("/api/check_doi")
async def check_doi(req: DOIRequest):
    """Memvalidasi DOI ke database resmi Crossref dan mengambil metadata"""
    clean_doi = re.sub(r'^(https?://doi\.org/)', '', req.doi.strip())
    crossref_url = f"https://api.crossref.org/works/{clean_doi}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(crossref_url)
            if res.status_code != 200:
                raise HTTPException(status_code=404, detail="DOI tidak terdaftar di Crossref")
            
            data = res.json()["message"]
            return {
                "doi": clean_doi,
                "title": data.get("title", [""])[0],
                "publisher": data.get("publisher", "Tidak diketahui"),
                "journal": data.get("container-title", [""])[0] if data.get("container-title") else "Bukan Jurnal",
                "citation_count": data.get("is-referenced-by-count", 0),
                "type": data.get("type", "unknown")
            }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/compare_text")
async def compare_text(req: TextCompareRequest):
    """Mendeteksi kemiripan judul atau abstrak menggunakan Machine Learning (TF-IDF)"""
    if not req.text1 or not req.text2:
        return {"similarity_score": 0.0}
    
    vectorizer = TfidfVectorizer().fit_transform([req.text1, req.text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    score = cosine_sim[0][1] * 100 # Konversi ke persentase
    
    return {
        "similarity_score": round(score, 2),
        "is_suspicious": score > 75.0 # Ambang batas 75% mirip
    }