from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import zlib, bz2, lzma, math, io
import textdistance
from nltk.corpus import wordnet as wn
import nltk
from collections import Counter

# Download WordNet data (only first time)
nltk.download("wordnet")
nltk.download("omw-1.4")

# ----------------------------
# Similarity Functions
# ----------------------------
def levenshtein_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    distance = textdistance.levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - distance / max_len

def compression_similarity_generic(s1, s2, compressor):
    s1, s2 = s1.encode("utf-8"), s2.encode("utf-8")
    c1, c2 = len(compressor(s1)), len(compressor(s2))
    c12 = len(compressor(s1 + s2))
    if max(c1, c2) == 0:
        return 1.0
    ncd = (c12 - min(c1, c2)) / max(c1, c2)
    return 1 - ncd

def compression_similarity_ensemble(s1, s2):
    # Use zlib, bz2, lzma
    sims = [
        compression_similarity_generic(s1, s2, zlib.compress),
        compression_similarity_generic(s1, s2, bz2.compress),
        compression_similarity_generic(s1, s2, lzma.compress),
    ]
    return sum(sims) / len(sims)

def entropy(text):
    probs = [freq / len(text) for freq in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def entropy_similarity(s1, s2):
    e1, e2 = entropy(s1), entropy(s2)
    return 1 - abs(e1 - e2) / max(e1, e2) if max(e1, e2) > 0 else 1.0

def ngram_model(text, n=2):
    tokens = text.split()
    model = Counter(zip(*[tokens[i:] for i in range(n)]))
    return model, len(tokens)

def perplexity_similarity(s1, s2, n=2):
    model, length = ngram_model(s1, n)
    if length < n or not model:
        return 0.5  # fallback
    tokens = s2.split()
    total, count = 0, 0
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i+n])
        if ngram in model:
            total += 1
        count += 1
    return total / count if count else 0.5

def wordnet_similarity(s1, s2):
    words1, words2 = s1.split(), s2.split()
    total, count = 0, 0
    for w1 in words1:
        syns1 = wn.synsets(w1)
        if not syns1:
            continue
        best = 0
        for w2 in words2:
            syns2 = wn.synsets(w2)
            if not syns2:
                continue
            sim = syns1[0].wup_similarity(syns2[0])
            if sim and sim > best:
                best = sim
        if best:
            total += best
            count += 1
    return total / count if count > 0 else 0.0

# ----------------------------
# Final Similarity Fusion
# ----------------------------
def compute_similarity(text1, text2):
    lev = levenshtein_similarity(text1, text2)
    comp = compression_similarity_ensemble(text1, text2)
    ent = entropy_similarity(text1, text2)
    perplex = perplexity_similarity(text1, text2)

    # Adjusted weights since WordNet is removed
    final = (
        0.45 * lev +
        0.25 * comp +
        0.2 * ent +
        0.1 * perplex
    )
    return round(min(max(final, 0), 1), 3)

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="Text Similarity API", version="2.0")

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/")
def get_similarity(data: TextPair):
    score = compute_similarity(data.text1, data.text2)
    return {"similarity score": score}

@app.post("/upload_csv/")
async def process_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if "text1" not in df.columns or "text2" not in df.columns:
        return {"error": "CSV must contain 'text1' and 'text2' columns"}

    df["similarity_score"] = [
        compute_similarity(str(t1), str(t2))
        for t1, t2 in zip(df["text1"], df["text2"])
    ]

    return {
        "message": "Processed successfully",
        "preview": df[["text1", "text2", "similarity_score"]].head(5).to_dict(orient="records")
    }

@app.post("/upload_csv_download/")
async def process_csv_download(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if "text1" not in df.columns or "text2" not in df.columns:
        return {"error": "CSV must contain 'text1' and 'text2' columns"}

    df["similarity_score"] = [
        compute_similarity(str(t1), str(t2))
        for t1, t2 in zip(df["text1"], df["text2"])
    ]

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )

@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Text Similarity CSV Uploader</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f9f9f9; }
        h2 { color: #333; }
        .box {
          border: 1px solid #ccc;
          padding: 1.5rem;
          border-radius: 10px;
          background: #fff;
          max-width: 450px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        input[type=file] {
          margin-top: 1rem;
        }
        button {
          margin-top: 1rem;
          padding: 0.6rem 1.2rem;
          border: none;
          background: #007BFF;
          color: white;
          border-radius: 6px;
          cursor: pointer;
          font-size: 1rem;
        }
        button:disabled { background: #999; cursor: not-allowed; }
        #status { margin-top: 1rem; font-weight: bold; }
      </style>
    </head>
    <body>
      <h2>Upload CSV for Text Similarity</h2>
      <div class="box">
        <form id="uploadForm">
          <label>Select CSV File:</label><br>
          <input type="file" id="fileInput" name="file" accept=".csv" required />
          <br><br>
          <button type="submit">Upload & Process</button>
        </form>
        <p id="status"></p>
      </div>

      <script>
  document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
      alert("Please select a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("status").innerText = "Processing... Please wait.";

    try {
      const response = await fetch("/upload_csv_download/", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      // 🔹 Convert to Blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      // ✅ Force CSV filename
      a.download = "results.csv";

      document.body.appendChild(a);
      a.click();
      a.remove();

      // Free memory after download
      window.URL.revokeObjectURL(url);

      document.getElementById("status").innerText = "✅ Done! Your CSV has been downloaded.";
    } catch (err) {
      document.getElementById("status").innerText = "❌ Error: " + err.message;
    }
  });
</script>

    </body>
    </html>
    """
