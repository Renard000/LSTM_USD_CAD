# app.py
import os
import pickle
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from io import StringIO
# Réduire le bruit TF si tu veux
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import load_model

# -----------------------
# CONFIG / chemins
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_lstm.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
CS_X_PATH = os.path.join(BASE_DIR, "cs_x.pkl")
CS_Y_PATH = os.path.join(BASE_DIR, "cs_y.pkl")

# MODEL_PATH = "C:/Users/pflor/Downloads/coursC#/2025/Automne/Revision/prjFinance/finance_predict/model_lstm.keras"    # tu l'as: regression.save("model_lstm.keras")
# ENCODER_PATH = "C:/Users/pflor/Downloads/coursC#/2025/Automne/Revision/prjFinance/finance_predict/encoder.pkl"
# CS_X_PATH = "C:/Users/pflor/Downloads/coursC#/2025/Automne/Revision/prjFinance/finance_predict/cs_x.pkl"
# CS_Y_PATH = "C:/Users/pflor/Downloads/coursC#/2025/Automne/Revision/prjFinance/finance_predict/cs_y.pkl"

# -----------------------
# CHARGEMENT DES ARTIFACTS
# -----------------------
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Sauvegarde-le via regression.save().")

    if not os.path.exists(ENCODER_PATH) or not os.path.exists(CS_X_PATH) or not os.path.exists(CS_Y_PATH):
        raise FileNotFoundError(
            "Transformeurs introuvables (encoder.pkl, cs_x.pkl, cs_y.pkl). "
            "Exécute save_artifacts.py après l'entraînement pour les générer."
        )

    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(CS_X_PATH, "rb") as f:
        cs_x = pickle.load(f)
    with open(CS_Y_PATH, "rb") as f:
        cs_y = pickle.load(f)

    return model, encoder, cs_x, cs_y

try:
    model, encoder, cs_x, cs_y = load_artifacts()
except Exception as e:
    # on garde l'exception pour la route /health
    model = None
    encoder = None
    cs_x = None
    cs_y = None
    load_error = str(e)
else:
    load_error = None

app = FastAPI(title="LSTM Finance API", version="1.0")

# -----------------------
# Helpers (préprocessing identique à ton training)
# -----------------------
def add_column(data: pd.DataFrame):
    # 1) SMA
    data["SMA fast"] = data["close"].rolling(30).mean().shift(1)
    data["SMA slow"] = data["close"].rolling(60).mean().shift(1)

    # 2) Position
    data["position"] = np.nan
    data.loc[data["SMA fast"] > data["SMA slow"], "position"] = 1
    data.loc[data["SMA fast"] < data["SMA slow"], "position"] = -1

    # 3) Rendement
    data["pct"] = data["close"].pct_change(1)
    data["return"] = data["pct"] * data["position"]

    # 4) Volatilité
    data["SMD fast"] = data["return"].rolling(15).std().shift(1)
    data["SMD slow"] = data["return"].rolling(30).std().shift(1)

    # 5) RSI (utilise ta lib ta)
    RSI = ta.momentum.RSIIndicator(data["close"], window=14)
    data["RSI"] = RSI.rsi().shift(1)

    return data

def prepare_features_from_df(df: pd.DataFrame):
    """
    Entrée : df doit avoir colonnes ['open','high','low','close','volume'] et index datetime
    Retour : X_seq (np.array) prêt pour model.predict et la date 'next_date'
    """
    df = df.copy()
    df.index.name = "time"

    # appliques les colonnes / indicateurs
    df = add_column(df)

    # ajouter colonnes de sentiment si manquantes
    if "sentiment" not in df.columns:
        df["sentiment"] = "neutral"
    if "score" not in df.columns:
        df["score"] = 0.0

    # remplace valeurs manquantes
    df = df.ffill().bfill()
    df = df.dropna()

    # colonnes features dans l'ordre qu'on a utilisé dans training
    feat_cols = [
        "open", "high", "low", "volume",
        "SMA fast", "SMA slow",
        "SMD fast", "SMD slow",
        "RSI", "sentiment", "score"
    ]

    X = df[feat_cols]

    # encoder & scale
    X_enc = encoder.transform(X)   # nécessite encoder.pkl
    X_scaled = cs_x.transform(X_enc)

    # construire fenêtres LSTM de 60
    seqs = []
    for i in range(60, len(X_scaled)):
        seqs.append(X_scaled[i-60:i, :])
    if len(seqs) == 0:
        raise ValueError("Pas assez de lignes pour construire une fenêtre de 60 pas (len < 60).")
    X_seq = np.array(seqs)
    next_date = df.index[-1] + pd.Timedelta(days=1)
    return X_seq, next_date, df

# -----------------------
# Routes
# -----------------------

class PredictResponse(BaseModel):
    next_date: str
    next_close_pred: float
    info: Optional[str] = None



@app.get("/")
async def root():
    """Route racine - Accueil de l'API"""
    return {
        "message": "LSTM Finance Prediction API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict_next_day": "/predict/next_day?ticker=USDCAD=X&start=2020-01-01&end=2025-11-29",
            "predict_history": "/predict/history?ticker=USDCAD=X&start=2020-01-01&end=2025-11-29",
            "upload_csv": "/predict/upload_csv (POST)"
        },
        "docs": "/docs"
    }
    
@app.get("/health")
async def health():
    if load_error:
        return {"ready": False, "error": load_error}
    return {"ready": True, "model": MODEL_PATH}

@app.get("/predict/next_day", response_model=PredictResponse)
async def predict_next_day(ticker: str = "USDCAD=X", start: str = "2020-01-01", end:str="2025-11-29"):
    """
    Télécharge les données Yahoo (Open/High/Low/Close/Volume),
    applique le preprocessing, et renvoie la prédiction du lendemain.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Artifacts non chargés : {load_error}")

    try:
        data = yf.download(ticker, start=start,end=end, progress=False)
        if data.empty:
            raise HTTPException(status_code=400, detail="Aucune donnée Yahoo pour ce ticker/dates.")
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.columns = ["open", "high", "low", "close", "volume"]
        data.index = pd.to_datetime(data.index)
        X_seq, next_date, df = prepare_features_from_df(data)

        pred_scaled = model.predict(X_seq, verbose=0)
        pred = cs_y.inverse_transform(pred_scaled)
        next_pred = float(pred[-1][0])

        return {"next_date": str(next_date.date()), "next_close_pred": next_pred, "info": f"ticker={ticker}"}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.get("/predict/history")
async def predict_history(
    ticker: str = "USDCAD=X",
    start: str = "2020-01-01",
    end: str = "2025-11-29"
):
    """
    Retourne toutes les prédictions historiques :
    - close réel
    - close prédit
    - erreur (abs / pct)
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Artifacts non chargés : {load_error}")

    try:
        # --- 1) Télécharger les données Yahoo
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            raise HTTPException(status_code=400, detail="Aucune donnée Yahoo pour ce ticker/dates.")

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.columns = ["open", "high", "low", "close", "volume"]
        data.index = pd.to_datetime(data.index)

        # --- 2) Préparation features
        X_seq, _, df_proc = prepare_features_from_df(data)

        # --- 3) Prédictions
        pred_scaled = model.predict(X_seq, verbose=0)
        preds = cs_y.inverse_transform(pred_scaled).flatten()

        # --- 4) Alignement données réelles
        dates = df_proc.index[60:]             # les prédictions commencent après 60 jours
        real_close = df_proc["close"].iloc[60:].values

        # --- 5) Dataframe résultat
        out = pd.DataFrame({
            "date": dates.astype(str),
            "real_close": real_close,
            "pred_close": preds,
        })
        out["abs_error"] = (out["pred_close"] - out["real_close"]).abs()
        out["pct_error"] = out["abs_error"] / out["real_close"]

        return out.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict/upload_csv", response_model=PredictResponse)
async def predict_upload_csv(file: UploadFile = File(...)):
    """
    Upload d'un CSV (format tab separated ou csv) similaire à ton fichier local.
    Le CSV doit avoir une colonne date en index ou colonne <DATE> selon ton preprocessing.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Artifacts non chargés : {load_error}")

    try:
        contents = await file.read()
        # tentative auto-detect sep et date
        s = contents.decode("utf-8")
        # essayer tab puis comma
        try:
            df = pd.read_csv(StringIO(s), sep="\t", parse_dates=True, index_col="<DATE>")
        except Exception:
            df = pd.read_csv(StringIO(s), sep=",", parse_dates=True, index_col=0)

        # s'assurer des noms de colonnes
        if df.shape[1] >= 5:
            # garder 5 premières colonnes (open,high,low,close,volume)
            df = df.iloc[:, :5]
            df.columns = ["open", "high", "low", "close", "volume"]
        else:
            raise HTTPException(status_code=400, detail="Le CSV doit contenir au moins 5 colonnes (Open,High,Low,Close,Volume).")

        df.index = pd.to_datetime(df.index)
        X_seq, next_date, _ = prepare_features_from_df(df)

        pred_scaled = model.predict(X_seq, verbose=0)
        pred = cs_y.inverse_transform(pred_scaled)
        next_pred = float(pred[-1][0])

        return {"next_date": str(next_date.date()), "next_close_pred": next_pred, "info": f"file={file.filename}"}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
    
    
    



