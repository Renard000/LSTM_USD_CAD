import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pyodbc as db 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import ta

#---------------------------------------------
#               PREPROCESSING
#---------------------------------------------
def preprocessig(path: str):
    
    data = pd.read_csv(path, delimiter="\t", parse_dates=True, index_col="<DATE>")
    data = data.iloc[:, :-2]
    data.columns = ["open", "high", "low", "close", "volume"]
    data.index.name = "time"
    
    # Imputation
    impute = SimpleImputer(missing_values=np.nan, strategy="mean")
    cols = data.values
    impute.fit(cols)
    data.iloc[:, :] = impute.transform(cols)
    
    data = data.sort_index(ascending=True)
    
    return data 


#---------------------------------------------
#           AJOUT COLONNES INDICATEURS
#---------------------------------------------
def add_column(data: pd.DataFrame):
    
    # SMA
    data["SMA fast"] = data["close"].rolling(30).mean().shift(1)
    data["SMA slow"] = data["close"].rolling(60).mean().shift(1)
    
    # Position
    data["position"] = np.nan
    data.loc[data["SMA fast"] > data["SMA slow"], "position"] = 1
    data.loc[data["SMA fast"] < data["SMA slow"], "position"] = -1
    
    # Rendement
    data["pct"] = data["close"].pct_change(1)
    data["return"] = data["pct"] * data["position"]
    
    # Volatilité
    data["SMD fast"] = data["return"].rolling(15).std().shift(1)
    data["SMD slow"] = data["return"].rolling(30).std().shift(1)
    
    # RSI
    RSI = ta.momentum.RSIIndicator(data["close"], window=14)
    data["RSI"] = RSI.rsi().shift(1)
    
    return data


#---------------------------------------------
#           LECTURE SQL + AJOUT SENTIMENT
#---------------------------------------------
def read_db(name_db: str, data: pd.DataFrame):
    
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=.\\SQLEXPRESS;"
        f"DATABASE={name_db};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
        f"UID=sa; PWD=ton_mot_de_passe;"
    )
   
    with db.connect(conn_str, autocommit=True) as conn:
        data["sentiment"] = np.nan
        data["sentiment"] = data["sentiment"].astype("object")
        data["score"] = np.nan
        data["score"] = data["score"].astype(float)
        
        cursor = conn.cursor()
        sql = "SELECT Sentiment_news, Score_news FROM News WHERE Date_news >= ? AND Date_news < DATEADD(day, 1, ?)"
        
        for date in data.index: 
            date_str = date.strftime("%Y-%m-%d")
            cursor.execute(sql, (date_str, date_str))
            row = cursor.fetchone()
            
            if row:
                data.loc[data.index == date, "sentiment"] = str(row[0])
                data.loc[data.index == date, "score"] = float(row[1])
    
    return data


#---------------------------------------------
#           CHARGEMENT DES DONNÉES
#---------------------------------------------
chemin = "USDCAD.Finance.csv"

data = preprocessig(chemin)
data = add_column(data)
data = read_db("Finance", data)

# Remplacement valeurs NaN ou objets
data = data.ffill().bfill()

# Supprimer NA
data = data.dropna()


#---------------------------------------------
#           PREPARATION TRAINING
#---------------------------------------------
x = data[[
    "open", "high", "low", "volume",
    "SMA fast", "SMA slow",
    "SMD fast", "SMD slow",
    "RSI", "sentiment", "score"
]]

y = data[["close"]].values

# Encodage
encoder = make_column_transformer(
    (OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), [9]),  # sentiment
    remainder='passthrough'
)

x = encoder.fit_transform(x)

# Scaling
cs_x = MinMaxScaler()
x = cs_x.fit_transform(x)

cs_y = MinMaxScaler()
y_smooth = pd.Series(y.flatten()).rolling(10).mean().bfill().values.reshape(-1, 1)
y = cs_y.fit_transform(y_smooth)

# Séquences LSTM (60 jours)
x_train = []
y_train = []

for i in range(60, len(x)):
    x_train.append(x[i-60:i, :])
    y_train.append(y[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

print("Shape final X :", x_train.shape)


#---------------------------------------------
#         MODELE LSTM PROFESSIONNEL
#---------------------------------------------
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
regression = Sequential()

regression.add(Input(shape=(x_train.shape[1], x_train.shape[2])))

regression.add(LSTM(50, return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(50, return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(50))
regression.add(Dropout(0.2))

regression.add(Dense(1, activation='linear'))  # prédiction continue

regression.compile(
    optimizer= Adam(learning_rate=0.0003),
    loss="mean_squared_error"
)


es = EarlyStopping(patience=15, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 1e-6)
regression.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, callbacks=[es, rl])


import yfinance as yf

def load_yahoo(ticker="USDCAD=X", start="2020-01-01"):
    data = yf.download(ticker, start=start)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.columns = ["open", "high", "low", "close", "volume"]
    data.index.name = "time"
    return data

# ---- Charger données ----
new_data = load_yahoo("USDCAD=X")

# ---- Prétraitement ----
new_data = add_column(new_data)

# Ajouter colonnes manquantes pour cohérence
new_data["sentiment"] = "neutral"
new_data["score"] = 0.0

new_data = new_data.ffill().bfill().dropna()

# ---- Features ----
x_new = new_data[[
    "open", "high", "low", "volume",
    "SMA fast", "SMA slow",
    "SMD fast", "SMD slow",
    "RSI", "sentiment", "score"
]]

# Encoder & scaler déjà fit
x_new = encoder.transform(x_new)
x_new = cs_x.transform(x_new)

# ---- Fenêtres LSTM ----
X_seq = []
for i in range(60, len(x_new)):
    X_seq.append(x_new[i-60:i])
X_seq = np.array(X_seq)

# ---- Prédiction ----
pred_scaled = regression.predict(X_seq)
pred = cs_y.inverse_transform(pred_scaled)

print("Prochaine clôture prédite :", pred[-1][0])

dates_pred = new_data.index[60:]

plt.figure(figsize=(15, 8))

plt.plot(new_data.index, new_data["close"], label="Cours réel", alpha=0.6)
plt.plot(dates_pred, pred, label="Prédiction LSTM", linewidth=3)

plt.title("Réel vs Prédiction LSTM")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.show()



# ---------------------------------------------------------
#         PRÉDICTION DU LENDEMAIN + AFFICHAGE
# ---------------------------------------------------------

# Dernière fenêtre de 60 jours
last_window = x_new[-60:]  # les 60 dernières lignes déjà scalées
X_input = last_window.reshape(1, 60, x_new.shape[1])

# Prédiction du lendemain (scaled → inverse)
next_day_scaled = regression.predict(X_input)
next_day = cs_y.inverse_transform(next_day_scaled)[0][0]

# Date du lendemain
last_date = new_data.index[-1]
next_date = last_date + pd.Timedelta(days=1)

print("Prédiction du lendemain :", next_day)

# ----- Affichage -----
plt.figure(figsize=(12, 6))

plt.plot(new_data.index, new_data["close"], label="Cours réel", alpha=0.7)

# Point rouge : prédiction du lendemain
plt.scatter(next_date, next_day, s=120, color="red", label="Prédiction du lendemain")

plt.title("Prédiction du lendemain (LSTM)")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.show()

regression.save("model_lstm.keras")