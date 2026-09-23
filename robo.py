# ==========================================
# ROBÔ TRADER — GitHub Actions (v2)
# LSTM + Macro + Trailing Stop + Alvo Parcial
# ==========================================
import os
import time
import requests
import pandas as pd
import json
from datetime import datetime

# --- Configurações gerais ---
PASTA         = "."
PASTA_MODELOS = f"{PASTA}/modelos_lstm"
ARQUIVO       = f"{PASTA}/portfolio.csv"
ARQUIVO_LOG   = f"{PASTA}/historico_sinais.csv"

# --- Telegram ---
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_tg_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

# --- Modo de execução (vem do workflow) ---
TREINAR_LSTM = os.environ.get("TREINAR_LSTM", "false").lower() == "true"

# --- Watchlists ---
CRIPTO_WATCHLIST = ["BTC", "ETH", "SOL", "BNB", "ADA"]
ACOES_WATCHLIST = {
    "PETR4":  "PETR4.SA",
    "VALE3":  "VALE3.SA",
    "ITUB4":  "ITUB4.SA",
    "BBAS3":  "BBAS3.SA",
    "WEGE3":  "WEGE3.SA",
}

CRIPTO_YF = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "ADA": "ADA-USD",
}

MACRO_TICKERS = {
    "Dolar":  "USDBRL=X",
    "SP500":  "^GSPC",
    "Ibov":   "^BVSP",
}

# --- Parâmetros de trading ---
VALOR_POR_TRADE = 1000

# --- Perfis ---
PERFIS = {
    "conservador": {
        "distancia_trailing": 3.0,
        "trailing_ativa_em":  5.0,
        "mult_alvo":          2.0,
        "mult_stop":          1.0,
        "alvo_parcial":       False,
    },
    "equilibrado": {
        "distancia_trailing": 5.0,
        "trailing_ativa_em":  2.0,
        "mult_alvo":          3.0,
        "mult_stop":          1.5,
        "alvo_parcial":       False,
    },
    "agressivo": {
        "distancia_trailing": 8.0,
        "trailing_ativa_em":  0.0,
        "mult_alvo":          5.0,
        "mult_stop":          2.0,
        "alvo_parcial":       True,
    },
}
PERFIL_ATIVO = "equilibrado"

# --- LSTM ---
JANELA           = 60
EPOCAS           = 50
BATCH            = 32
PROPORCAO_TREINO = 0.8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

os.makedirs(PASTA_MODELOS, exist_ok=True)


# ==========================================
# PORTFÓLIO
# ==========================================
def _df_vazio():
    return pd.DataFrame({
        "ID":            pd.Series(dtype="int64"),
        "Ativo":         pd.Series(dtype="object"),
        "Tipo":          pd.Series(dtype="object"),
        "Data_Compra":   pd.Series(dtype="object"),
        "Preco_Compra":  pd.Series(dtype="float64"),
        "Qtd":           pd.Series(dtype="float64"),
        "Qtd_Restante":  pd.Series(dtype="float64"),
        "Data_Venda":    pd.Series(dtype="object"),
        "Preco_Venda":   pd.Series(dtype="float64"),
        "Lucro_R$":      pd.Series(dtype="float64"),
        "Lucro_%":       pd.Series(dtype="float64"),
        "Status":        pd.Series(dtype="object"),
        "Alvo":          pd.Series(dtype="float64"),
        "Stop_Inicial":  pd.Series(dtype="float64"),
        "Stop_Atual":    pd.Series(dtype="float64"),
        "Perfil":        pd.Series(dtype="object"),
    })


def carregar_portfolio():
    if os.path.exists(ARQUIVO):
        try:
            df = pd.read_csv(ARQUIVO)
        except Exception:
            return _df_vazio()
        for c in ["Qtd_Restante", "Stop_Inicial", "Stop_Atual", "Perfil"]:
            if c not in df.columns:
                df[c] = None
        if "Stop" in df.columns:
            df["Stop_Inicial"] = df["Stop_Inicial"].fillna(df["Stop"])
            df["Stop_Atual"]   = df["Stop_Atual"].fillna(df["Stop"])
        for c in ["Ativo", "Tipo", "Data_Compra", "Data_Venda", "Status", "Perfil"]:
            if c in df.columns:
                df[c] = df[c].astype(object)
        return df
    return _df_vazio()


def salvar_portfolio(df):
    df.to_csv(ARQUIVO, index=False)


def registrar_compra(ativo, tipo, preco, qtd, alvo, stop, perfil="equilibrado"):
    df = carregar_portfolio()
    if not df[(df["Ativo"] == ativo) & (df["Status"] == "ABERTO")].empty:
        print(f"⚠️ {ativo} já está aberto.")
        return None
    novo_id = int(df["ID"].max() + 1) if not df.empty else 1
    nova = pd.DataFrame([{
        "ID": novo_id, "Ativo": ativo, "Tipo": tipo,
        "Data_Compra": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Preco_Compra": preco, "Qtd": qtd, "Qtd_Restante": qtd,
        "Data_Venda": None, "Preco_Venda": None,
        "Lucro_R$": None, "Lucro_%": None,
        "Status": "ABERTO", "Alvo": alvo,
        "Stop_Inicial": stop, "Stop_Atual": stop,
        "Perfil": perfil,
    }])
    df = pd.concat([df, nova], ignore_index=True)
    salvar_portfolio(df)
    print(f"✅ Compra: {ativo} @ {preco:.2f} | Perfil: {perfil}")
    return nova


def atualizar_trailing(ativo, preco_atual, distancia_pct):
    df = carregar_portfolio()
    mask = (df["Ativo"] == ativo) & (df["Status"] == "ABERTO")
    if df[mask].empty:
        return None
    idx = df[mask].index[0]
    stop_atual = float(df.at[idx, "Stop_Atual"] or 0)
    novo_stop = preco_atual * (1 - distancia_pct / 100)
    if novo_stop > stop_atual:
        df.at[idx, "Stop_Atual"] = round(novo_stop, 2)
        salvar_portfolio(df)
        return novo_stop
    return stop_atual


def registrar_venda(ativo, preco_venda, motivo="", parcial=False, pct_parcial=0.5):
    df = carregar_portfolio()
    mask = (df["Ativo"] == ativo) & (df["Status"] == "ABERTO")
    if df[mask].empty:
        print(f"⚠️ Sem posição aberta de {ativo}.")
        return None

    idx = df[mask].index[0]
    pc = float(df.at[idx, "Preco_Compra"])
    qtd_restante = float(df.at[idx, "Qtd_Restante"] or df.at[idx, "Qtd"])

    if parcial:
        qtd_vendida = qtd_restante * pct_parcial
        qtd_nova = qtd_restante - qtd_vendida
        lucro_rs = (preco_venda - pc) * qtd_vendida
        lucro_pct = ((preco_venda / pc) - 1) * 100
        df.at[idx, "Qtd_Restante"] = round(qtd_nova, 6)
        salvar_portfolio(df)
        print(f"✅ Venda PARCIAL: {ativo} @ {preco_venda:.2f} ({int(pct_parcial*100)}%) | "
              f"Lucro: R$ {lucro_rs:.2f} ({lucro_pct:+.2f}%)")
        return {"ativo": ativo, "preco_compra": pc, "preco_venda": preco_venda,
                "lucro_rs": lucro_rs, "lucro_pct": lucro_pct,
                "parcial": True, "restante": qtd_nova}
    else:
        lucro_rs = (preco_venda - pc) * qtd_restante
        lucro_pct = ((preco_venda / pc) - 1) * 100
        df.at[idx, "Data_Venda"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
        df.at[idx, "Preco_Venda"]  = preco_venda
        df.at[idx, "Lucro_R$"]     = round(lucro_rs, 2)
        df.at[idx, "Lucro_%"]      = round(lucro_pct, 2)
        df.at[idx, "Qtd_Restante"] = 0
        df.at[idx, "Status"]       = "FECHADO"
        salvar_portfolio(df)
        print(f"✅ Venda TOTAL: {ativo} @ {preco_venda:.2f} | "
              f"Lucro: R$ {lucro_rs:.2f} ({lucro_pct:+.2f}%) | {motivo}")
        return {"ativo": ativo, "preco_compra": pc, "preco_venda": preco_venda,
                "lucro_rs": lucro_rs, "lucro_pct": lucro_pct,
                "parcial": False, "motivo": motivo}


def resumo_portfolio():
    df = carregar_portfolio()
    if df.empty:
        print("Portfólio vazio.")
        return
    abertas  = df[df["Status"] == "ABERTO"]
    fechadas = df[df["Status"] == "FECHADO"]
    ganhos   = fechadas[fechadas["Lucro_R$"] > 0] if not fechadas.empty else fechadas
    total    = fechadas["Lucro_R$"].sum() if not fechadas.empty else 0
    win      = (len(ganhos) / len(fechadas) * 100) if len(fechadas) else 0

    print("="*50)
    print("📊 RESUMO DO PORTFÓLIO")
    print("-"*50)
    print(f"Abertas  : {len(abertas)}")
    print(f"Fechadas : {len(fechadas)}")
    print(f"Acerto   : {win:.1f}%")
    print(f"Lucro    : R$ {total:.2f}")
    print("="*50)


def salvar_analise(tabela):
    linhas = []
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in tabela.iterrows():
        linhas.append({
            "DataHora":        agora,
            "Ativo":           row["Ativo"],
            "Preco":           row["Preço"],
            "RSI":             row["RSI"],
            "Score":           row["Score"],
            "Veredito":        row["Veredito"],
            "LSTM_variacao":   row.get("LSTM_variacao"),
            "LSTM_tendencia":  row.get("LSTM_tendencia"),
            "Macro_Score":     row.get("Macro_Score"),
            "Motivos":         row["Motivos"],
        })
    df_novo = pd.DataFrame(linhas)
    if os.path.exists(ARQUIVO_LOG):
        try:
            df_antigo = pd.read_csv(ARQUIVO_LOG)
            df_final  = pd.concat([df_antigo, df_novo], ignore_index=True)
        except Exception:
            df_final = df_novo
    else:
        df_final = df_novo
    df_final.to_csv(ARQUIVO_LOG, index=False)
  

# ==========================================
# COLETA DE DADOS
# ==========================================
def _baixar_yahoo(ticker, periodo="6mo", intervalo="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "range": periodo,
        "interval": intervalo,
        "includePrePost": "false",
        "events": "div,split",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ HTTP {r.status_code} para {ticker}")
            return None
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        r0 = result[0]
        timestamps = r0.get("timestamp", [])
        quote = r0.get("indicators", {}).get("quote", [{}])[0]
        if not timestamps or not quote:
            return None
        df = pd.DataFrame({
            "data":   pd.to_datetime(timestamps, unit="s", utc=True),
            "open":   quote.get("open"),
            "high":   quote.get("high"),
            "low":    quote.get("low"),
            "close":  quote.get("close"),
            "volume": quote.get("volume"),
        })
        df = df.dropna(subset=["close"]).set_index("data").astype(float)
        return df
    except Exception as e:
        print(f"⚠️ Exceção em {ticker}: {e}")
        return None


def buscar_cripto(moeda="BTC", periodo="1mo", intervalo="1h"):
    ticker = CRIPTO_YF.get(moeda.upper())
    if not ticker:
        return None
    return _baixar_yahoo(ticker, periodo, intervalo)


def buscar_acao(ticker="PETR4.SA", periodo="2y", intervalo="1d"):
    return _baixar_yahoo(ticker, periodo, intervalo)


def buscar_macro():
    macro = {}
    for nome, ticker in MACRO_TICKERS.items():
        print(f"   🌍 {nome}")
        df = _baixar_yahoo(ticker, periodo="3mo", intervalo="1d")
        macro[nome] = df
        time.sleep(0.3)
    return macro


def calcular_contexto_macro(macro_dfs):
    contexto = {"score": 0, "resumo": [], "detalhes": {}}
    if not macro_dfs:
        return contexto

    dolar = macro_dfs.get("Dolar")
    if dolar is not None and len(dolar) >= 10:
        preco_hoje = float(dolar["close"].iloc[-1])
        preco_5d   = float(dolar["close"].iloc[-6])
        var_5d = (preco_hoje / preco_5d - 1) * 100
        contexto["detalhes"]["Dolar"] = round(var_5d, 2)
        if var_5d > 2.0:
            contexto["score"] -= 2
            contexto["resumo"].append(f"Dólar +{var_5d:.1f}% (ruim p/ ações)")
        elif var_5d > 1.0:
            contexto["score"] -= 1
            contexto["resumo"].append(f"Dólar +{var_5d:.1f}%")
        elif var_5d < -2.0:
            contexto["score"] += 2
            contexto["resumo"].append(f"Dólar {var_5d:.1f}% (bom p/ ações)")
        elif var_5d < -1.0:
            contexto["score"] += 1
            contexto["resumo"].append(f"Dólar {var_5d:.1f}%")

    sp = macro_dfs.get("SP500")
    if sp is not None and len(sp) >= 5:
        preco_hoje = float(sp["close"].iloc[-1])
        preco_5d   = float(sp["close"].iloc[-6])
        var_5d = (preco_hoje / preco_5d - 1) * 100
        contexto["detalhes"]["SP500"] = round(var_5d, 2)
        if var_5d < -3.0:
            contexto["score"] -= 2
            contexto["resumo"].append(f"S&P {var_5d:.1f}% (cautela global)")
        elif var_5d < -1.5:
            contexto["score"] -= 1
            contexto["resumo"].append(f"S&P {var_5d:.1f}%")
        elif var_5d > 3.0:
            contexto["score"] += 2
            contexto["resumo"].append(f"S&P +{var_5d:.1f}% (bom humor)")
        elif var_5d > 1.5:
            contexto["score"] += 1
            contexto["resumo"].append(f"S&P +{var_5d:.1f}%")

    ibov = macro_dfs.get("Ibov")
    if ibov is not None and len(ibov) >= 5:
        preco_hoje = float(ibov["close"].iloc[-1])
        preco_5d   = float(ibov["close"].iloc[-6])
        var_5d = (preco_hoje / preco_5d - 1) * 100
        contexto["detalhes"]["Ibov"] = round(var_5d, 2)
        if var_5d < -3.0:
            contexto["score"] -= 1
            contexto["resumo"].append(f"Ibov {var_5d:.1f}%")
        elif var_5d > 3.0:
            contexto["score"] += 1
            contexto["resumo"].append(f"Ibov +{var_5d:.1f}%")

    contexto["score"] = max(-5, min(5, contexto["score"]))
    return contexto


# ==========================================
# INDICADORES
# ==========================================
def calcular_indicadores(df, coluna_preco="close"):
    d = df.copy()
    preco = d[coluna_preco]

    d["SMA_9"]  = preco.rolling(9).mean()
    d["SMA_21"] = preco.rolling(21).mean()
    d["SMA_50"] = preco.rolling(50).mean()
    d["SMA_50_slope"] = d["SMA_50"].diff(5)
    d["EMA_9"]  = preco.ewm(span=9, adjust=False).mean()

    delta = preco.diff()
    ganho = delta.clip(lower=0).rolling(14).mean()
    perda = (-delta.clip(upper=0)).rolling(14).mean()
    rs = ganho / perda
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = preco.ewm(span=12, adjust=False).mean()
    ema26 = preco.ewm(span=26, adjust=False).mean()
    d["MACD"]      = ema12 - ema26
    d["MACD_sig"]  = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]

    d["BB_mid"] = preco.rolling(20).mean()
    desvio = preco.rolling(20).std()
    d["BB_up"]  = d["BB_mid"] + 2 * desvio
    d["BB_low"] = d["BB_mid"] - 2 * desvio

    if all(c in d.columns for c in ["high", "low", "close"]):
        high, low, close = d["high"], d["low"], d["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        d["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    else:
        d["ATR"] = preco.rolling(14).std()

    return d


def gerar_sinal(row, coluna_preco="close", macro_score=0, lstm_variacao=None, perfil="equilibrado"):
    preco_atual = row[coluna_preco]
    score = 0
    motivos = []

    em_alta_forte = (
        pd.notna(row["SMA_50"]) and pd.notna(row["SMA_50_slope"]) and
        preco_atual > row["SMA_50"] and row["SMA_50_slope"] > 0 and
        pd.notna(row["SMA_21"]) and row["SMA_21"] > row["SMA_50"]
    )
    em_baixa_forte = (
        pd.notna(row["SMA_50"]) and pd.notna(row["SMA_50_slope"]) and
        preco_atual < row["SMA_50"] and row["SMA_50_slope"] < 0 and
        pd.notna(row["SMA_21"]) and row["SMA_21"] < row["SMA_50"]
    )

    rsi = row["RSI"]
    if pd.notna(rsi):
        if rsi < 25:   pts_rsi = 3;  txt = f"RSI {rsi:.0f} sobrevendido"
        elif rsi < 35: pts_rsi = 2;  txt = f"RSI {rsi:.0f} sobrevendido"
        elif rsi < 45: pts_rsi = 1;  txt = f"RSI {rsi:.0f} baixo"
        elif rsi < 55: pts_rsi = 0;  txt = f"RSI {rsi:.0f} neutro"
        elif rsi < 65: pts_rsi = -1; txt = f"RSI {rsi:.0f} alto"
        elif rsi < 75: pts_rsi = -2; txt = f"RSI {rsi:.0f} sobrecomprado"
        else:          pts_rsi = -3; txt = f"RSI {rsi:.0f} sobrecomprado"
        if em_alta_forte and pts_rsi < 0:
            pts_rsi = pts_rsi // 2; txt += " (atenuado)"
        if em_baixa_forte and pts_rsi > 0:
            pts_rsi = pts_rsi // 2; txt += " (atenuado)"
        score += pts_rsi
        motivos.append(txt)

    if pd.notna(row["SMA_9"]) and pd.notna(row["SMA_21"]):
        if row["SMA_9"] > row["SMA_21"]:
            score += 1; motivos.append("SMA9>SMA21")
        else:
            score -= 1; motivos.append("SMA9<SMA21")

    if pd.notna(row["SMA_50"]):
        if preco_atual > row["SMA_50"]:
            score += 1; motivos.append("Acima SMA50")
        else:
            score -= 1; motivos.append("Abaixo SMA50")

    if pd.notna(row["MACD"]) and pd.notna(row["MACD_sig"]):
        if row["MACD"] > row["MACD_sig"]:
            score += 1; motivos.append("MACD+")
        else:
            score -= 1; motivos.append("MACD-")

    if pd.notna(row["BB_low"]) and pd.notna(row["BB_up"]):
        if preco_atual <= row["BB_low"]:
            score += 1; motivos.append("Banda inf")
        elif preco_atual >= row["BB_up"]:
            if not em_alta_forte:
                score -= 1; motivos.append("Banda sup")
            else:
                motivos.append("Banda sup (força)")

    if em_alta_forte:
        score += 1; motivos.append("Bônus tendência")
    if em_baixa_forte:
        score -= 1; motivos.append("Penalidade tendência")

    if macro_score != 0:
        score += macro_score
        motivos.append(f"Macro {macro_score:+d}")

    if lstm_variacao is not None:
        if lstm_variacao > 2.0:
            score += 1; motivos.append(f"LSTM +{lstm_variacao:.1f}%")
        elif lstm_variacao < -2.0:
            score -= 1; motivos.append(f"LSTM {lstm_variacao:.1f}%")

    if score >= 6:    veredito = "🟢🟢 COMPRA FORTE"
    elif score >= 4:  veredito = "🟢 COMPRAR"
    elif score == 3:  veredito = "🟢 COMPRAR"
    elif score == 2 and em_alta_forte: veredito = "🟢 COMPRAR (conf.)"
    elif score <= -6: veredito = "🔴🔴 VENDA FORTE"
    elif score <= -4: veredito = "🔴 VENDER"
    elif score <= -3: veredito = "🔴 VENDER"
    else:             veredito = "🟡 AGUARDAR"

    return veredito, score, " | ".join(motivos)


def calcular_alvo_stop(df, perfil="equilibrado"):
    cfg = PERFIS[perfil]
    d = calcular_indicadores(df, coluna_preco="close")
    ultima = d.iloc[-1]
    preco = float(ultima["close"])
    atr = float(ultima["ATR"]) if pd.notna(ultima["ATR"]) else preco * 0.02
    alvo = preco + cfg["mult_alvo"] * atr
    stop = preco - cfg["mult_stop"] * atr
    return preco, alvo, stop, atr
  

# ==========================================
# LSTM
# ==========================================
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    _lstm_disponivel = True
except Exception as e:
    print(f"⚠️ LSTM indisponível: {e}")
    _lstm_disponivel = False


np.random.seed(42)
if _lstm_disponivel:
    tf.random.set_seed(42)


def _preparar_dados_lstm(df, janela=JANELA):
    serie = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie_norm = scaler.fit_transform(serie)

    X, y = [], []
    for i in range(janela, len(serie_norm)):
        X.append(serie_norm[i-janela:i, 0])
        y.append(serie_norm[i, 0])

    X = np.array(X).reshape(-1, janela, 1)
    y = np.array(y)
    return X, y, scaler


def _construir_modelo(janela=JANELA):
    modelo = Sequential([
        Input(shape=(janela, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def treinar_lstm(ativo, df, forcar_retreino=False):
    if not _lstm_disponivel:
        return None, None

    caminho_modelo = f"{PASTA_MODELOS}/{ativo}_lstm.keras"
    caminho_scaler = f"{PASTA_MODELOS}/{ativo}_scaler.pkl"

    if os.path.exists(caminho_modelo) and not forcar_retreino:
        try:
            return load_model(caminho_modelo), joblib.load(caminho_scaler)
        except Exception as e:
            print(f"   ⚠️ Erro carregando modelo de {ativo}: {e}")

    print(f"   🧠 Treinando LSTM para {ativo}...")
    X, y, scaler = _preparar_dados_lstm(df)

    if len(X) < 100:
        print(f"   ⚠️ Dados insuficientes para {ativo}")
        return None, None

    corte = int(len(X) * PROPORCAO_TREINO)
    modelo = _construir_modelo()
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    hist = modelo.fit(
        X[:corte], y[:corte],
        validation_data=(X[corte:], y[corte:]),
        epochs=EPOCAS, batch_size=BATCH,
        callbacks=[early], verbose=0,
    )
    print(f"   ✅ {ativo} treinado — val_loss: {hist.history['val_loss'][-1]:.6f}")

    modelo.save(caminho_modelo)
    joblib.dump(scaler, caminho_scaler)
    return modelo, scaler


def prever_proximo_preco(ativo, df, modelo=None, scaler=None):
    if not _lstm_disponivel:
        return None
    if modelo is None or scaler is None:
        modelo, scaler = treinar_lstm(ativo, df)
        if modelo is None:
            return None

    X, _, _ = _preparar_dados_lstm(df)
    if len(X) == 0:
        return None

    prev_norm = modelo.predict(X[-1].reshape(1, JANELA, 1), verbose=0)[0][0]
    previsao = scaler.inverse_transform([[prev_norm]])[0][0]

    preco_atual = float(df["close"].iloc[-1])
    variacao_pct = (previsao / preco_atual - 1) * 100

    if abs(variacao_pct) < 0.5:   confianca = "Baixa"
    elif abs(variacao_pct) < 2:   confianca = "Média"
    elif abs(variacao_pct) < 5:   confianca = "Alta"
    else:                          confianca = "Muito Alta"

    return {
        "preco_atual":  preco_atual,
        "previsao":     float(previsao),
        "variacao_pct": variacao_pct,
        "tendencia":    "📈 ALTA" if variacao_pct > 0 else "📉 QUEDA",
        "confianca":    confianca,
    }


# ==========================================
# ANÁLISE EM LOTE
# ==========================================
def analisar_ativo(nome, df, macro_score=0, lstm_variacao=None, perfil="equilibrado"):
    if df is None or df.empty or len(df) < 30:
        return {"Ativo": nome, "Preço": None, "RSI": None,
                "Score": None, "Veredito": "⚠️ Sem dados", "Motivos": "—",
                "Macro_Score": macro_score,
                "LSTM_variacao": lstm_variacao,
                "LSTM_tendencia": None, "LSTM_confianca": None}

    d = calcular_indicadores(df, coluna_preco="close")
    ultima = d.iloc[-1]

    tipo = "Cripto" if nome in CRIPTO_WATCHLIST else "Ação"
    if tipo == "Cripto":
        macro_ponderado = int(round(macro_score * 0.6))
    else:
        macro_ponderado = int(round(macro_score * 1.0))

    v, s, m = gerar_sinal(ultima, coluna_preco="close",
                          macro_score=macro_ponderado,
                          lstm_variacao=lstm_variacao,
                          perfil=perfil)

    return {
        "Ativo":          nome,
        "Preço":          round(float(ultima["close"]), 2),
        "RSI":            round(float(ultima["RSI"]), 1) if pd.notna(ultima["RSI"]) else None,
        "Score":          s,
        "Veredito":       v,
        "Macro_Score":    macro_ponderado,
        "LSTM_variacao":  lstm_variacao,
        "LSTM_tendencia": None,
        "LSTM_confianca": None,
        "Motivos":        m,
    }


def _analisar_com_lstm(nome, df, treinar=True, macro_score=0, perfil="equilibrado"):
    lstm_variacao = None
    lstm_resultado = None

    if df is not None and not df.empty and len(df) >= 160 and _lstm_disponivel:
        try:
            modelo, scaler = treinar_lstm(nome, df, forcar_retreino=treinar)
            if modelo is not None:
                prev = prever_proximo_preco(nome, df, modelo, scaler)
                if prev:
                    lstm_variacao = round(prev["variacao_pct"], 2)
                    lstm_resultado = prev
        except Exception as e:
            print(f"   ⚠️ LSTM falhou para {nome}: {e}")

    resultado = analisar_ativo(nome, df, macro_score=macro_score,
                                lstm_variacao=lstm_variacao, perfil=perfil)

    if lstm_resultado:
        resultado["LSTM_tendencia"] = lstm_resultado["tendencia"]
        resultado["LSTM_confianca"] = lstm_resultado["confianca"]
        resultado["LSTM_previsao"]  = round(lstm_resultado["previsao"], 2)

    return resultado


def rodar_watchlist_completa(treinar_lstm_flag=False, perfil=None):
    if perfil is None:
        perfil = PERFIL_ATIVO

    print("   🌍 Coletando contexto macro...")
    macro_dfs = buscar_macro()
    contexto_macro = calcular_contexto_macro(macro_dfs)
    macro_score = contexto_macro["score"]
    resumo_macro = " | ".join(contexto_macro["resumo"]) if contexto_macro["resumo"] else "neutro"
    print(f"   🌍 Macro score: {macro_score:+d} | {resumo_macro}")

    resultados = []
    dfs = {}

    for nome in CRIPTO_WATCHLIST:
        print(f"   🔎 {nome}")
        df = buscar_cripto(nome, periodo="1mo", intervalo="1h")
        dfs[nome] = df
        resultados.append(_analisar_com_lstm(nome, df,
                                              treinar=treinar_lstm_flag,
                                              macro_score=macro_score,
                                              perfil=perfil))
        time.sleep(0.3)

    for nome, ticker in ACOES_WATCHLIST.items():
        print(f"   🔎 {nome}")
        df = buscar_acao(ticker, periodo="2y", intervalo="1d")
        dfs[nome] = df
        resultados.append(_analisar_com_lstm(nome, df,
                                              treinar=treinar_lstm_flag,
                                              macro_score=macro_score,
                                              perfil=perfil))

    return pd.DataFrame(resultados), dfs
  

# ==========================================
# TELEGRAM
# ==========================================
def enviar_telegram(mensagem):
    if not _tg_ok:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensagem,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"⚠️ Erro Telegram: {e}")
        return False


def alerta_compra(ativo, preco, alvo, stop, veredito, lstm_pct, lstm_tend, perfil):
    lstm_str = f"{lstm_pct:+.2f}% {lstm_tend}" if lstm_pct is not None else "n/a"
    msg = (
        f"🟢 *SINAL DE COMPRA* 🟢\n\n"
        f"📌 Ativo: `{ativo}`\n"
        f"💰 Preço: {preco:,.2f}\n"
        f"🎯 Alvo: {alvo:,.2f}\n"
        f"🛑 Stop inicial: {stop:,.2f}\n"
        f"📊 Veredito: {veredito}\n"
        f"🧠 LSTM: {lstm_str}\n"
        f"⚙️ Perfil: {perfil}\n\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    enviar_telegram(msg)


def alerta_venda(ativo, preco_compra, preco_venda, lucro_rs, lucro_pct, motivo, parcial=False):
    emoji = "✅" if lucro_rs > 0 else "❌"
    tipo  = "PARCIAL" if parcial else "TOTAL"
    msg = (
        f"{emoji} *VENDA {tipo}* {emoji}\n\n"
        f"📌 Ativo: `{ativo}`\n"
        f"💵 Entrada: {preco_compra:,.2f}\n"
        f"💵 Saída: {preco_venda:,.2f}\n"
        f"💸 Lucro: R$ {lucro_rs:,.2f} ({lucro_pct:+.2f}%)\n"
        f"📋 Motivo: {motivo}\n\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    enviar_telegram(msg)


def alerta_trailing(ativo, stop_antigo, stop_novo, preco_atual):
    msg = (
        f"📈 *TRAILING STOP ATUALIZADO*\n\n"
        f"📌 Ativo: `{ativo}`\n"
        f"💰 Preço atual: {preco_atual:,.2f}\n"
        f"🛑 Stop: {stop_antigo:,.2f} → *{stop_novo:,.2f}*\n\n"
        f"🔒 Lucro protegido\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    enviar_telegram(msg)


# ==========================================
# AUTO-TRADING
# ==========================================
def verificar_posicoes_abertas(tabela, dfs, perfil=None):
    if perfil is None:
        perfil = PERFIL_ATIVO
    cfg = PERFIS[perfil]

    portfolio = carregar_portfolio()
    abertas = portfolio[portfolio["Status"] == "ABERTO"]
    fechamentos = []

    for _, pos in abertas.iterrows():
        ativo = pos["Ativo"]
        df = dfs.get(ativo)
        if df is None or df.empty:
            continue

        preco_atual = float(df["close"].iloc[-1])
        preco_compra = float(pos["Preco_Compra"])
        alvo = float(pos["Alvo"])
        stop_atual = float(pos["Stop_Atual"] or pos["Stop_Inicial"] or pos.get("Stop", 0))
        qtd_restante = float(pos["Qtd_Restante"] or pos["Qtd"])

        # 1. Trailing stop
        lucro_pct = (preco_atual / preco_compra - 1) * 100
        if lucro_pct >= cfg["trailing_ativa_em"]:
            novo_stop = preco_atual * (1 - cfg["distancia_trailing"] / 100)
            if novo_stop > stop_atual:
                atualizar_trailing(ativo, preco_atual, cfg["distancia_trailing"])
                alerta_trailing(ativo, stop_atual, novo_stop, preco_atual)
                stop_atual = novo_stop

        # 2. Alvo
        if preco_atual >= alvo:
            if cfg["alvo_parcial"] and qtd_restante > 0:
                res = registrar_venda(ativo, preco_atual,
                                       motivo="🎯 Alvo (parcial)",
                                       parcial=True, pct_parcial=0.5)
                if res:
                    alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                                 res["lucro_rs"], res["lucro_pct"],
                                 "🎯 Alvo 50%", parcial=True)
                    novo_alvo = alvo + (alvo - preco_compra) * 0.5
                    df_port = carregar_portfolio()
                    mask = (df_port["Ativo"] == ativo) & (df_port["Status"] == "ABERTO")
                    if not df_port[mask].empty:
                        idx2 = df_port[mask].index[0]
                        df_port.at[idx2, "Alvo"] = round(novo_alvo, 2)
                        salvar_portfolio(df_port)
            else:
                res = registrar_venda(ativo, preco_atual, motivo="🎯 Alvo")
                if res:
                    alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                                 res["lucro_rs"], res["lucro_pct"], "🎯 Alvo")
                fechamentos.append(ativo)
                continue

        # 3. Stop
        if preco_atual <= stop_atual:
            motivo = "🛑 Stop" if stop_atual == float(pos["Stop_Inicial"] or 0) else "📉 Trailing Stop"
            res = registrar_venda(ativo, preco_atual, motivo=motivo)
            if res:
                alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                             res["lucro_rs"], res["lucro_pct"], motivo)
            fechamentos.append(ativo)
            continue

        # 4. Reversão técnica
        linha = tabela[tabela["Ativo"] == ativo]
        if not linha.empty and "VENDER" in str(linha.iloc[0]["Veredito"]):
            res = registrar_venda(ativo, preco_atual, motivo="🔴 Reversão técnica")
            if res:
                alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                             res["lucro_rs"], res["lucro_pct"], "🔴 Reversão")
            fechamentos.append(ativo)

    return fechamentos


def abrir_novas_posicoes(tabela, dfs, perfil=None):
    if perfil is None:
        perfil = PERFIL_ATIVO

    novas = []
    for _, row in tabela.iterrows():
        ativo = row["Ativo"]
        veredito = str(row["Veredito"])
        if "COMPRA" not in veredito:
            continue

        portfolio = carregar_portfolio()
        if not portfolio[(portfolio["Ativo"] == ativo) &
                          (portfolio["Status"] == "ABERTO")].empty:
            continue

        df = dfs.get(ativo)
        if df is None or df.empty:
            continue

        preco, alvo, stop, atr = calcular_alvo_stop(df, perfil=perfil)
        qtd = round(VALOR_POR_TRADE / preco, 6)
        tipo = "Cripto" if ativo in CRIPTO_WATCHLIST else "Ação"
        print(f"   🟢 ABRINDO {ativo} @ {preco:.2f} | Alvo {alvo:.2f} | Stop {stop:.2f} | {perfil}")
        registrar_compra(ativo, tipo, round(preco, 2), qtd,
                         round(alvo, 2), round(stop, 2), perfil=perfil)

        lstm_pct = row.get("LSTM_variacao")
        lstm_tend = row.get("LSTM_tendencia") or ""
        alerta_compra(ativo, preco, alvo, stop, veredito, lstm_pct, lstm_tend, perfil)
        novas.append(ativo)

    return novas
  

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("=" * 75)
    print("🤖 ROBÔ TRADER — CICLO GITHUB ACTIONS")
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"⚙️ Perfil: {PERFIL_ATIVO} | Modo: {'TREINO' if TREINAR_LSTM else 'CARREGAR'}")
    print("=" * 75)

    print("\n🧠 Analisando ativos...\n")
    tabela, dfs = rodar_watchlist_completa(treinar_lstm_flag=TREINAR_LSTM,
                                            perfil=PERFIL_ATIVO)

    print("\n🔄 Verificando posições...\n")
    fechamentos = verificar_posicoes_abertas(tabela, dfs, perfil=PERFIL_ATIVO)

    print("\n🎯 Abrindo novas posições...\n")
    novas = abrir_novas_posicoes(tabela, dfs, perfil=PERFIL_ATIVO)

    salvar_analise(tabela)

    print("\n" + "=" * 75)
    print("📊 SINAIS POR ATIVO")
    print("=" * 75)
    cols = ["Ativo", "Preço", "RSI", "Score", "Veredito",
            "LSTM_variacao", "LSTM_tendencia", "Macro_Score"]
    cols = [c for c in cols if c in tabela.columns]
    print(tabela[cols].to_string(index=False))

    print("\n" + "=" * 75)
    print("💼 PORTFÓLIO")
    print("=" * 75)
    portfolio = carregar_portfolio()
    if portfolio.empty:
        print("Nenhuma operação registrada.")
    else:
        cols_p = ["Ativo", "Tipo", "Preco_Compra", "Alvo",
                  "Stop_Atual", "Preco_Venda", "Lucro_R$", "Lucro_%", "Status"]
        cols_p = [c for c in cols_p if c in portfolio.columns]
        print(portfolio[cols_p].to_string(index=False))

    print("\n" + "=" * 75)
    resumo_portfolio()
    print("=" * 75)
    print(f"🟢 Compras: {', '.join(novas) if novas else 'nenhuma'}")
    print(f"🔴 Vendas : {', '.join(fechamentos) if fechamentos else 'nenhuma'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
