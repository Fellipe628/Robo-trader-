# ==========================================
# ROBÔ TRADER — GitHub Actions (v4)
# LSTM peso real + Circuit breaker + Let winners run
# Confirmação de stop + Cooldown após alvo
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
ARQUIVO_COOLDOWN = f"{PASTA}/cooldowns.csv"
ARQUIVO_SENTIMENTO = f"{PASTA}/sentimento_cache.json"
ARQUIVO_CIRCUIT = f"{PASTA}/circuit_breaker.json"
ARQUIVO_STOPS_PEND = f"{PASTA}/stops_pendentes.csv"
COOLDOWN_HORAS   = 6

# --- Circuit Breaker ---
STOPS_24H_CAUTELOSO    = 2
STOPS_24H_DEFENSIVO    = 3
PAUSA_DEFENSIVO_HORAS  = 12
JANELA_STOPS_HORAS     = 24

# --- Telegram ---
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_tg_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

# --- APITube (sentimento) ---
APITUBE_KEY = os.environ.get("APITUBE_KEY")
_sentimento_ok = bool(APITUBE_KEY)

# --- Modo de execução ---
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

QUERIES_SENTIMENTO = {
    "PETR4": "Petrobras",
    "VALE3": "Vale",
    "ITUB4": "Itaú",
    "BBAS3": "Banco do Brasil",
    "WEGE3": "WEG",
    "BTC":   "Bitcoin",
    "ETH":   "Ethereum",
    "SOL":   "Solana",
    "BNB":   "BNB cripto",
    "ADA":   "Cardano cripto",
}

# --- Parâmetros de trading ---
VALOR_POR_TRADE = 1000
MAX_POSICOES    = 5

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
# UTILITÁRIOS
# ==========================================
def _precisao(preco):
    if preco >= 10:
        return 2
    elif preco >= 1:
        return 3
    else:
        return 4


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
        "Motivo":        pd.Series(dtype="object"),
    })
def _deduplicar_portfolio(df):
    """Remove duplicatas: mantém apenas o registro mais recente por ativo ABERTO."""
    if df.empty:
        return df
    abertos = df[df["Status"] == "ABERTO"].copy()
    fechados = df[df["Status"] != "ABERTO"].copy()
    if not abertos.empty:
        abertos = abertos.sort_values("ID", ascending=False)
        abertos = abertos.drop_duplicates(subset=["Ativo"], keep="first")
    resultado = pd.concat([fechados, abertos], ignore_index=True)
    resultado = resultado.sort_values("ID").reset_index(drop=True)
    return resultado


def _corrigir_nan_antigo(df):
    """Corrige NaN em Lucro_R$ de trades FECHADO recalculando."""
    if df.empty:
        return df
    mask = (df["Status"] == "FECHADO") & (df["Lucro_R$"].isna())
    if not mask.any():
        return df
    for idx in df[mask].index:
        try:
            pc = float(df.at[idx, "Preco_Compra"])
            pv = float(df.at[idx, "Preco_Venda"])
            qtd = float(df.at[idx, "Qtd"])
            if pd.notna(pc) and pd.notna(pv) and pd.notna(qtd):
                lucro_rs = (pv - pc) * qtd
                lucro_pct = ((pv / pc) - 1) * 100
                df.at[idx, "Lucro_R$"] = round(lucro_rs, 2)
                df.at[idx, "Lucro_%"]  = round(lucro_pct, 2)
        except Exception:
            pass
    return df


def carregar_portfolio():
    if os.path.exists(ARQUIVO):
        try:
            df = pd.read_csv(ARQUIVO)
        except Exception:
            return _df_vazio()
        for c in ["Qtd_Restante", "Stop_Inicial", "Stop_Atual", "Perfil", "Motivo"]:
            if c not in df.columns:
                df[c] = None
        if "Stop" in df.columns:
            df["Stop_Inicial"] = df["Stop_Inicial"].fillna(df["Stop"])
            df["Stop_Atual"]   = df["Stop_Atual"].fillna(df["Stop"])
        for c in ["Ativo", "Tipo", "Data_Compra", "Data_Venda", "Status", "Perfil", "Motivo"]:
            if c in df.columns:
                df[c] = df[c].astype(object)
        # Deduplicação + correção de NaN antigo
        df = _deduplicar_portfolio(df)
        df = _corrigir_nan_antigo(df)
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
    casas = _precisao(preco)
    nova = pd.DataFrame([{
        "ID": novo_id, "Ativo": ativo, "Tipo": tipo,
        "Data_Compra": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Preco_Compra": preco, "Qtd": qtd, "Qtd_Restante": qtd,
        "Data_Venda": None, "Preco_Venda": None,
        "Lucro_R$": None, "Lucro_%": None,
        "Status": "ABERTO", "Alvo": alvo,
        "Stop_Inicial": stop, "Stop_Atual": stop,
        "Perfil": perfil, "Motivo": None,
    }])
    df = pd.concat([df, nova], ignore_index=True)
    salvar_portfolio(df)
    print(f"✅ Compra: {ativo} @ {preco:.{casas}f} | Perfil: {perfil}")
    return nova


def atualizar_trailing(ativo, preco_atual, distancia_pct):
    df = carregar_portfolio()
    mask = (df["Ativo"] == ativo) & (df["Status"] == "ABERTO")
    if df[mask].empty:
        return None
    idx = df[mask].index[0]
    stop_atual = float(df.at[idx, "Stop_Atual"] or 0)
    novo_stop = preco_atual * (1 - distancia_pct / 100)
    casas = _precisao(preco_atual)
    if round(novo_stop, casas) > round(stop_atual, casas):
        df.at[idx, "Stop_Atual"] = round(novo_stop, casas)
        salvar_portfolio(df)
        return round(novo_stop, casas)
    return round(stop_atual, casas)


def registrar_venda(ativo, preco_venda, motivo="", parcial=False, pct_parcial=0.5):
    df = carregar_portfolio()
    mask = (df["Ativo"] == ativo) & (df["Status"] == "ABERTO")
    if df[mask].empty:
        print(f"⚠️ Sem posição aberta de {ativo}.")
        return None

    idx = df[mask].index[0]
    pc = float(df.at[idx, "Preco_Compra"])
    qtd_restante_raw = df.at[idx, "Qtd_Restante"]
    if pd.isna(qtd_restante_raw):
        qtd_restante_raw = df.at[idx, "Qtd"]
    qtd_restante = float(qtd_restante_raw) if pd.notna(qtd_restante_raw) else 0

    casas = _precisao(preco_venda)

    if parcial:
        qtd_vendida = qtd_restante * pct_parcial
        qtd_nova = qtd_restante - qtd_vendida
        lucro_rs = (preco_venda - pc) * qtd_vendida
        lucro_pct = ((preco_venda / pc) - 1) * 100
        df.at[idx, "Qtd_Restante"] = round(qtd_nova, 6)
        salvar_portfolio(df)
        print(f"✅ Venda PARCIAL: {ativo} @ {preco_venda:.{casas}f} ({int(pct_parcial*100)}%) | "
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
        df.at[idx, "Motivo"]       = motivo
        salvar_portfolio(df)
        print(f"✅ Venda TOTAL: {ativo} @ {preco_venda:.{casas}f} | "
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
            "Sentimento":      row.get("Sentimento"),
            "MTF":             row.get("MTF"),
            "Motivos":         row["Motivos"],
        })
    df_novo = pd.DataFrame(linhas)
    if os.path.exists(ARQUIVO_LOG):
        try:
            df_antigo = pd.read_csv(ARQUIVO_LOG)
            df_final  = pd.concat([df_antigo, df_novo], ignore_index=True)
            # Limita a 30.000 linhas (aprox. 7 dias de operação)
            if len(df_final) > 30000:
                df_final = df_final.tail(30000)
        except Exception:
            df_final = df_novo
    else:
        df_final = df_novo
    df_final.to_csv(ARQUIVO_LOG, index=False)

# ==========================================
# COOLDOWN (após stop OU alvo)
# ==========================================
def _carregar_cooldowns():
    if os.path.exists(ARQUIVO_COOLDOWN):
        try:
            df = pd.read_csv(ARQUIVO_COOLDOWN)
            df["Cooldown_Ate"] = pd.to_datetime(df["Cooldown_Ate"], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Ativo", "Cooldown_Ate", "Motivo"])


def _salvar_cooldowns(df):
    df.to_csv(ARQUIVO_COOLDOWN, index=False)


def registrar_cooldown(ativo, motivo="Stop"):
    df = _carregar_cooldowns()
    df = df[df["Ativo"] != ativo]
    nova = pd.DataFrame([{
        "Ativo":        ativo,
        "Cooldown_Ate": datetime.now() + pd.Timedelta(hours=COOLDOWN_HORAS),
        "Motivo":       motivo,
    }])
    df = pd.concat([df, nova], ignore_index=True)
    _salvar_cooldowns(df)
    print(f"   ⏸️ Cooldown: {ativo} até "
          f"{(datetime.now() + pd.Timedelta(hours=COOLDOWN_HORAS)).strftime('%d/%m %H:%M')}")


def em_cooldown(ativo):
    df = _carregar_cooldowns()
    linha = df[df["Ativo"] == ativo]
    if linha.empty:
        return False
    ate = linha.iloc[0]["Cooldown_Ate"]
    if pd.isna(ate):
        return False
    return datetime.now() < ate


# ==========================================
# CIRCUIT BREAKER (estado de risco)
# ==========================================
def _carregar_circuit_breaker():
    if os.path.exists(ARQUIVO_CIRCUIT):
        try:
            with open(ARQUIVO_CIRCUIT, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "stops_24h": 0,
        "stops_consecutivos": 0,
        "ultimo_stop": None,
        "estado": "normal",
        "atualizado_em": datetime.now().isoformat(),
    }


def _salvar_circuit_breaker(cb):
    cb["atualizado_em"] = datetime.now().isoformat()
    with open(ARQUIVO_CIRCUIT, "w") as f:
        json.dump(cb, f, indent=2)


def registrar_stop_no_circuit():
    cb = _carregar_circuit_breaker()
    agora = datetime.now()

    if cb["ultimo_stop"]:
        try:
            ultimo = datetime.fromisoformat(cb["ultimo_stop"])
            if (agora - ultimo).total_seconds() > JANELA_STOPS_HORAS * 3600:
                cb["stops_24h"] = 0
        except Exception:
            cb["stops_24h"] = 0

    cb["stops_24h"] += 1
    cb["stops_consecutivos"] += 1
    cb["ultimo_stop"] = agora.isoformat()

    if cb["stops_24h"] >= STOPS_24H_DEFENSIVO:
        cb["estado"] = "defensivo"
    elif cb["stops_24h"] >= STOPS_24H_CAUTELOSO:
        cb["estado"] = "cauteloso"
    else:
        cb["estado"] = "normal"

    _salvar_circuit_breaker(cb)
    print(f"   ⚠️ Circuit breaker: {cb['stops_24h']} stops em 24h → estado {cb['estado'].upper()}")
    return cb


def resetar_stops_consecutivos():
    cb = _carregar_circuit_breaker()
    cb["stops_consecutivos"] = 0
    _salvar_circuit_breaker(cb)


def verificar_circuit_breaker():
    """Retorna (estado, score_min_extra, tamanho_pct, pausado)."""
    cb = _carregar_circuit_breaker()
    agora = datetime.now()

    if cb["ultimo_stop"]:
        try:
            ultimo = datetime.fromisoformat(cb["ultimo_stop"])
            horas = (agora - ultimo).total_seconds() / 3600

            if horas > JANELA_STOPS_HORAS:
                cb["stops_24h"] = 0
                cb["estado"] = "normal"
                _salvar_circuit_breaker(cb)

            if cb["estado"] == "defensivo" and horas > PAUSA_DEFENSIVO_HORAS:
                cb["estado"] = "cauteloso"
                _salvar_circuit_breaker(cb)
        except Exception:
            pass

    estado = cb["estado"]
    if estado == "defensivo":
        return estado, 2, 0.5, True
    elif estado == "cauteloso":
        return estado, 1, 0.75, False
    else:
        return estado, 0, 1.0, False


# ==========================================
# STOPS PENDENTES (confirmação)
# ==========================================
def _carregar_stops_pendentes():
    if os.path.exists(ARQUIVO_STOPS_PEND):
        try:
            return pd.read_csv(ARQUIVO_STOPS_PEND)
        except Exception:
            pass
    return pd.DataFrame(columns=["Ativo", "Preco_Stop", "Data_Pendencia"])


def _salvar_stops_pendentes(df):
    df.to_csv(ARQUIVO_STOPS_PEND, index=False)


def registrar_stop_pendente(ativo, preco):
    df = _carregar_stops_pendentes()
    df = df[df["Ativo"] != ativo]
    nova = pd.DataFrame([{
        "Ativo": ativo,
        "Preco_Stop": preco,
        "Data_Pendencia": datetime.now().isoformat(),
    }])
    df = pd.concat([df, nova], ignore_index=True)
    _salvar_stops_pendentes(df)


def stop_esta_pendente(ativo):
    df = _carregar_stops_pendentes()
    return not df[df["Ativo"] == ativo].empty


def remover_stop_pendente(ativo):
    df = _carregar_stops_pendentes()
    df = df[df["Ativo"] != ativo]
    _salvar_stops_pendentes(df)


# ==========================================
# SENTIMENTO (APITube + Cache diário)
# ==========================================
def _carregar_cache_sentimento():
    if os.path.exists(ARQUIVO_SENTIMENTO):
        try:
            with open(ARQUIVO_SENTIMENTO, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _salvar_cache_sentimento(cache):
    with open(ARQUIVO_SENTIMENTO, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _buscar_sentimento_ativo(query, limit=10):
    if not _sentimento_ok:
        return 0.0
    url = "https://api.apitube.io/v1/news/everything"
    params = {
        "query": query,
        "language.code": "pt",
        "per_page": limit,
        "api_key": APITUBE_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        noticias = data.get("results", data.get("data", []))
        if not noticias:
            return 0.0
        scores = []
        for n in noticias:
            sent = (n.get("sentiment") or {}).get("overall") or {}
            score = sent.get("score")
            if score is not None:
                scores.append(float(score))
        return round(sum(scores) / len(scores), 3) if scores else 0.0
    except Exception as e:
        print(f"   ⚠️ Erro sentimento {query}: {e}")
        return None


def obter_sentimento(ativo, forcar=False):
    if not _sentimento_ok:
        return 0.0
    cache = _carregar_cache_sentimento()
    hoje = datetime.now().strftime("%Y-%m-%d")
    if not forcar:
        entrada = cache.get(ativo, {})
        if entrada.get("data") == hoje:
            return entrada.get("score", 0.0)
    query = QUERIES_SENTIMENTO.get(ativo)
    if not query:
        return 0.0
    score = _buscar_sentimento_ativo(query)
    if score is None:
        return 0.0
    cache[ativo] = {"score": score, "data": hoje}
    _salvar_cache_sentimento(cache)
    return score


def obter_sentimento_todos(forcar=False):
    if not _sentimento_ok:
        return {ativo: 0.0 for ativo in QUERIES_SENTIMENTO}
    cache = _carregar_cache_sentimento()
    hoje = datetime.now().strftime("%Y-%m-%d")
    todos = list(QUERIES_SENTIMENTO.keys())
    cache_ok = all(cache.get(a, {}).get("data") == hoje for a in todos)
    if cache_ok and not forcar:
        print("   💾 Usando cache de sentimento do dia")
        return {a: cache[a]["score"] for a in todos}
    print("   🌐 Buscando sentimento atualizado...")
    resultado = {}
    for ativo in todos:
        print(f"      📰 {ativo}...")
        resultado[ativo] = obter_sentimento(ativo, forcar=True)
        time.sleep(0.5)
    return resultado


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

def buscar_mtf(ativo, tipo="Ação"):
    """Busca 3 timeframes para análise multi-timeframe.
    - Cripto: 4H + 1H + 15m
    - Ação:  1D + 4H + 1H
    """
    if tipo == "Cripto":
        ticker = CRIPTO_YF.get(ativo.upper())
        if not ticker:
            return None
        df_maior = _baixar_yahoo(ticker, periodo="1mo", intervalo="4h")
        df_medio = _baixar_yahoo(ticker, periodo="1mo", intervalo="1h")
        df_menor = _baixar_yahoo(ticker, periodo="5d",  intervalo="15m")
    else:
        ticker = ACOES_WATCHLIST.get(ativo, f"{ativo}.SA")
        df_maior = _baixar_yahoo(ticker, periodo="6mo", intervalo="1d")
        df_medio = _baixar_yahoo(ticker, periodo="1mo", intervalo="1h")
        df_menor = _baixar_yahoo(ticker, periodo="5d",  intervalo="1h")

    return {"maior": df_maior, "medio": df_medio, "menor": df_menor}


def calcular_confluencia(dfs_mtf):
    """Analisa 3 timeframes e retorna (score, motivo).
    - +2: confluência de ALTA (maioria alinhada para cima)
    - -2: confluência de BAIXA (maioria alinhada para baixo)
    -  0: sem confluência clara
    """
    if not dfs_mtf:
        return 0, "sem dados"

    sinais = []
    detalhes = []

    for nome, df in dfs_mtf.items():
        if df is None or df.empty or len(df) < 30:
            continue
        try:
            d = calcular_indicadores(df, coluna_preco="close")
            ultima = d.iloc[-1]
            if pd.isna(ultima["SMA_9"]) or pd.isna(ultima["SMA_21"]):
                continue
            if ultima["SMA_9"] > ultima["SMA_21"]:
                sinais.append(1)
                detalhes.append(f"{nome}:↑")
            else:
                sinais.append(-1)
                detalhes.append(f"{nome}:↓")
        except Exception:
            continue

    if len(sinais) < 2:
        return 0, "dados insuficientes"

    positivos = sum(1 for s in sinais if s > 0)
    negativos = sum(1 for s in sinais if s < 0)
    total = len(sinais)

    if positivos >= 2 and positivos > negativos:
        return 2, f"ALTA ({positivos}/{total}) [{' '.join(detalhes)}]"
    elif negativos >= 2 and negativos > positivos:
        return -2, f"BAIXA ({negativos}/{total}) [{' '.join(detalhes)}]"
    else:
        return 0, f"neutro [{' '.join(detalhes)}]"


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

def gerar_sinal(row, coluna_preco="close", macro_score=0, lstm_variacao=None,
     perfil="equilibrado", sentimento=0.0, confluencia=0, confluencia_motivo=""):
    """Motor de sinais com LSTM peso real + macro + sentimento + multi-timeframe."""
    preco_atual = row[coluna_preco]
    score = 0
    motivos = []

    # --- BLOQUEIO por LSTM muito pessimista ---
    if lstm_variacao is not None and lstm_variacao < -2.0:
        return "🚫 BLOQUEADO", 0, f"LSTM {lstm_variacao:.1f}% (queda forte)"

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

    # --- LSTM com PESO REAL ---
    if lstm_variacao is not None:
        if lstm_variacao > 1.0:
            score += 2; motivos.append(f"LSTM +{lstm_variacao:.1f}%")
        elif lstm_variacao < -1.0:
            score -= 2; motivos.append(f"LSTM {lstm_variacao:.1f}%")

    # --- Sentimento ---
    if sentimento > 0.2:
        score += 1
        motivos.append(f"Sentimento +{sentimento:.2f}")
    elif sentimento < -0.2:
        score -= 1
        motivos.append(f"Sentimento {sentimento:.2f}")

    # --- Multi-Timeframe (confluência) ---
    if confluencia != 0:
        score += confluencia
        motivos.append(f"MTF {confluencia_motivo}")

    if score >= 7:    veredito = "🟢🟢 COMPRA FORTE"
    elif score >= 4:  veredito = "🟢 COMPRAR"
    elif score == 3:  veredito = "🟢 COMPRAR"
    elif score == 2 and em_alta_forte: veredito = "🟢 COMPRAR (conf.)"
    elif score <= -7: veredito = "🔴🔴 VENDA FORTE"
    elif score <= -4: veredito = "🔴 VENDER"
    elif score <= -3: veredito = "🔴 VENDER"
    else:             veredito = "🟡 AGUARDAR"

    return veredito, score, " | ".join(motivos)

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
def analisar_ativo(nome, df, macro_score=0, lstm_variacao=None,
                   perfil="equilibrado", sentimento=0.0,
                   confluencia=0, confluencia_motivo=""):
    if df is None or df.empty or len(df) < 30:
        return {"Ativo": nome, "Preço": None, "RSI": None,
                "Score": None, "Veredito": "⚠️ Sem dados", "Motivos": "—",
                "Macro_Score": macro_score,
                "LSTM_variacao": lstm_variacao,
                "LSTM_tendencia": None, "LSTM_confianca": None,
                "Sentimento": sentimento,
                "MTF": 0, "MTF_Motivo": ""}

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
                          perfil=perfil,
                          sentimento=sentimento,
                          confluencia=confluencia,
                          confluencia_motivo=confluencia_motivo)

    return {
        "Ativo":          nome,
        "Preço":          round(float(ultima["close"]), _precisao(float(ultima["close"]))),
        "RSI":            round(float(ultima["RSI"]), 1) if pd.notna(ultima["RSI"]) else None,
        "Score":          s,
        "Veredito":       v,
        "Macro_Score":    macro_ponderado,
        "LSTM_variacao":  lstm_variacao,
        "LSTM_tendencia": None,
        "LSTM_confianca": None,
        "Sentimento":     round(sentimento, 3),
        "MTF":            confluencia,
        "MTF_Motivo":     confluencia_motivo,
        "Motivos":        m,
    }


def _analisar_com_lstm(nome, df, treinar=True, macro_score=0,
                       perfil="equilibrado", sentimento=0.0,
                       confluencia=0, confluencia_motivo=""):
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
                                lstm_variacao=lstm_variacao, perfil=perfil,
                                sentimento=sentimento,
                                confluencia=confluencia,
                                confluencia_motivo=confluencia_motivo)

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

    print("   📰 Coletando sentimento de notícias...")
    sentimentos = obter_sentimento_todos()

    resultados = []
    dfs = {}

    print("   🔬 Coletando multi-timeframe...")
    # Cripto
    for nome in CRIPTO_WATCHLIST:
        print(f"   🔎 {nome}")
        df = buscar_cripto(nome, periodo="1mo", intervalo="1h")
        dfs[nome] = df
        sent = sentimentos.get(nome, 0.0)

        # MTF
        try:
            dfs_mtf = buscar_mtf(nome, tipo="Cripto")
            confluencia, confluencia_motivo = calcular_confluencia(dfs_mtf)
        except Exception as e:
            print(f"      ⚠️ MTF falhou {nome}: {e}")
            confluencia, confluencia_motivo = 0, ""

        resultados.append(_analisar_com_lstm(nome, df,
                                              treinar=treinar_lstm_flag,
                                              macro_score=macro_score,
                                              perfil=perfil,
                                              sentimento=sent,
                                              confluencia=confluencia,
                                              confluencia_motivo=confluencia_motivo))
        time.sleep(0.3)

    # Ações
    for nome, ticker in ACOES_WATCHLIST.items():
        print(f"   🔎 {nome}")
        df = buscar_acao(ticker, periodo="2y", intervalo="1d")
        dfs[nome] = df
        sent = sentimentos.get(nome, 0.0)

        # MTF
        try:
            dfs_mtf = buscar_mtf(nome, tipo="Ação")
            confluencia, confluencia_motivo = calcular_confluencia(dfs_mtf)
        except Exception as e:
            print(f"      ⚠️ MTF falhou {nome}: {e}")
            confluencia, confluencia_motivo = 0, ""

        resultados.append(_analisar_com_lstm(nome, df,
                                              treinar=treinar_lstm_flag,
                                              macro_score=macro_score,
                                              perfil=perfil,
                                              sentimento=sent,
                                              confluencia=confluencia,
                                              confluencia_motivo=confluencia_motivo))
        time.sleep(0.3)

    return pd.DataFrame(resultados), dfs



# ==========================================
# TELEGRAM
# ==========================================
def enviar_telegram(mensagem, tentativas=3):
    if not _tg_ok:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensagem,
        "parse_mode": "Markdown",
    }
    for t in range(tentativas):
        try:
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                return True
            elif r.status_code == 429:
                time.sleep(5 * (t + 1))
                continue
        except Exception as e:
            if t < tentativas - 1:
                time.sleep(2)
                continue
            print(f"⚠️ Erro Telegram após {tentativas} tentativas: {e}")
    return False

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
        casas = _precisao(preco_atual)

        stop_atual_raw = pos.get("Stop_Atual")
        if pd.isna(stop_atual_raw):
            stop_atual_raw = pos.get("Stop_Inicial")
        if pd.isna(stop_atual_raw):
            stop_atual_raw = pos.get("Stop")
        stop_atual = float(stop_atual_raw) if pd.notna(stop_atual_raw) else preco_compra * 0.98

        qtd_restante_raw = pos.get("Qtd_Restante")
        if pd.isna(qtd_restante_raw):
            qtd_restante_raw = pos.get("Qtd")
        qtd_restante = float(qtd_restante_raw) if pd.notna(qtd_restante_raw) else 0

        # --- Trailing normal ---
        lucro_pct = (preco_atual / preco_compra - 1) * 100
        if lucro_pct >= cfg["trailing_ativa_em"]:
            novo_stop = preco_atual * (1 - cfg["distancia_trailing"] / 100)
            novo_stop_arred = round(novo_stop, casas)
            stop_atual_arred = round(stop_atual, casas)
            if novo_stop_arred > stop_atual_arred:
                atualizar_trailing(ativo, preco_atual, cfg["distancia_trailing"])
                alerta_trailing(ativo, stop_atual_arred, novo_stop_arred, preco_atual)
                stop_atual = novo_stop_arred

        # --- LET WINNERS RUN: bateu alvo ---
        if preco_atual >= alvo:
            linha = tabela[tabela["Ativo"] == ativo]
            lstm_pct = None
            sma_alta = False
            if not linha.empty:
                lstm_pct = linha.iloc[0].get("LSTM_variacao")
                d = calcular_indicadores(df, coluna_preco="close")
                ultima = d.iloc[-1]
                if pd.notna(ultima["SMA_9"]) and pd.notna(ultima["SMA_21"]):
                    sma_alta = ultima["SMA_9"] > ultima["SMA_21"]

            lstm_positivo = (lstm_pct is not None and lstm_pct > 0)

            if lstm_positivo and sma_alta and qtd_restante > 0:
                # Continua posição: move stop para o alvo
                df_port = carregar_portfolio()
                mask = (df_port["Ativo"] == ativo) & (df_port["Status"] == "ABERTO")
                if not df_port[mask].empty:
                    idx2 = df_port[mask].index[0]
                    df_port.at[idx2, "Stop_Atual"] = round(alvo, casas)
                    salvar_portfolio(df_port)
                alerta_let_winner(ativo, alvo, lstm_pct)
                stop_atual = round(alvo, casas)
            else:
                # Vende no alvo + cooldown
                res = registrar_venda(ativo, preco_atual, motivo="🎯 Alvo")
                if res:
                    alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                                 res["lucro_rs"], res["lucro_pct"], "🎯 Alvo")
                    resetar_stops_consecutivos()
                    registrar_cooldown(ativo, motivo="Alvo")
                fechamentos.append(ativo)
                continue

        # --- STOP com confirmação de 1 ciclo ---
        if preco_atual <= stop_atual:
            if stop_esta_pendente(ativo):
                # 2º acionamento: vende
                motivo = "🛑 Stop" if stop_atual == float(pos["Stop_Inicial"] or 0) else "📉 Trailing Stop"
                res = registrar_venda(ativo, preco_atual, motivo=motivo)
                if res:
                    alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                                 res["lucro_rs"], res["lucro_pct"], motivo)
                    registrar_cooldown(ativo, motivo="Stop")
                    registrar_stop_no_circuit()
                remover_stop_pendente(ativo)
                fechamentos.append(ativo)
                continue
            else:
                # 1º acionamento: registra pendente, aguarda
                registrar_stop_pendente(ativo, preco_atual)
                print(f"   ⏳ {ativo} stop acionado — aguardando confirmação no próximo ciclo")
                continue
        else:
            # Preço recuperou: cancela pendência
            if stop_esta_pendente(ativo):
                remover_stop_pendente(ativo)
                print(f"   ✅ {ativo} recuperou — stop pendente cancelado")

        # --- Reversão técnica ---
        linha = tabela[tabela["Ativo"] == ativo]
        if not linha.empty and "VENDER" in str(linha.iloc[0]["Veredito"]):
            res = registrar_venda(ativo, preco_atual, motivo="🔴 Reversão técnica")
            if res:
                alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                             res["lucro_rs"], res["lucro_pct"], "🔴 Reversão")
                registrar_cooldown(ativo, motivo="Reversão")
                registrar_stop_no_circuit()
            fechamentos.append(ativo)

    return fechamentos


def abrir_novas_posicoes(tabela, dfs, perfil=None):
    if perfil is None:
        perfil = PERFIL_ATIVO

    # --- Circuit breaker ---
    estado_cb, score_extra, tamanho_pct, pausado = verificar_circuit_breaker()
    print(f"   🔌 Circuit breaker: {estado_cb.upper()} | score_extra=+{score_extra} | tamanho={int(tamanho_pct*100)}%")

    if pausado:
        print(f"   ⏸️ PAUSADO (circuit breaker defensivo). Sem novas entradas.")
        return []

    portfolio = carregar_portfolio()
    posicoes_abertas = len(portfolio[portfolio["Status"] == "ABERTO"])
    if posicoes_abertas >= MAX_POSICOES:
        print(f"   ⛔ Limite de {MAX_POSICOES} posições atingido "
              f"({posicoes_abertas} abertas). Pulando novas aberturas.")
        return []

    score_minimo = 3 + score_extra

    novas = []
    for _, row in tabela.iterrows():
        ativo = row["Ativo"]
        veredito = str(row["Veredito"])
        if "COMPRA" not in veredito:
            continue

        # Aplica score mínimo aumentado
        score_ativo = row.get("Score")
        if score_ativo is not None and score_ativo < score_minimo:
            print(f"   ⚠️ {ativo} score {score_ativo} < mínimo {score_minimo} (CB {estado_cb})")
            continue

        if em_cooldown(ativo):
            print(f"   ⏸️ {ativo} em cooldown, pulando.")
            continue

        portfolio = carregar_portfolio()
        if not portfolio[(portfolio["Ativo"] == ativo) &
                          (portfolio["Status"] == "ABERTO")].empty:
            continue

        portfolio = carregar_portfolio()
        if len(portfolio[portfolio["Status"] == "ABERTO"]) >= MAX_POSICOES:
            print(f"   ⛔ Limite de {MAX_POSICOES} posições atingido no meio do ciclo.")
            break

        df = dfs.get(ativo)
        if df is None or df.empty:
            continue

        tipo = "Cripto" if ativo in CRIPTO_WATCHLIST else "Ação"
        preco, alvo, stop, atr = calcular_alvo_stop(df, perfil=perfil, tipo=tipo)
        casas = _precisao(preco)
        valor_trade = VALOR_POR_TRADE * tamanho_pct
        qtd = round(valor_trade / preco, 6)
        print(f"   🟢 ABRINDO {ativo} @ {preco:.{casas}f} | "
              f"Alvo {alvo:.{casas}f} | Stop {stop:.{casas}f} | "
              f"Tamanho {int(tamanho_pct*100)}% | {perfil}")
        registrar_compra(ativo, tipo, round(preco, casas), qtd,
                         round(alvo, casas), round(stop, casas), perfil=perfil)

        lstm_pct = row.get("LSTM_variacao")
        lstm_tend = row.get("LSTM_tendencia") or ""
        sent = row.get("Sentimento", 0.0) or 0.0
        alerta_compra(ativo, preco, alvo, stop, veredito, lstm_pct, lstm_tend,
                      perfil, sent, estado_cb)
        novas.append(ativo)

    return novas


# ==========================================
# RELATÓRIO SEMANAL
# ==========================================
def gerar_relatorio_semanal():
    print("=" * 75)
    print("📊 RELATÓRIO SEMANAL — ROBÔ TRADER")
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 75)

    agora = datetime.now()
    inicio_semana = agora - pd.Timedelta(days=7)

    portfolio = carregar_portfolio()

    if portfolio.empty:
        enviar_telegram("📊 *RELATÓRIO SEMANAL*\n\nSem trades registrados.")
        return

    portfolio["Data_Compra"] = pd.to_datetime(portfolio["Data_Compra"], errors="coerce")
    if "Data_Venda" in portfolio.columns:
        portfolio["Data_Venda"] = pd.to_datetime(portfolio["Data_Venda"], errors="coerce")

    fechadas = portfolio[portfolio["Status"] == "FECHADO"].copy()
    fechadas_semana = fechadas[fechadas["Data_Venda"] >= inicio_semana] if len(fechadas) else fechadas
    abertas_semana = portfolio[portfolio["Data_Compra"] >= inicio_semana]

    n_fechadas = len(fechadas_semana)
    n_abertas = len(abertas_semana)
    n_total_abertas = len(portfolio[portfolio["Status"] == "ABERTO"])

    if n_fechadas > 0:
        ganhos = fechadas_semana[fechadas_semana["Lucro_R$"] > 0]
        win_rate = len(ganhos) / n_fechadas * 100
        lucro_total = fechadas_semana["Lucro_R$"].sum()
        melhor = fechadas_semana.loc[fechadas_semana["Lucro_%"].idxmax()]
        pior = fechadas_semana.loc[fechadas_semana["Lucro_%"].idxmin()]
        motivos = fechadas_semana["Motivo"].fillna("").astype(str)
        stops = int(motivos.str.contains("Stop").sum())
        alvos = int(motivos.str.contains("Alvo").sum())
        reversoes = int(motivos.str.contains("Revers").sum())
    else:
        win_rate = 0
        lucro_total = 0
        melhor = pior = None
        stops = alvos = reversoes = 0

    msg = f"📊 *RELATÓRIO SEMANAL*\n"
    msg += f"📅 {inicio_semana.strftime('%d/%m')} a {agora.strftime('%d/%m/%Y')}\n\n"
    msg += f"📌 *Trades*\n"
    msg += f"• Abertos: {n_abertas}\n"
    msg += f"• Fechados: {n_fechadas}\n"
    msg += f"• Em carteira: {n_total_abertas}\n"

    if n_fechadas > 0:
        emoji_lucro = "✅" if lucro_total > 0 else "❌"
        msg += f"• Acerto: {win_rate:.1f}%\n"
        msg += f"• Lucro: {emoji_lucro} R$ {lucro_total:+,.2f}\n\n"
        msg += f"📊 *Saídas*\n"
        msg += f"• 🎯 Alvos: {alvos}\n"
        msg += f"• 🛑 Stops: {stops}\n"
        msg += f"• 🔴 Reversões: {reversoes}\n\n"
        if melhor is not None:
            msg += f"🏆 *Melhor*: {melhor['Ativo']} ({melhor['Lucro_%']:+.2f}%)\n"
        if pior is not None:
            msg += f"💔 *Pior*: {pior['Ativo']} ({pior['Lucro_%']:+.2f}%)\n"
    else:
        msg += "\n⚠️ Nenhum trade fechado no período.\n"

    msg += f"\n💡 *Aprendizado*\n"
    if n_fechadas == 0:
        msg += "• Semana de observação — mercado sem oportunidade clara.\n"
    elif win_rate < 30:
        msg += "• Acerto baixo — considerar apertar stops ou filtros.\n"
    elif win_rate > 60:
        msg += "• Acerto alto — estratégia funcionando bem.\n"
    else:
        msg += "• Acerto moderado — normal em mercado misto.\n"

    if n_fechadas >= 3 and stops > alvos:
        msg += "• Muitos stops — mercado volátil ou entradas prematuras.\n"
    if n_fechadas >= 3 and alvos > stops:
        msg += "• Alvos dominando — momentum favorável.\n"

    msg += f"\n⏰ {agora.strftime('%d/%m/%Y %H:%M')}"

    enviar_telegram(msg)
    print(msg)
    print("=" * 75)


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    try:
        _main_interno()
    except Exception as e:
        import traceback
        erro = traceback.format_exc()
        print(f"❌ ERRO NO ROBÔ: {e}")
        print(erro)
        msg = (
            f"🚨 *ERRO NO ROBÔ*\n\n"
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            f"```\n{erro[-400:]}\n```"
        )
        enviar_telegram(msg)
        raise

def _main_interno(): 
    print("=" * 75)
    print("🤖 ROBÔ TRADER — CICLO GITHUB ACTIONS")
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"⚙️ Perfil: {PERFIL_ATIVO} | Modo: {'TREINO' if TREINAR_LSTM else 'CARREGAR'}")
    print(f"📰 Sentimento: {'ATIVO' if _sentimento_ok else 'DESATIVADO'}")

    estado_cb, _, _, _ = verificar_circuit_breaker()
    print(f"🔌 Circuit breaker: {estado_cb.upper()}")
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
            "LSTM_variacao", "LSTM_tendencia", "Macro_Score", "Sentimento", "MTF"]
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
    if os.environ.get("MODO_RELATORIO", "false").lower() == "true":
        gerar_relatorio_semanal()
    else:
        main()
