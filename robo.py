# ==========================================
# ROBÔ TRADER — Versão GitHub Actions (sem LSTM)
# ==========================================
import os
import time
import requests
import pandas as pd
import json
from datetime import datetime

# --- Configurações gerais ---
PASTA         = "."
ARQUIVO       = f"{PASTA}/portfolio.csv"
ARQUIVO_LOG   = f"{PASTA}/historico_sinais.csv"

# --- Telegram ---
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_tg_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

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

VALOR_POR_TRADE = 1000
MULT_ALVO       = 3.0
MULT_STOP       = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ==========================================
# PORTFÓLIO
# ==========================================
def _df_vazio():
    return pd.DataFrame({
        "ID":           pd.Series(dtype="int64"),
        "Ativo":        pd.Series(dtype="object"),
        "Tipo":         pd.Series(dtype="object"),
        "Data_Compra":  pd.Series(dtype="object"),
        "Preco_Compra": pd.Series(dtype="float64"),
        "Qtd":          pd.Series(dtype="float64"),
        "Data_Venda":   pd.Series(dtype="object"),
        "Preco_Venda":  pd.Series(dtype="float64"),
        "Lucro_R$":     pd.Series(dtype="float64"),
        "Lucro_%":      pd.Series(dtype="float64"),
        "Status":       pd.Series(dtype="object"),
        "Alvo":         pd.Series(dtype="float64"),
        "Stop":         pd.Series(dtype="float64"),
    })


def carregar_portfolio():
    if os.path.exists(ARQUIVO):
        df = pd.read_csv(ARQUIVO)
        for c in ["Ativo", "Tipo", "Data_Compra", "Data_Venda", "Status"]:
            if c in df.columns:
                df[c] = df[c].astype(object)
        return df
    return _df_vazio()


def salvar_portfolio(df):
    df.to_csv(ARQUIVO, index=False)


def registrar_compra(ativo, tipo, preco, qtd, alvo, stop):
    df = carregar_portfolio()
    if not df[(df["Ativo"] == ativo) & (df["Status"] == "ABERTO")].empty:
        print(f"⚠️ {ativo} já está aberto.")
        return None
    novo_id = int(df["ID"].max() + 1) if not df.empty else 1
    nova = pd.DataFrame([{
        "ID": novo_id, "Ativo": ativo, "Tipo": tipo,
        "Data_Compra": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Preco_Compra": preco, "Qtd": qtd,
        "Data_Venda": None, "Preco_Venda": None,
        "Lucro_R$": None, "Lucro_%": None,
        "Status": "ABERTO", "Alvo": alvo, "Stop": stop
    }])
    df = pd.concat([df, nova], ignore_index=True)
    salvar_portfolio(df)
    print(f"✅ Compra: {ativo} @ {preco:.2f}")
    return nova


def registrar_venda(ativo, preco_venda):
    df = carregar_portfolio()
    mask = (df["Ativo"] == ativo) & (df["Status"] == "ABERTO")
    if df[mask].empty:
        print(f"⚠️ Sem posição aberta de {ativo}.")
        return None
    idx = df[mask].index[0]
    pc, qtd = df.at[idx, "Preco_Compra"], df.at[idx, "Qtd"]
    lucro_rs = (preco_venda - pc) * qtd
    lucro_pct = ((preco_venda / pc) - 1) * 100

    df.at[idx, "Data_Venda"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.at[idx, "Preco_Venda"] = preco_venda
    df.at[idx, "Lucro_R$"]    = round(lucro_rs, 2)
    df.at[idx, "Lucro_%"]     = round(lucro_pct, 2)
    df.at[idx, "Status"]      = "FECHADO"
    salvar_portfolio(df)

    print(f"✅ Venda: {ativo} @ {preco_venda:.2f} | Lucro: R$ {lucro_rs:.2f} ({lucro_pct:+.2f}%)")
    return {"ativo": ativo, "preco_compra": pc, "preco_venda": preco_venda,
            "lucro_rs": lucro_rs, "lucro_pct": lucro_pct}

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
            print(f"⚠️ Sem dados para {ticker}")
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


# ==========================================
# INDICADORES E SINAIS
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


def gerar_sinal(row, coluna_preco="close"):
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
        else:          pts_rsi = -3; txt = f"RSI {rsi:.0f} muito sobrecomprado"

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
        score += 1; motivos.append("Bônus tendência alta")
    if em_baixa_forte:
        score -= 1; motivos.append("Penalidade tendência baixa")

    if score >= 5:    veredito = "🟢🟢 COMPRA FORTE"
    elif score >= 3:  veredito = "🟢 COMPRAR"
    elif score == 2 and em_alta_forte: veredito = "🟢 COMPRAR (conf.)"
    elif score <= -5: veredito = "🔴🔴 VENDA FORTE"
    elif score <= -3: veredito = "🔴 VENDER"
    else:             veredito = "🟡 AGUARDAR"

    return veredito, score, " | ".join(motivos)

# ==========================================
# ANÁLISE EM LOTE
# ==========================================
def analisar_ativo(nome, df):
    if df is None or df.empty or len(df) < 30:
        return {"Ativo": nome, "Preço": None, "RSI": None,
                "Score": None, "Veredito": "⚠️ Sem dados", "Motivos": "—"}
    d = calcular_indicadores(df, coluna_preco="close")
    ultima = d.iloc[-1]
    v, s, m = gerar_sinal(ultima, coluna_preco="close")
    return {
        "Ativo":    nome,
        "Preço":    round(float(ultima["close"]), 2),
        "RSI":      round(float(ultima["RSI"]), 1) if pd.notna(ultima["RSI"]) else None,
        "Score":    s,
        "Veredito": v,
        "Motivos":  m,
    }


def rodar_watchlist_completa():
    resultados = []
    dfs = {}
    for nome in CRIPTO_WATCHLIST:
        print(f"   🔎 {nome}")
        df = buscar_cripto(nome, periodo="1mo", intervalo="1h")
        dfs[nome] = df
        resultados.append(analisar_ativo(nome, df))
        time.sleep(0.3)
    for nome, ticker in ACOES_WATCHLIST.items():
        print(f"   🔎 {nome}")
        df = buscar_acao(ticker, periodo="2y", intervalo="1d")
        dfs[nome] = df
        resultados.append(analisar_ativo(nome, df))
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


def alerta_compra(ativo, preco, alvo, stop, veredito):
    msg = (
        f"🟢 *SINAL DE COMPRA* 🟢\n\n"
        f"📌 Ativo: `{ativo}`\n"
        f"💰 Preço: {preco:,.2f}\n"
        f"🎯 Alvo: {alvo:,.2f}\n"
        f"🛑 Stop: {stop:,.2f}\n"
        f"📊 Veredito: {veredito}\n\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    enviar_telegram(msg)


def alerta_venda(ativo, preco_compra, preco_venda, lucro_rs, lucro_pct, motivo):
    emoji = "✅" if lucro_rs > 0 else "❌"
    msg = (
        f"{emoji} *VENDA* {emoji}\n\n"
        f"📌 Ativo: `{ativo}`\n"
        f"💵 Entrada: {preco_compra:,.2f}\n"
        f"💵 Saída: {preco_venda:,.2f}\n"
        f"💸 Lucro: R$ {lucro_rs:,.2f} ({lucro_pct:+.2f}%)\n"
        f"📋 Motivo: {motivo}\n\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    enviar_telegram(msg)


# ==========================================
# AUTO-TRADING
# ==========================================
def _calcular_alvo_stop(df):
    d = calcular_indicadores(df, coluna_preco="close")
    ultima = d.iloc[-1]
    preco = float(ultima["close"])
    atr = float(ultima["ATR"]) if pd.notna(ultima["ATR"]) else preco * 0.02
    return preco, preco + MULT_ALVO * atr, preco - MULT_STOP * atr, atr


def verificar_posicoes_abertas(tabela, dfs):
    portfolio = carregar_portfolio()
    abertas = portfolio[portfolio["Status"] == "ABERTO"]
    fechamentos = []
    for _, pos in abertas.iterrows():
        ativo = pos["Ativo"]
        df = dfs.get(ativo)
        if df is None or df.empty:
            continue
        preco_atual = float(df["close"].iloc[-1])
        alvo, stop = float(pos["Alvo"]), float(pos["Stop"])
        motivo = None
        if preco_atual >= alvo:
            motivo = f"🎯 Alvo ({preco_atual:.2f} ≥ {alvo:.2f})"
        elif preco_atual <= stop:
            motivo = f"🛑 Stop ({preco_atual:.2f} ≤ {stop:.2f})"
        else:
            linha = tabela[tabela["Ativo"] == ativo]
            if not linha.empty and "VENDER" in str(linha.iloc[0]["Veredito"]):
                motivo = "🔴 Reversão técnica"
        if motivo:
            print(f"   💼 FECHANDO {ativo} — {motivo}")
            res = registrar_venda(ativo, preco_atual)
            if res:
                alerta_venda(res["ativo"], res["preco_compra"], res["preco_venda"],
                             res["lucro_rs"], res["lucro_pct"], motivo)
            fechamentos.append(ativo)
    return fechamentos


def abrir_novas_posicoes(tabela, dfs):
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
        preco, alvo, stop, atr = _calcular_alvo_stop(df)
        qtd = round(VALOR_POR_TRADE / preco, 6)
        tipo = "Cripto" if ativo in CRIPTO_WATCHLIST else "Ação"
        print(f"   🟢 ABRINDO {ativo} @ {preco:.2f} | Alvo {alvo:.2f} | Stop {stop:.2f}")
        registrar_compra(ativo, tipo, round(preco, 2), qtd,
                         round(alvo, 2), round(stop, 2))
        alerta_compra(ativo, preco, alvo, stop, veredito)
        novas.append(ativo)
    return novas

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("=" * 75)
    print("🤖 ROBÔ TRADER — CICLO GITHUB ACTIONS")
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 75)

    print("\n🧠 Analisando ativos...\n")
    tabela, dfs = rodar_watchlist_completa()

    print("\n🔄 Verificando posições...\n")
    fechamentos = verificar_posicoes_abertas(tabela, dfs)
    novas = abrir_novas_posicoes(tabela, dfs)

    print("\n" + "=" * 75)
    print("📊 SINAIS POR ATIVO")
    print("=" * 75)
    print(tabela[["Ativo", "Preço", "RSI", "Score", "Veredito"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("💼 PORTFÓLIO")
    print("=" * 75)
    portfolio = carregar_portfolio()
    if portfolio.empty:
        print("Nenhuma operação registrada.")
    else:
        print(portfolio[["Ativo", "Tipo", "Preco_Compra", "Preco_Venda",
                         "Lucro_R$", "Lucro_%", "Status"]].to_string(index=False))

    print("\n" + "=" * 75)
    print(f"🟢 Compras: {', '.join(novas) if novas else 'nenhuma'}")
    print(f"🔴 Vendas : {', '.join(fechamentos) if fechamentos else 'nenhuma'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
