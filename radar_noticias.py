    except ValueError:
        return value[:45]


def _score(article, category):
    title = _plain(article.get('title')).lower()
    description = _plain(article.get('description') or article.get('summary')).lower()
    text = title + ' ' + description
    assets = [ticker for ticker, terms in ASSETS.items() if any(term.lower() in text for term in terms)]
    event_hits = sum(1 for term in EVENTS if term in text)
    macro_hits = sum(1 for term in MACRO if term in text)
    if not assets and not macro_hits:
        return 0, []
    # Notícias de BTC precisam falar de BTC/ETH; macro Brasil precisa de contexto econômico.
    if category == 'BTC / cripto' and not any(x in assets for x in ('BTC', 'ETH')):
        return 0, []
    if category == 'Brasil / macro' and not macro_hits and not assets:
        return 0, []
    points = min(5, 3 * len(assets)) + min(3, event_hits) + min(2, macro_hits)
    return points, assets


def executar_radar(enviar=None):
    """Até 2 janelas UTC/dia, 2 consultas/janela; preserva estado v12."""
    key = os.getenv('APITUBE_KEY', '').strip()
    if not key:
        print('📰 Radar: APITUBE_KEY ausente; ignorando.')
        return
    now = _now()
    day = now.date().isoformat()
    state = _load()
    if state.get('day') != day:
        state['day'], state['requests'], state['runs'] = day, 0, []
    runs = state.get('runs') if isinstance(state.get('runs'), list) else []
    try:
        requests_today = max(0, int(state.get('requests', 0)))
    except (ValueError, TypeError):
        requests_today = MAX_REQUESTS_DAY  # estado inválido: falha segura
    slot = 'am' if now.hour < 12 else 'pm'
    if slot in runs or requests_today >= MAX_REQUESTS_DAY:
        print('📰 Radar: janela já executada ou limite interno atingido.')
        return
    seen = state.get('seen') if isinstance(state.get('seen'), dict) else {}
    candidates = []
    for topic, category in TOPICS:
        if requests_today >= MAX_REQUESTS_DAY:
            break
        requests_today += 1
        state['requests'] = requests_today
        _save(state)  # reserva a consulta antes de chamar a API
        try:
            response = requests.get(
                URL, params={'title': topic, 'per_page': 10},
                headers={'X-API-Key': key, 'Accept': 'application/json'}, timeout=15,
            )
            if response.status_code != 200:
                print(f'📰 APITube HTTP {response.status_code} ({category}); sem retry.')
                continue
            articles = _articles(response.json())
            print(f'📰 APITube {category}: {len(articles)} artigos recebidos.')
            for article in articles:
                if not isinstance(article, dict):
                    continue
                title = _plain(article.get('title'))[:180]
                if not title:
                    continue
                link = _url(article.get('url') or article.get('link') or article.get('article_url'))
                identifier = hashlib.sha256((link or title.lower()).encode('utf-8')).hexdigest()[:20]
                if identifier in seen:
                    continue
                # Preserva comportamento original: marca todos como vistos, mesmo os filtrados.
                seen[identifier] = day
                points, assets = _score(article, category)
                if points < 3:
                    continue
                source, source_link = _source(article)
                candidates.append({
                    'score': points, 'category': category, 'title': title,
                    'link': link or source_link, 'source': source,
                    'published': _date(article), 'assets': assets,
                })
        except (requests.RequestException, ValueError) as exc:
            print(f'📰 Falha APITube ({category}): {type(exc).__name__}')
    runs.append(slot)
    state['runs'] = runs
    state['seen'] = dict(list(seen.items())[-600:])
    _save(state)
    candidates.sort(key=lambda item: item['score'], reverse=True)
    print(f'📰 Radar v2: {len(candidates)} notícias relevantes inéditas; '
          f'{requests_today}/{MAX_REQUESTS_DAY} consultas reservadas hoje.')
    if not callable(enviar):
        return
    for item in candidates[:MAX_ALERTS]:
        msg = (
            '📰 RADAR DE NOTÍCIAS — CONTEXTO (v2)\n'
            '⚠️ APITube Free: acesso com atraso de 12h; não é alerta em tempo real.\n'
            f"Tema: {item['category']}\n"
            f"Ativos citados: {', '.join(item['assets']) if item['assets'] else 'contexto macro'}\n"
            f"Título: {item['title']}\n"
            f"Publicado: {item['published']}\n"
            f"Consultado: {now.strftime('%d/%m/%Y %H:%M UTC')}\n"
            f"Veículo: {item['source']}\n"
            f"Link: {item['link'] or 'não disponível'}\n"
            'Relevância: filtro textual (não mede impacto ou confiabilidade).\n'
            'Não é recomendação de compra ou venda.'
        )
        try:
            enviar(msg, markdown=False)
        except Exception as exc:
            print(f'📰 Falha ao enviar alerta: {type(exc).__name__}')
