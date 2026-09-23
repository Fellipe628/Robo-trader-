name: Robo Trader

on:
  schedule:
    - cron: '0 3 * * *'         # 00h BRT — checagem diária
    - cron: '0 6 * * 1,4'       # 03h BRT segunda e quinta — TREINA LSTM
    - cron: '0 9,10 * * *'      # 06h, 07h BRT
    - cron: '0,30 11,12 * * *'  # 08h, 08h30, 09h, 09h30 BRT
    - cron: '*/15 13-19 * * *'  # 10h-16h45 BRT (PICO)
    - cron: '0 20-23 * * *'     # 17h-20h BRT
    - cron: '0 0-1 * * *'       # 21h-22h BRT
  workflow_dispatch:

permissions:
  contents: write

jobs:
  robo:
    runs-on: ubuntu-latest
    steps:
      - name: Baixar código
        uses: actions/checkout@v4

      - name: Configurar Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache de pacotes
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Instalar dependências
        run: pip install -r requirements.txt

      - name: Definir modo (treinar ou usar LSTM salvo)
        id: modo
        run: |
          if [[ "${{ github.event.schedule }}" == "0 6 * * 1,4" ]]; then
            echo "treinar=true" >> $GITHUB_OUTPUT
            echo "🎓 Modo TREINO ativado"
          else
            echo "treinar=false" >> $GITHUB_OUTPUT
            echo "💾 Modo CARREGAR LSTM salvo"
          fi

      - name: Rodar robô
        env:
          TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          TREINAR_LSTM: ${{ steps.modo.outputs.treinar }}
        run: python robo.py

      - name: Salvar dados atualizados
        run: |
          git config user.name "robo-trader-bot"
          git config user.email "bot@robo-trader.local"
          git add -A
          git commit -m "🤖 Atualiza dados [skip ci]" || echo "Sem mudanças"
          git push
