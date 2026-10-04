# Experimentos com TimesFM 2.5

Bancada de teste do modelo de fundação para séries temporais do Google (TimesFM 2.5, 200M, backend PyTorch, CPU) aplicada a séries de demanda e tráfego orgânico.

## Ambiente

```
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "timesfm[torch]" pandas openpyxl pytrends matplotlib --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `backtest.py` | Backtest rolling-origin (3 origens, horizonte 26 semanas) em séries semanais do Google Trends, contra baselines naive e sazonal |
| `trends_terms_weekly.csv` | Séries do Google Trends (Brasil, semanal, 5 anos) usadas no backtest; dado público |
| `backtest_results.csv` | Métricas por termo e origem (MASE, WQL, cobertura 80%) |
| `gsc_experiment.py` | Backtest diário, contrafactuais exploratórios e linha de base prospectiva sobre a série do Search Console |
| `gsc_backtest.csv` | Métricas do backtest diário (sem valores da série) |
| `baseline_90d_Cliques.csv`, `baseline_90d_Impressões.csv` | Projeção de 90 dias registrada em 4/out/2026, com quantis 10/50/90. Previsão falseável: verificar em jan/2027 |

Os arquivos com valores reais do Search Console (`cf_*.csv`, `gsc/`, gráfico) ficam fora do versionamento por `.gitignore`.

## Métricas

MASE: erro absoluto médio escalado pelo erro do baseline sazonal dentro da amostra; abaixo de 1 supera esse baseline. WQL: perda quantílica ponderada nos quantis 0,1 a 0,9. Cobertura 80%: fração de observações dentro do intervalo q10–q90.
