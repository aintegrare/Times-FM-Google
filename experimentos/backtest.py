import numpy as np, pandas as pd, timesfm, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('trends_terms_weekly.csv', index_col=0, parse_dates=True).drop(columns=['agência de marketing maringá'])
H = 26; M = 52
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True,
    use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
# sanity: which quantile column equals the point forecast?
x = df.iloc[:235, 1].values.astype(float)
p, q = model.forecast(horizon=H, inputs=[x])
print('col0==point?', np.allclose(p[0], q[0,:,0]), '| q shape', q.shape)
QS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
def wql(y, qf):  # qf: (H, 9) quantiles 0.1..0.9
    tot = 0
    for j, a in enumerate(QS):
        d = y - qf[:, j]; tot += np.sum(np.maximum(a*d, (a-1)*d))
    return 2*tot/ (len(QS)*np.sum(np.abs(y)))
rows = []
origins = [183, 209, 235]
for term in df.columns:
    y_all = df[term].values.astype(float)
    for o in origins:
        ctx, y = y_all[:o], y_all[o:o+H]
        scale = np.mean(np.abs(ctx[M:] - ctx[:-M]))  # in-sample seasonal naive MAE
        p, q = model.forecast(horizon=H, inputs=[ctx]); p = p[0]; qf = q[0,:,1:]
        naive = np.repeat(ctx[-1], H); snaive = ctx[-M:][:H]
        mae = lambda f: np.mean(np.abs(y-f))
        rows.append(dict(termo=term, origem=str(df.index[o].date()),
            mase_timesfm=mae(p)/scale, mase_naive=mae(naive)/scale, mase_snaive=mae(snaive)/scale,
            wql_timesfm=wql(y, qf), cobertura80=np.mean((y>=qf[:,0])&(y<=qf[:,8]))))
r = pd.DataFrame(rows); r.to_csv('backtest_results.csv', index=False)
pd.set_option('display.width', 200); print(r.round(3).to_string(index=False))
print('\nMédia por termo:'); print(r.groupby('termo')[['mase_timesfm','mase_naive','mase_snaive','wql_timesfm','cobertura80']].mean().round(3).to_string())
print('\nMédia geral:'); print(r[['mase_timesfm','mase_naive','mase_snaive','wql_timesfm','cobertura80']].mean().round(3).to_string())
