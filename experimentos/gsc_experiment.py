import pandas as pd, numpy as np, timesfm, warnings, json
warnings.filterwarnings('ignore')
g = pd.read_csv('gsc/Gráfico.csv', parse_dates=['Data']).sort_values('Data').set_index('Data')
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128, normalize_inputs=True,
    use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
QS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
def fc(ctx, H):
    p,q = model.forecast(horizon=H, inputs=[np.asarray(ctx,float)]); return p[0], q[0,:,1:]
def wql(y,qf):
    t=0
    for j,a in enumerate(QS):
        d=y-qf[:,j]; t+=np.sum(np.maximum(a*d,(a-1)*d))
    return 2*t/(len(QS)*np.sum(np.abs(y)))

# 1) Backtest diário, H=28, origens mensais de out/2025 a set/2026, m=7
print('### 1. Backtest diário (H=28 dias)')
H=28; M=7; rows=[]
for col in ['Cliques','Impressões']:
    y_all=g[col].values.astype(float)
    for o in range(120, len(y_all)-H+1, 28):
        ctx,y=y_all[:o],y_all[o:o+H]
        scale=np.mean(np.abs(ctx[M:]-ctx[:-M]))
        p,qf=fc(ctx,H); naive=np.repeat(ctx[-1],H); snaive=np.tile(ctx[-M:],H//M+1)[:H]; mean28=np.repeat(ctx[-28:].mean(),H)
        mae=lambda f: np.mean(np.abs(y-f))
        rows.append(dict(serie=col, origem=str(g.index[o].date()), mase_timesfm=mae(p)/scale, mase_naive=mae(naive)/scale,
                         mase_snaive=mae(snaive)/scale, mase_media28=mae(mean28)/scale, wql=wql(y,qf), cob80=np.mean((y>=qf[:,0])&(y<=qf[:,8]))))
r=pd.DataFrame(rows); r.to_csv('gsc_backtest.csv',index=False)
pd.set_option('display.width',220)
print(r.round(2).to_string(index=False))
print('\nMédia por série:'); print(r.groupby('serie')[['mase_timesfm','mase_naive','mase_snaive','mase_media28','wql','cob80']].mean().round(3).to_string())
print('Vitórias do TimesFM sobre o melhor baseline:', int((r.mase_timesfm < r[['mase_naive','mase_snaive','mase_media28']].min(axis=1)).sum()), 'de', len(r))

# 2) Contrafactual exploratório nos dois marcos
print('\n### 2. Contrafactual exploratório')
def counterfactual(col, cut, H, label):
    y_all=g[col]; ctx=y_all[:cut].values.astype(float); y=y_all[cut:cut+H].values.astype(float)
    p,qf=fc(ctx,H)
    real, prev = y.sum(), p.sum()
    lo, hi = qf[:,0].sum(), qf[:,8].sum()
    out=dict(marco=label, serie=col, inicio=str(y_all.index[cut].date()), dias=H, real=int(real), previsto=int(prev),
             int80=f'{int(lo)}–{int(hi)}', diferenca=int(real-prev), dif_pct=round((real-prev)/prev*100,1),
             dias_fora_int80=int(np.sum((y<qf[:,0])|(y>qf[:,8]))))
    pd.DataFrame({'data':y_all.index[cut:cut+H],'real':y,'previsto':p,'q10':qf[:,0],'q90':qf[:,8]}).to_csv(f'cf_{label}_{col}.csv',index=False)
    return out
cf=[]
cut_dec = g.index.get_loc(pd.Timestamp('2025-11-15'))   # treino jun–meados nov; avalia queda
cut_sep = g.index.get_loc(pd.Timestamp('2026-09-01'))   # treino até ago; avalia salto
for col in ['Cliques','Impressões']:
    cf.append(counterfactual(col, cut_dec, 90, 'queda_dez25'))
    cf.append(counterfactual(col, cut_sep, 33, 'salto_set26'))
print(pd.DataFrame(cf).to_string(index=False))

# 3) Linha de base prospectiva: 90 dias a partir de 4/out/2026
print('\n### 3. Linha de base prospectiva (90 dias)')
base={}
for col in ['Cliques','Impressões']:
    ctx=g[col].values.astype(float); p,qf=fc(ctx,90)
    idx=pd.date_range(g.index[-1]+pd.Timedelta(days=1), periods=90)
    d=pd.DataFrame({'data':idx,'previsto':p,'q10':qf[:,0],'q50':qf[:,4],'q90':qf[:,8]}); d.to_csv(f'baseline_90d_{col}.csv',index=False)
    print(f'{col}: total 90d previsto {int(p.sum())} | int80 {int(qf[:,0].sum())}–{int(qf[:,8].sum())} | média diária {p.mean():.1f} vs últimos 28d {ctx[-28:].mean():.1f} vs últimos 90d {ctx[-90:].mean():.1f}')
# detalhe setembro
print('\nSetembro 2026, cliques diários:'); s=g.loc['2026-09','Cliques']; print(s.values.tolist())
