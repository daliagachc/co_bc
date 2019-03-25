# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from useful_scit.imps import *
# plt.style.use('Solarize_Light2')
plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100
# mpl.rcParams['figure.figsize']=[4.0, 3.0]
# # %config InlineBackend.figure_format = 'retina'

# %%
# mpl.rcParams

# %%
sys.path.insert(0,'../util/')
import funs
importlib.reload(funs);

# %%
df_bc = funs.get_bc_data('../data/MAAP/')

# %%
df_abs = df_bc[['abs670']]

# %%
df_co = funs.read_co_data('../data/Chacaltaya_HORIBA_2013_2019_Local-Time.co')

# %%
df_co.columns

# %%
df_join = pd.merge(df_co,df_abs,how='outer',right_index=True,left_index=True)

# %%
df_join['both']=False
df_join.loc[
    (df_join['CO_ppbv'].isna()==False) &
    (df_join['abs670' ].isna()==False),
    'both'
] = True
    

# %%
df_join.both.describe()

# %%
jp=sns.jointplot(
    'CO_ppbv','abs670',
    data=df_join[df_join.both]['2014-01'],
    color=[0,0,0,.2]
)

jp.ax_joint.set_ylim(-.5,4)
# jp.ax_joint.set_yscale('log')
jp.ax_joint.set_xlim(30,140)
# jp.ax_joint.set_xscale('log')
# jp.fig

# %%
ax=sns.distplot(df_join.CO_ppbv.dropna(),bins=200)
ax.set_xlim(0,200)

# %%
ax=sns.distplot(df_join.abs670.dropna(),bins=200)
ax.set_xlim(-1,5)

# %%
timy_things = ['year','month','day','hour']
df_join1 = df_join.copy()
for t in timy_things:
    df_join1[t] = getattr(df_join.index,t)

# %%
_d_fake = df_join1.copy()
_d_fake['day']=1
df_join1['y_m']=pd.to_datetime(_d_fake[['year','month','day']])

# %%
dg = df_join1.groupby('y_m')
dg.CO_ppbv.median().plot()

# %%
d1,d2 = 5,10
n1,n2 = 15,18
days = 30
par = 'abs670'
def comb_plot(d1,d2,n1,n2,days,par,l1,l2,ax=False):
    if ax==False: 
        fig, ax=plt.subplots()
    ax=df_join1[(df_join1.hour>=d1)&(df_join1.hour<=d2)][par].resample('1H').mean().rolling(24*days,min_periods=5).median  (   ).plot(ax=ax, label='night_median')
    ax=df_join1[(df_join1.hour>=d1)&(df_join1.hour<=d2)][par].resample('1H').mean().rolling(24*days,min_periods=5).quantile(.75).plot(ax=ax, label='night_90    ')
    ax=df_join1[(df_join1.hour>=n1)&(df_join1.hour<=n2)][par].resample('1H').mean().rolling(24*days,min_periods=5).median  (   ).plot(ax=ax, label='day_median  ')
    ax=df_join1[(df_join1.hour>=n1)&(df_join1.hour<=n2)][par].resample('1H').mean().rolling(24*days,min_periods=5).quantile(.75).plot(ax=ax, label='day_90      ') 
    
#     ax.grid(which='major',color=[0,0,0,1]    )
#     ax.grid(which='minor',color=[.8,.8,.8,1] )
    ax.set_ylim(l1,l2)
    return ax
# %matplotlib inline
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2,sharex=ax1)
comb_plot(d1,d2,n1,n2,days,par,0,10,ax1)
# plt.subplots()
par = 'CO_ppbv'
comb_plot(d1,d2,n1,n2,days,par,30,300,ax2)
fig.tight_layout()
fig.set_figwidth(15)
fig.set_figheight(10)
ax1.legend()
ax2.legend()


# %%
xs_co = funs.open_sat_co('../data/co_sat/')

# %%
cla,clo=-16.3472497,-68.1383381

# %%
_m = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median('time')
_c = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.count('time')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
_name='CO [ppbv] (counts>30)'
_m = _m.where(_c>30)
_m.name = _name
_m.plot(ax=ax)
ax.set_extent([-72, -60, -10, -25], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(crs=ccrs.PlateCarree(),color = 'grey',draw_labels=True)
ax.scatter(
    [clo], [cla], 
    color = 'red'
)
# ax.figure.set_size_inches(6,6)

# %%
_c.plot()

# %%
xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.sel(lat=-18.,lon=-68.).plot()

# %%
ax = sns.boxenplot(
    x='hour',
    y='CO_ppbv',
    data=df_join1,
    
)
ax.set_ylim(0,200)

# %%
df_join1.sample()

# %%
_x = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median(['lat','lon'])
_x = _x.to_dataframe()
_x.index = pd.to_datetime(_x.index.date)
_x.index.name='y_m'

# %%


# %%
_d = df_join1.groupby('y_m')['CO_ppbv'].median()
# pd.to_datetime(_x.index)

# %%
plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots()
# df_join1.groupby('y_m')['CO_ppbv'].median().plot(ax=ax)
# xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median(['lat','lon']).plot(ax=ax)
_d.plot(ax=ax,label = 'CO [ppbv] at chc')
(_x+25).MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.plot(ax=ax, label= 'CO[ppbv] from MOPIT')
# xd = _x.join(_d,how='outer')
# xd.plot()

# %%
_x.join(_d)

# %%
_x

# %%
d1 = '2013-03-01'
ddc = ddn = df_join1.CO_ppbv.dropna()
ddc =ddc[(ddc<200) & (ddc>0)]
ddn = ddn[ddn.index<d1]
ddn = ddn[(ddn<200) & (ddn>0)]
fig, ax = plt.subplots()
sns.distplot(ddn,ax=ax)
sns.distplot(ddc,ax=ax)

# %%
_ = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.values
vals = np.ndarray.flatten(_)
vals = vals[np.isnan(vals)==False]

# %%
sns.distplot(vals)

# %%
g = sns.FacetGrid(df_join1[(df_join1.CO_ppbv>0)&(df_join1.CO_ppbv<200)],col='month',row='year')
g.map(sns.distplot,'CO_ppbv')


# %%
for a in g.axes.flatten():
    sns.distplot(ddc,ax=a,color=ucp.cl[2])

# %%
g.fig

# %%
cok = 'CO_ppbv'
cos = 'CO [ppbv] MOPIT' 
df_hor = df_join1.groupby('y_m')[cok].median()
coh = 'CO [ppbv] HORIB'
df_hor.name = coh
df_sat = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median(['lat','lon'])
df_sat.name = cos
df_sat = df_sat.to_dataframe()
df_sat = df_sat.resample('m').mean()
df_sat.index = df_sat.index + pd.Timedelta(1,'D')
df_sh = df_sat.join(df_hor,how='outer')
# df_sat[~np.isnan(df_sat)]
df_sh['y_m']=df_sh.index.strftime('%Y-%m')
df_sh['year']=df_sh.index.strftime('%Y-01')
# df_sh

# %%
fg = sns.relplot(coh,cos,data=df_sh, 
            hue='year',
#             style='year',
#             hue_order = [2012,2013,2014,2015,2016,2018,2019]
           )

fg.ax.plot([50,180],[50,180])

# %%
df_sh['month']=df_sh.index.month

# %%
sns.relplot('month',cos,data=df_sh)

# %%
sns.relplot('month',coh,data=df_sh[~df_sh[coh].isna()],hue='year',kind='scatter',markers=True)

# %%
df_join1.month.unique()

# %%
def p_dist(month,year,ax,df_join1):
    b1 = df_join1.month == month
    b2 = df_join1.year  == year
    cok = 'CO_ppbv'
    d1 = df_join1[b1 & b2][cok]
    d1=d1.dropna()
    d1 = d1[d1<=200]
    d1 = d1[d1>=0  ]
    sns.distplot(d1,ax=ax,label=year)
    ax.legend()
    ax.set_xlim(0,200)
    ax.set_xlabel('')
    return ax


fig, axs = plt.subplots(12,1,sharex=True,figsize=(4,25))

years = 2013,2014,2015,2016,2017,2018

i = 0 
axl = axs.flatten()
for m in range(1,13):
    for y in range(len(years)):
        ax=p_dist(m,years[y],axl[m-1],df_join1)    
    ax.set_title(m)
    
        
plt.close(fig)


# %%
fig

# %%
fig, ax = plt.subplots()
years = 2013,2014,2015,2016,2017,2018
  
for y in years:
    b1 = df_join1.month == 5 
    b2 = df_join1.year  == y
    cok = 'abs670'
    d1 = df_join1[b1 & b2][cok]
    d1=d1.dropna()
    d1 = d1[d1<=20  ]
    d1 = d1[d1>=.1  ]
    d1 = np.log10(d1)
    d1.name = '${\log}_{10}({abs}_{670})$'
    sns.distplot(d1,ax=ax,label=y)
    ax.legend()

# %%
fig, ax = plt.subplots()
years = 2014,2015,2016,2017,2018
  
for y in years:
    b1 = df_join1.month == 10 
    b2 = df_join1.year  == y
    cok = 'abs670'
    d1 = df_join1[b1 & b2][cok]
    d1=d1.dropna()
    d1 = d1[d1<=20  ]
    d1 = d1[d1>=.1  ]
    d1 = np.log10(d1)
    d1.name = '${\log}_{10}({abs}_{670})$'
    sns.distplot(d1,ax=ax,label=y)
    ax.legend()

# %%
fig, ax = plt.subplots()
for y in years:
    ax = p_dist(4,y,ax,df_join1)

# %%
years = 2013,2014,2015,2016,2017,2018
d2 = df_join1.copy()
cok = 'CO_ppbv'
d2.loc[d2.year==2017,cok] = d2.loc[d2.year==2017,cok]/2+80
d2.loc[d2.year==2016,cok] = d2.loc[d2.year==2016,cok]/2+80
d2.loc[d2.year==2018,cok] = d2.loc[d2.year==2018,cok]/3+105
d2.loc[d2.year==2013,cok] = d2.loc[d2.year==2013,cok]/2+83

fig, ax = plt.subplots()
for y in years:
    ax = p_dist(5,y,ax,d2)

# %%
def p_dist(month,year,ax,df_join1):
    b1 = df_join1.month == month
    b2 = df_join1.year  == year
    cok = 'CO_ppbv'
    d1 = df_join1[b1 & b2][cok]
    d1=d1.dropna()
    d1 = d1[d1<=200]
    d1 = d1[d1>=0  ]
    sns.distplot(d1,ax=ax,label=year)
    ax.legend()
    ax.set_xlim(0,200)
    ax.set_xlabel('')
    return ax


fig, axs = plt.subplots(12,1,sharex=True,figsize=(4,25))

years = 2013,2014,2015,2016,2017,2018

i = 0 
axl = axs.flatten()
for m in range(1,13):
    for y in range(len(years)):
        ax=p_dist(m,years[y],axl[m-1],d2)    
    ax.set_title(m)
    
        
plt.close(fig)

# %%
fig

# %%
d2 = df_join1.copy()
ra = np.arange(-50,500,.5)
va7 = d2.loc[(d2.month==5) &(d2.year==2018),cok].dropna()
va5 = d2.loc[(d2.month==5) &(d2.year==2015),cok].dropna()

# %%
hi5=np.histogram(va5,ra)[0]
hi5 = hi5/hi5.sum()

def optimize_me(x):
    hi7=np.histogram(va7/(x[0])+x[1],ra)[0]
    hi7 = hi7/hi7.sum()
    return ((hi7-hi5)**2).sum()

# %%
from scipy.optimize import brute

# %%
brute(optimize_me,(slice(1.5,4,.01),slice(60,120,1)))

# %%

