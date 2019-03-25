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

sns.set(style="darkgrid")

# %% [markdown]
# ## bc

# %% [markdown]
# ## data

# %%
fold = './data/MAAP/'

# %%
files = glob.glob(fold + '*.nas')

# %%
file_data_frame = pd.DataFrame(files, columns=['path'])
file_data_frame = file_data_frame.sort_values('path')
file_data_frame['year'] = file_data_frame.path.str.extract(r'\.(20\d\d)').astype(int)


# %%
def get_date_from(day, year):
    d1 = pd.to_datetime(dt.date(year, 1, 1))
    dT = pd.Timedelta(day, unit='D')
    return d1 + dT


# %%
def read_maap(row):
    df = pd.read_csv(
        row.path,
        sep='\s+',
        skiprows=75
    )
    df['date'] = df.apply(lambda rr: get_date_from(rr.start_time, row.year), axis=1)
    df = df.set_index('date')
    return df


# %%
file_data_frame['df'] = file_data_frame.apply(lambda r: read_maap(r), axis=1)

# %%
joined_data_frame = pd.concat(list(file_data_frame['df']))

# %%



# %%
joined_data_frame.describe()

# %%
joine_data_frame_1 = joined_data_frame[joined_data_frame.abs670 < 9999]

# %%
# %matplotlib inline
joined_data_frame.numflag.hist()
ax = plt.gca()
ax.set_title('avail data');

# %%
joine_data_frame_1.numflag.hist()

# %%
joine_data_frame_1.resample('H').count().iloc[:, 1].hist()

# %%
j1 = joine_data_frame_1.shift(1, freq=pd.Timedelta(5, 'm')).resample('H')
j1.count().iloc[:, 1].hist()

# %%
j2 = j1.mean()
j2[:2]

# %%
j2['month'] = j2.index.month
j2['hour'] = j2.index.hour
j2['year'] = j2.index.year
j2['day'] = j2.index.date
j2['abs670_log']=np.log10(j2.abs670)

# %%
sns.distplot(j2.abs670_log[j2.abs670_log>-3].dropna())

# %%
g = sns.FacetGrid(j2, row="year", col="month", margin_titles=True,height=2,aspect=1)
bins = np.arange(-1, 1, .2)
gmap = g.map(plt.hist, "abs670_log", color="steelblue", bins=bins,orientation='horizontal')
# axs = gmap.axes
# axl = np.ndarray.flatten(axs)
# for ax in axl:
#     ax.set_yscale('log')

# %%
# %matplotlib inline
ax = sns.boxenplot(x='month', y='abs670', data=j2)
ax.set_ylim(0, 6)

# %%
# %matplotlib inline
ax = sns.boxenplot(x='year', y='abs670', data=j2)
ax.set_ylim(0, 2.5)

# %%
# %matplotlib inline
ax = sns.boxenplot(x='month', y='abs670', data=j2)
ax.set_ylim(0, 2.5)

# %%
# %matplotlib inline
ax = sns.boxenplot(x='hour', y='abs670', data=j2)
ax.set_ylim(0, 2.5);

# %%
# %matplotlib inline
ax = sns.boxenplot(x='hour', y='abs670', data=j2)
ax.set_ylim(.1, 4.5);
ax.set_yscale('log')

# %%
# %matplotlib inline
# # %matplotlib widget
fig, axs = plt.subplots(2, 4, sharey=True, sharex=False, figsize=(16, 8))
axs = np.ndarray.flatten(axs)
ys = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
for i, year in enumerate(ys):
    ax = axs[i]
    ax = sns.boxenplot(x='month', y='abs670', data=j2[j2.year == year], ax=ax,
                       #                      color = 'grey'
                       )
    ax.set_ylim(.1, 5);
    ax.set_xlim(-1, 12);
    ax.set_yscale('log')
    ax.set_title(year)
fig.tight_layout()

# %%
# %matplotlib inline
fig, axs = plt.subplots(2, 4, sharey=True, sharex=False, figsize=(16, 8))
axs = np.ndarray.flatten(axs)
ys = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
for i, year in enumerate(ys):
    ax = axs[i]
    ax = sns.boxenplot(x='month', y='abs670', data=j2[j2.year == year], ax=ax, color='grey')
    ax.set_ylim(.1, 4);
    ax.set_xlim(-1, 12);
    ax.set_yscale('linear')
    ax.set_title(year)
fig.tight_layout()

# %%

isna = j2.abs670.isna()
isna.value_counts().plot.bar()

# %%
n3 = len(j2)
l = []
for i in range(1, 2 * 24):
    j3 = j2.abs670.interpolate(limit=i)
    isna = j3.isna()
    ll = isna.value_counts()
    l.append([i, ll[False] / n3])

# %%
plt.bar(*np.array(l).T)

# %%
j2[['abs670', 'day']].groupby('day').count().plot()

# %%
j2[['abs670', 'day']].groupby('day').count().plot.hist(bins=range(25))

# %%
j2[['abs670', 'day']].groupby('day').count().plot.hist()

# %%
j2['abs670_filt']=j2.abs670
j2.loc[(
    (j2.abs670<=0.01) &
    (j2.abs670>=0.00)
),'abs670_filt']=0.01
j2.loc[j2.abs670<0,'abs670_filt']=np.nan
j2['abs670_log']=np.log10(j2.abs670_filt)


j2.abs670_log.hist()

# %%
j2['abs670_log_g']=j2.abs670_log.interpolate(limit=2)
j2['abs670_log_g']=j2['abs670_log_g'].rolling(4,min_periods=2,center=True,win_type='gaussian').mean(std=2)
j2.abs670_log_g[2000:2200].plot()
j2.abs670_log[2000:2200].plot()

# %%
jgbd = j2.groupby('day')
pdfs = [df for i,df in jgbd]
for d in pdfs[::20]:
    d1 = d.set_index('hour')
    d1.abs670_log_g.plot()

# %%



# %%
j22 = j2.shift(12, 'H')
j22['day'] = j22.index.date
j22['pos'] = j22.index.hour
j22 = j22.shift(-12, 'H')
j22['d'] = j22.index.day

jd = j22[['abs670_log_g', 'day']].groupby('day').count().reset_index()
jd['n'] = jd['abs670_log_g']
j22 = j22.reset_index()
j22 = j22.set_index('day')
# j22['day']=pd.to_datetime(j2.day)
# jd['day']=pd.to_datetime(jd.day)
jd = jd.set_index('day')
jj = jd[['n']].join(j22, how='outer')
jj = jj.reset_index()
jj['year'] = jj['year'].astype(int)
jj['month'] = jj['month'].astype(int)
jj['hour'] = jj['hour'].astype(int)

# jj = jj.set_index(['year','month','d','hour'])[['n','abs670','day']]
jj = jj.set_index(['day', 'pos'])[['year', 'month', 'd', 'n', 'abs670_log_g', 'hour']]

# %%
j_clean = j22.reset_index()
j_clean['datetime']=j_clean['date']
j_clean['date_pos']=j_clean['day']
j_clean = j_clean.set_index('datetime')
j_clean['date']=j_clean.index.date
j_clean['day']=j_clean['d']
j_clean = j_clean.drop(
    [
        'd','T_int','ed_y','abs670pc16','abs670pc84','p_int','st_y',
        'end_time','numflag','start_time'
    ],
    axis=1
)

j_clean.sample()

# %%


# %%
j_clean.sample()

# %%
ds = xr.Dataset.from_dataframe(jj)

# %%
ds

# %%
ds1 = ds.dropna('day')
ds1

# %%
ds1.isel(day=slice(0, -1))['abs670_log_g'].plot(x='day', robust=True)

# %%



# %% [markdown]
# ###### Let try to normalize the vectors 

# %%
from sklearn.preprocessing import normalize

# %%
ds1['abs_norm_log'] = (['day', 'pos'], normalize(ds1.abs670_log_g))

# %%
sns.boxenplot(
    x='pos',
    y='abs_norm_log',
    data=ds1['abs_norm_log'].to_dataframe().reset_index()
)
ax = plt.gca()
ax.set_ylim(-.25, .5)

# %%



# %%
sns.boxenplot(
    x='pos',
    y='abs670_log_g',
    data=ds1['abs670_log_g'].to_dataframe().reset_index()
)
ax = plt.gca()
ax.set_ylim(-1, 1)

# %%
ds1['abs670_g']=10**ds1['abs670_log_g']
ds1['abs_norm_g'] = (['day', 'pos'], normalize(ds1.abs670_g))
sns.boxenplot(
    x='pos',
    y='abs670_g',
    data=ds1['abs670_g'].to_dataframe().reset_index(),
    color = 'grey'
)
ax = plt.gca()
ax.set_ylim(0, 12);

# %%
# ds1['abs_norm1']=(['day','pos'],normalize(ds1.sel(pos=3).abs670))
# tot = np.sum(normalize(ds1.abs_norm.sel(pos=slice(3,200))),axis=1)
tot = np.linalg.norm(ds1.abs670_log_g.sel(pos=slice(4, 200)), axis=1)
ds1['tot_norm_4_log'] = ('day', tot)
ds1['abs_norm_4_log'] = ds1.abs670_log_g / ds1.tot_norm_4_log

tot = np.linalg.norm(ds1.abs670_g.sel(pos=slice(4, 200)), axis=1)
ds1['tot_norm_4_g'] = ('day', tot)
ds1['abs_norm_4_g'] = ds1.abs670_g / ds1.tot_norm_4_g

# %%
# len(tot)

# %%



# %%
sns.boxenplot(
    x='pos',
    y='abs_norm_4_log',
    data=ds1['abs_norm_4_log'].to_dataframe().reset_index()
)
ax = plt.gca()
ax.set_ylim(-1, .55)

# %%
sns.boxenplot(
    x='pos',
    y='abs_norm_4_g',
    data=ds1['abs_norm_4_g'].to_dataframe().reset_index()
)
ax = plt.gca()
ax.set_ylim(0, .55)

# %%
sns.boxenplot(
    x='pos',
    y='abs_norm',
    data=ds1['abs_norm'].to_dataframe().reset_index()
)
ax = plt.gca()
ax.set_ylim(-1, .55)

# %%
ds1['abs670_g'].isel(pos=slice(4,200))

# %%
from sklearn.cluster import KMeans

# %%
ucp.cl

# %%
from sklearn.metrics import silhouette_score
sc = []
for nc in range(2,20):
    X=ds1['abs_norm_4_g'].isel(pos=slice(4,200))
    kmeans = KMeans(n_clusters=nc, random_state=np.random.randint(0,1000)).fit(X)
    score = silhouette_score(X,kmeans.labels_)
    sc.append([nc,score])

# %%
plt.plot(*np.array(sc).T)

# %%
nc = 4
X=ds1['abs_norm_4_g'].isel(pos=slice(4,200))
kmeans = KMeans(n_clusters=nc, random_state=np.random.randint(0,1000)).fit(X)
score = silhouette_score(X,kmeans.labels_)
sc.append([nc,score])

# %%
silhouette_score(X,kmeans.labels_)

# %%
labs = kmeans.labels_
un = np.unique(labs,return_counts=True)
plt.bar(*un,color=ucp.cl)

# %%
ds1['lab']=('day',kmeans.labels_)

# %%
def pl_nc(nc,ax, lab='s',par='abs_norm_4_g',yr=(-.5,.5),col='k'):
    dd=ds1.where(ds1.lab==nc)
    dd=dd[par].dropna(dim='day')
    dd=dd.to_dataframe().reset_index()
    sns.boxenplot(x='pos',y=par,data=dd,ax=ax,color=col)
#     ax = plt.gca()
    ax.set_ylim(*yr)
    ax.set_title(i)

# %%
fig,axs=plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,5))
axl = np.ndarray.flatten(axs)

for i in range(nc):
#     print(i)
#     fig,ax=plt.subplots()
    pl_nc(i,axl[i],lab=i,yr=(0,1),col=ucp.cl[i])
for a in np.ndarray.flatten(axs[:,-1:]): a.set_ylabel('')
for a in np.ndarray.flatten(axs[:-1,:]): a.set_xlabel('')

fig.tight_layout()

# %%
fig,axs=plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,10))
axl = np.ndarray.flatten(axs)

for i in range(nc):
#     print(i)
#     fig,ax=plt.subplots()
    pl_nc(i,axl[i],lab=i,par = 'abs670_g',yr=(0,12),col=ucp.cl[i])
for a in np.ndarray.flatten(axs[:,-1:]): a.set_ylabel('')
for a in np.ndarray.flatten(axs[:-1,:]): a.set_xlabel('')

fig.tight_layout()

# %%
fig,axs=plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,10))
axl = np.ndarray.flatten(axs)

for i in range(nc):
#     print(i)
#     fig,ax=plt.subplots()
    pl_nc(i,axl[i],lab=i,par = 'abs670_log_g',yr=(-1,1),col=ucp.cl[i])
for a in np.ndarray.flatten(axs[:,-1:]): a.set_ylabel('')
for a in np.ndarray.flatten(axs[:-1,:]): a.set_xlabel('')

fig.tight_layout()

# %%
va = ds1.where(True).month[:,1]
va = va.dropna('day')
va.plot.hist(bins = np.arange(.5,13,1))
tot_days = pd.DataFrame(*np.flip(np.unique(va,return_counts=True),axis=0))

# %%
def pl_his(ds1,lab,ax,col,tot):
    va = ds1.where(ds1.lab==lab).month[:,1]
    va = va.dropna('day')
    va = pd.DataFrame(*np.flip(np.unique(va,return_counts=True),axis=0))
    va = va / tot
    va = va.reset_index().T.values
#     print(va)
    ax.bar(*va,color = col)
    ax.set_xticks(np.arange(2,12,2));

# %%
cols = 2 
rows = int(np.ceil(nc/cols))


# %%
fig, axs = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(cols*3*2,rows*3))
axl = np.ndarray.flatten(axs)
for i in range(nc):
    pl_his(ds1,i,axl[i],ucp.cl[i], tot_days)

# %%
ds1


# %%
hour_utc = ds1.hour.median('day').astype(int)
hour_utc.name='hour_utc'
ds1 = ds1.assign_coords(hour_utc=hour_utc)
hour_loc = ds1.hour.median('day').astype(int)-4
hour_loc.name='hour_loc'
ds1=ds1.assign_coords(hour_loc=hour_loc)

# %%
ds1

# %%
path_save = './data/bc_data_v01.h5'
# ds1.to_netcdf(path_save)

# %%



# %%



# %%



# %%



# %%



# %%



