# project name: co_bc
# created by diego aliaga daliaga_at_chacaltaya.edu.bo
from useful_scit.imps import *
import funs


def test():
    pass


def get_maap_horiba_data(
):
    df_bc = funs.get_bc_data('../data/MAAP/')

    # %%
    df_abs = df_bc[['abs670']]

    # %%
    df_co = funs.read_co_data('../data/Chacaltaya_HORIBA_2013_2019_Local-Time.co')

    # %%
    df_co.columns

    # %%
    df_join = pd.merge(df_co, df_abs, how='outer', right_index=True, left_index=True)

    # %%
    df_join['both'] = False
    df_join.loc[
        (df_join['CO_ppbv'].isna() == False) &
        (df_join['abs670'].isna() == False),
        'both'
    ] = True
    return df_join


def plot_co_sat(
        xs_co
):
    cla, clo = -16.3472497, -68.1383381
    _m = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median('time')
    _c = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.count('time')
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    plt.style.use('seaborn')
    mpl.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    _name = 'CO [ppbv] (counts>30)'
    _m = _m.where(_c > 30)
    _m.name = _name
    _m.plot(ax=ax)
    ax.set_extent([-72, -60, -10, -25], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(crs=ccrs.PlateCarree(), color='grey', draw_labels=True)
    ax.scatter(
        [clo], [cla],
        color='red'
    )
    # ax.figure.set_size_inches(6,6)
    return ax


def process_df_hm(df_join):
    timy_things = ['year', 'month', 'day', 'hour']
    df_join1 = df_join.copy()
    for t in timy_things:
        df_join1[t] = getattr(df_join.index, t)
    _d_fake = df_join1.copy()
    _d_fake['day'] = 1
    df_join1['y_m'] = pd.to_datetime(_d_fake[['year', 'month', 'day']])
    return df_join1


def plot_mopit_vs_hor(
        xs_co, df_hm1
):
    _x = xs_co.MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.median(['lat', 'lon'])
    _x = _x.to_dataframe()
    _x.index = pd.to_datetime(_x.index.date)
    _x.index.name = 'y_m'

    _d = df_hm1.groupby('y_m')['CO_ppbv'].median()
    fig, ax = plt.subplots()
    _d.plot(ax=ax, label='CO [ppbv] at chc')
    (_x).MOP03JM_007_RetrievedCOSurfaceMixingRatioDay.plot(ax=ax, label='CO [ppbv] from MOPIT')
    # xd = _x.join(_d,how='outer')
    # xd.plot()
    ax.legend()


def plot_dists(
        year_list, df_hm1
):
    plt.style.use('seaborn')
    df_hm2 = df_hm1[df_hm1.CO_ppbv >= 0]
    df_hm2 = df_hm2[df_hm2.CO_ppbv <= 200]
    df_hm2 = df_hm2[df_hm2.year.isin(year_list)]

    fg = sns.FacetGrid(df_hm2, hue='year', col='month', col_wrap=4, sharey=False, size=2)

    def dp(*args, **kargs):
        #     print(args)
        ax = sns.distplot(*args, **kargs)
        ax.set_xlim(0, 200)

    fg.map(dp, 'CO_ppbv', bins=np.arange(0, 200, 5))
    fg.add_legend()
