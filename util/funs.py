# project name: co_bc
# created by diego aliaga daliaga_at_chacaltaya.edu.bo

from useful_scit.imps import *


def test():
    pass


def get_bc_data(
        fold
):
    files = glob.glob(fold + '*.nas')

    file_data_frame = pd.DataFrame(files, columns=['path'])
    file_data_frame = file_data_frame.sort_values('path')
    # print(file_data_frame)
    file_data_frame['year'] = file_data_frame.path.str.extract(r'\.(20\d\d)').astype(int)

    file_data_frame['df'] = file_data_frame.apply(lambda r: read_maap(r), axis=1)

    joined_data_frame = pd.concat(list(file_data_frame['df']))

    joined_data_frame_1 = joined_data_frame[joined_data_frame.abs670 < 9999]

    j1 = joined_data_frame_1.shift(1, freq=pd.Timedelta(5, 'm')).resample('H')
    j2 = j1.mean()
    return j2


def read_maap(row):
    df = pd.read_csv(
        row.path,
        sep='\s+',
        skiprows=75
    )
    df['date'] = df.apply(lambda rr: get_date_from(rr.start_time, row.year), axis=1)
    df = df.set_index('date')
    return df


def get_date_from(day, year):
    d1 = pd.to_datetime(dt.date(year, 1, 1))
    dT = pd.Timedelta(day, unit='D')
    return d1 + dT


def read_co_data(
        file
):
    cols = ['year', 'month', 'day', 'hour', 'minute', 'CO_ppbv']
    d1 = pd.read_csv(file, sep='\s+', skiprows=1, names=cols, na_values='NaN')

    d1['date'] = pd.to_datetime(d1[['year', 'month', 'day', 'hour', 'minute']])
    d1 = d1.set_index('date')

    d1.loc[d1.CO_ppbv < -100, 'CO_ppbv'] = np.nan

    d2 = d1.CO_ppbv.shift(4, 'H').resample('10T').mean()
    d2 = d2.resample('1H').median()
    dd1 = d1.resample('1H').median()
    dd1['CO_ppbv'] = d2
    dd1 = dd1[['CO_ppbv']]
    return dd1


def open_sat_co(path):
    files = glob.glob(pjoin(path, '*.nc'))
    files = np.sort(files)
    xss = []
    for f in files:
        xs = xr.open_dataset(f)
        xss.append(xs)

    xc = xr.concat(xss, dim='time')
    return xc
