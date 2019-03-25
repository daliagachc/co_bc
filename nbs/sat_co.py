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

# %%
path = '../data/co_sat/'

# %%
files = glob.glob(pjoin(path,'*.nc'))

# %%
files= np.sort(files)

# %%
f = files[0]
xss = []
for f in files:
    xs = xr.open_dataset(f)
    xss.append(xs)

# %%
xc = xr.concat(xss,dim='time')

# %%
for xla in xc[coc].transpose('lat','lon','time'):
    for xlo in xla:
        line=xlo.plot(color=[0,0,0,.1])
        line = line[0]
line.axes.grid()

# %%
xc[coc]

# %%
# gf = xc[coc].plot(col='lon',row='lat')
# axl = np.ndarray.flatten(gf.axes)
# for ax in axl:
#     ax.grid()


# %%
xc[coc].plot(col='time',col_wrap=12)

# %%

