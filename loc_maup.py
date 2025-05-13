import csv
import matplotlib.pyplot as plt
from math import cos, pi
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm
from vedo import Points, show, Grid, Plotter, Line, Mesh
import vedo
from shapely.wkt import loads
csv.field_size_limit(1024 * 1024 * 100)


def read_cover(file, minpop=0):
    # reading the cover data, which is extracted from trajectories
    # the first column is the potential location (geographic cells),
    # the second column is the set of individual ids covered by this potential location
    cid_cover = {}  # 即格子覆盖人数，用minpop筛选
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            if len(eval(row[1])) >= minpop:
                cid_cover[eval(row[0])] = eval(row[1])
    print('cover read')
    return cid_cover



def get_store(xgap, ygap, ynum,xmin=113.67561783007596,ymax=22.852485545898546):
    #extracting current store locations from poi data, and converting them to cell ids
    res = []
    cids = []
    tags = set()
    with open(r'E:\基础数据\aoipoi\深圳\bmap-poi-深圳市.csv', 'r', encoding='utf-8') as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            # if row[1].find('vivo')!=-1 and row[2] in ['购物', '公司企业']:
            if row[1].find('vivo') != -1:
                tags.add(row[2])
                # print(row)
                x, y = float(row[-2]), float(row[-1])
                xid, yid = int((x - xmin) / xgap), int((ymax - y) / ygap)
                cid = xid * ynum + yid + 1
                if cid not in cids:
                    res.append([cid, x, y])
                    cids.append(cid)
    print(len(res), tags)
    return res


def solve_greed_nodup(cover, k, title):
    # solving the location choice problem
    loc = []
    nums = []
    for i in range(k):
        temp = {}
        for cid in cover:
            temp[cid] = len(cover[cid])
        stemp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        cid0 = stemp[0][0]
        loc.append(cid0)
        nums.append(stemp[0][1])
        covered = cover[cid0]
        cover1 = {}
        for cid in cover:
            if cid != cid0:
                cover1[cid] = cover[cid] - covered
        cover = cover1
    np.save(title, np.array(loc))
    return loc, nums


def get_nodup_curve(cover, cids):
    # for plotting the curve in fig4.b
    temp = {}
    for cid in cids:
        if cid in cover:
            temp[cid] = cover[cid]
        else:
            temp[cid] = set()
    loc1, nums = solve_greed_nodup(temp, len(cids), r'E:\门店选址\20250402算法数据验证\temp.npy')
    return nums


def get_dup_curve(cover, cids):
    # for plotting the curve in fig4.a
    temp = {}
    for cid in cids:
        if cid in cover:
            temp[cid] = len(cover[cid])
        else:
            temp[cid] = 0
    stemp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
    nums = []
    for i in stemp:
        nums.append(i[1])
    return nums


def plot_curve_dup_nodup(store, cover, loc, title, title1):
    # plotting fig4, given cover data, current stores, and optimal location choices
    res = []
    sto = []
    for _ in store:
        sto.append(_[0])
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('rank', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_ylabel('Number of people covered by the store site', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    xs, ys, xo, yo = range(len(loc)), [], range(len(loc)), []
    ys = get_dup_curve(cover, sto)
    yo = get_dup_curve(cover, loc)
    loc1 = []
    temp = {}
    for cid in cover:
        temp[cid] = len(cover[cid])

    stemp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
    nums = []
    for i in range(len(loc)):
        nums.append(stemp[i][1])
    yo = nums
    res += [ys, yo]
    plt.plot(xs, ys, marker='s',  # 方形标记
             markersize=4,  # 标记大小
             markerfacecolor='blue',  # 标记填充色
             markeredgecolor='blue',  # 标记边缘色
             markeredgewidth=0,  # 标记边缘宽度
             linestyle='-',  # 实线
             linewidth=2,  # 线宽
             color='blue',  # 线条颜色
             label='Current store locations')
    plt.plot(xo, yo, marker='s',  # 方形标记
             markersize=4,  # 标记大小
             markerfacecolor='red',  # 标记填充色
             markeredgecolor='red',  # 标记边缘色
             markeredgewidth=0,  # 标记边缘宽度
             linestyle='-',  # 实线
             linewidth=2,  # 线宽
             color='red',  # 线条颜色
             label='Optimized store locations')
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('rank',fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_ylabel('Number of people covered by the store site', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    xs, ys, xo, yo = range(len(loc)), [], range(len(loc)), []
    ys = get_nodup_curve(cover, sto)
    yo = get_nodup_curve(cover, loc)
    res += [ys, yo]
    plt.plot(xs, ys, marker='s',  # 方形标记
             markersize=4,  # 标记大小
             markerfacecolor='blue',  # 标记填充色
             markeredgecolor='blue',  # 标记边缘色
             markeredgewidth=0,  # 标记边缘宽度
             linestyle='-',  # 实线
             linewidth=2,  # 线宽
             color='blue',  # 线条颜色
             label='Current store locations')
    plt.plot(xo, yo, marker='s',  # 方形标记
             markersize=4,  # 标记大小
             markerfacecolor='red',  # 标记填充色
             markeredgecolor='red',  # 标记边缘色
             markeredgewidth=0,  # 标记边缘宽度
             linestyle='-',  # 实线
             linewidth=2,  # 线宽
             color='red',  # 线条颜色
             label='Optimized store locations')
    plt.legend()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(title, dpi=300)
    plt.show()
    np.save(title1, np.array(res))
    return


def powerlaw_analysis(xdata, ydata):
    # fitting the truncated power law curves
    def trun_power1(x, a, b, c, d, e):
        return a * np.power((c + x), -b) * np.exp(-1 * x / d) + e


    xdata = np.array(xdata)
    popt, pcov = curve_fit(trun_power1, xdata, ydata, \
                           bounds=(np.array([0, 0, 0, 0, -np.inf]), \
                                   np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
                                   ), \
                           maxfev=100000)

    y2 = [trun_power1(i, popt[0], popt[1], popt[2], popt[3], popt[4]) for i in xdata]
    print('trun_power', popt, r2_score(ydata, y2))
    x = np.log(np.array(xdata))
    x = sm.add_constant(x)
    y = np.log(np.array(ydata))
    res = sm.OLS(y, x).fit()
    print('power')
    print(res.summary())
    return


def fit_power_curve(file, k):
    res = np.load(file).tolist()
    print(res)
    print('---------')
    print('store greedy')
    powerlaw_analysis(range(1, k + 1), res[0])
    print('---------')
    print('greedy')
    powerlaw_analysis(range(1, k + 1), res[1])
    print('---------')
    print('store nodup')
    powerlaw_analysis(range(1, k + 1), res[2])
    print('---------')
    print('greedy nodup')
    powerlaw_analysis(range(1, k + 1), res[3])
    return


def cid_to_xy(cid, ynum):
    xid = int((cid - 1) / ynum)
    yid = cid - 1 - xid * ynum
    return xid, yid


def maup_data_single(cover, scale):
    # 输入的cover字典，key是格子行列编号以便计算
    # 输出的字典同样以行列编号作为key
    # scale决定聚合大小，即scale*scale的正方形形式聚合，起点为xid=0,yid=0即地图西北角
    # transforming the cover data to apply to maup
    # merging cells in a square form, and merging the individual sets covered by the cells
    coverk = {}
    for cid in cover:
        xid, yid = cid
        xid1, yid1 = int(xid / scale), int(yid / scale)
        coverk[(xid1, yid1)] = coverk.get((xid1, yid1), set()) | cover[cid]
    return coverk


def output_cover(title, cover):
    with open(title, 'w', newline='') as f:
        wt = csv.writer(f)
        for cid in cover:
            wt.writerow([cid, cover[cid]])
    return


def maup_data(file, maxs, dirs, ynum):
    # file表示读取的cover原始字典
    # maxs是最大聚合尺度，采用栅格正方形聚合
    # dirs是存储路径
    # 读取cover，将其栅格id转化为行列编号，输出；而后调用maup_data_single按尺度分别输出
    # forming data for the maup
    cover0 = read_cover(file)
    cover1 = {}
    for cid in cover0:
        xid, yid = cid_to_xy(cid, ynum)
        cover1[(xid, yid)] = cover0[cid]
    print('xyid ready')
    for scale in range(2, maxs + 1):
        cover = maup_data_single(cover1, scale)
        title = r'{dirs}\cover_scale{scale}.csv'.format(dirs=dirs, scale=scale)
        output_cover(title, cover)
        print('scale', scale)
    return


def maup_solve(dirs, maxs, k):
    # 求解不同尺度空间单元下的无重复贪心选址
    # solving the cover problem in maup
    for scale in range(2, maxs + 1):
        print('scale', scale)
        file = r'{dirs}\cover_scale{scale}.csv'.format(dirs=dirs, scale=scale)
        cover = read_cover(file)
        title = r'{dirs}\loc_scale{scale}.npy'.format(dirs=dirs, scale=scale)
        loc, nums = solve_greed_nodup(cover, k, title)
    return


if __name__ == '__main__':
    xmin, xmax, ymin, ymax = 113.67561783007596, 114.60880792079337, \
        22.28129833936937, 22.852485545898546  # 深圳最大最小经纬度
    r = 6371 * 1000
    ymid = (ymin + ymax) / 2
    r1 = r * cos(ymid / 180 * pi)
    scale = 150
    xgap = scale / r1 / pi * 180
    ygap = scale / r / pi * 180
    xnum = int((xmax - xmin) / xgap) + 1
    ynum = int((ymax - ymin) / ygap) + 1
    print(ynum)
    k = 136

    coverfile = r'E:\基础数据\轨迹\深圳百度20191202\stay_150m_t1800_cover.csv'
    coverfile=r'E:\基础数据\轨迹\深圳百度20191202\stay_150m_home_t1800_cover.csv'
    loctitle = r'E:\门店选址\20250427基于home\nodup_nodub_stay.npy'
    curvetitle = r'E:\门店选址\20250427基于home\curves_stay.png'
    curvedatatitle = r'E:\门店选址\20250427基于home\curves_stay.npy'

    maxs = 10
    dirs = r'E:\门店选址\20250427基于home\maup'

    layergap = 100

    nrow = 3
    ncol = 3

    store = get_store(xgap, ygap, ynum)

    cover = read_cover(coverfile, minpop=0)

    #loc, nums = solve_greed_nodup(cover, k, loctitle)

    loc = np.load(loctitle).tolist()

    plot_curve_dup_nodup(store, cover, loc, curvetitle, curvedatatitle)

    #fit_power_curve(curvedatatitle, k)


    # maup_data(coverfile, maxs, dirs, ynum)
    # maup_solve(dirs, maxs, k)


