from matplotlib import pyplot as plt
import pickle as pck
from scipy import interpolate
import numpy as np
from matplotlib import rc

plt.style.use('seaborn')
# plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text', usetex=True)

def get_points(filename):
    infile = open(filename,'rb')
    x, y = pck.load(infile)
    # return x,y
    # print(y[-1])
    x_new = np.linspace(0,1000, 100)
    intfunc = interpolate.interp1d(x,y,fill_value='extrapolate', kind='nearest')
    y_interp = intfunc(x_new)
    infile.close()
    return x_new, y_interp


def plotnorm():
    USABLE_WIDTH_mm = 100
    USABLE_HEIGHT_mm = 60
    YANK_RATIO = 0.0393701
    USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
    USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
    SUBPLOT_FONT_SIZE = 10
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
    x,y = get_points('losses.pkl')
    axes.plot(x,y)
    x,y = get_points('lossesrandom.pkl')
    axes.plot(x,y)
    axes.set_ylabel('Mean Error',fontsize=SUBPLOT_FONT_SIZE)
    # axes[1].set_ylabel(r'Average $\Delta w$ per epoch',fontsize=SUBPLOT_FONT_SIZE)
    axes.set_xticklabels([])
    axes.set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
    axes.set_xlim(xmin=0)
    # axes[1].set_xlim(xmin=0)
    # axes.set_yscale('log')
    axes.legend(['Learned feedback weight', 'Random feedback weight'])
    fig.savefig('Results.pdf')
    plt.show()

def plotinfo():
    USABLE_WIDTH_mm = 100
    USABLE_HEIGHT_mm = 120
    YANK_RATIO = 0.0393701
    USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
    USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
    SUBPLOT_FONT_SIZE = 10
    fig, axes = plt.subplots(nrows=2, ncols=1, sharey='row', figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
    x,y = get_points('losses.pkl')
    axes[0].plot(x,y)
    x,y = get_points('deltas.pkl')
    axes[1].plot(x,y)
    axes[0].set_ylabel('Mean Error',fontsize=SUBPLOT_FONT_SIZE)
    axes[1].set_ylabel(r'Average $\Delta w$ per epoch',fontsize=SUBPLOT_FONT_SIZE)
    axes[0].set_xticklabels([])
    axes[1].set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
    axes[0].set_xlim(xmin=0)
    axes[1].set_xlim(xmin=0)
    fig.savefig('Results.pdf')
    plt.show()
    
def get_data(data):
    # infile = open(filename,'rb')
    # x = np.linspace(0,10000,)
    y = data
    # if mode == 'entropy':
    #     return x,y
    # else:
    #     return x, z
    # print(y[-1])
    x = [100*a for a in range(int(100))]
    x_new = np.linspace(0,10000, 100)
    intfunc = interpolate.interp1d(x,y,fill_value='extrapolate', kind='nearest')
    y_interp = intfunc(x_new)
    # infile.close()
    return x_new, y_interp

def plotinfo2():
    USABLE_WIDTH_mm = 200
    USABLE_HEIGHT_mm = 120
    YANK_RATIO = 0.0393701
    USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
    USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
    SUBPLOT_FONT_SIZE = 10
    infile = open('infos2.pkl','rb')
    HYXs,IYXs,HZXs,IZXs = pck.load(infile)
    infile.close()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
    for i in range(3):
        x,y = get_data(np.mean(HYXs[i], axis=0))
        axes[0][0].plot(x,y)
        x,y = get_data(np.mean(IYXs[i], axis=0))
        axes[0][1].plot(x,y)
        x,y = get_data(np.mean(HZXs[i], axis=0))
        axes[1][0].plot(x,y)
        x,y = get_data(np.mean(IZXs[i], axis=0))
        axes[1][1].plot(x,y)
    axes[0][0].set_ylabel('H(Y|X)',fontsize=SUBPLOT_FONT_SIZE)
    axes[0][0].set_title('a)', loc='left')
    axes[0][1].set_ylabel('I(X;Y)',fontsize=SUBPLOT_FONT_SIZE)
    axes[0][1].set_title('b)', loc='left')
    axes[1][0].set_ylabel('H(Z|X)',fontsize=SUBPLOT_FONT_SIZE)
    axes[1][0].set_title('c)', loc='left')
    axes[1][1].set_ylabel('I(X;Z)',fontsize=SUBPLOT_FONT_SIZE)
    axes[1][1].set_title('d)', loc='left')
    # axes[0].set_xticklabels([])
    for i in range(2):
        for j in range(2):
            axes[i][j].set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
            axes[i][j].set_xlim(xmin=0)
    axes[0][0].legend(['16','32','64'], loc = 'upper center', ncol=3)
    fig.savefig('Results.pdf')
    # plt.show()

# plotinfo2()
plotnorm()