import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from matplotlib.widgets import Slider, Button

# Расчет изменения давления
def p_r_t(rx, ry ,t):
    return - 1 / 2 * sc.expi(- (np.sqrt(rx * rx + ry * ry)) ** 2 / 4 / t)
# Функция изменения точек для расчета при изменении положения слайдера
def update_depth(val):
    t = int(round(slider_depth.val))
    i = 0
    xwells_prod = [10, 10, 90, 90]
    ywells_prod = [10, 90, 90, 10]
    xwells_inj = [50]
    ywells_inj = [50]
    p_mesh_i_prod=[]
    for xi in xwells_prod:
        p_mesh_i_ = p_r_t(rd_from_r(xv-xi), rd_from_r(yv-ywells_prod[i]), td_from_t(t))
        i = i + 1
        p_mesh_i_prod.append(p_mesh_i_)
    p_mesh_i_inj=[]
    k = 0
    for xi in xwells_inj:
        p_mesh_i_ = - p_r_t(rd_from_r(xv-xi), rd_from_r(yv-ywells_inj[k]), td_from_t(t))
        k = k + 1
        p_mesh_i_inj.append(p_mesh_i_)
    p_mesh_sum = 0
    for p_mesh_i_ in p_mesh_i_prod:
        p_mesh_sum = p_mesh_i_ + p_mesh_sum
    p_mesh_sum_inj = 0
    for p_mesh_i_ in p_mesh_i_inj:
        p_mesh_sum_inj = p_mesh_i_ + p_mesh_sum_inj
        p_mesh_sum_inj = p_from_pd_atma(p_mesh_sum_inj, q_sm3day=60)
    p_mesh_sum = p_from_pd_atma(p_mesh_sum) + p_mesh_sum_inj - 250
    im_h.set_data(p_mesh_sum)
    print(p_mesh_sum)

# функции безразмерных координат:
def r_from_rd_m(rd, rw_m=0.1):
    return rd*rw_m

def rd_from_r(r_m, rw_m=0.1):
    return r_m/rw_m

def t_from_td_hr(td, k_mD=10, phi=0.2, mu_cP=1, ct_1atm=1e-5, rw_m=0.1):
    return td * phi * mu_cP * ct_1atm * rw_m * rw_m / k_mD / 0.00036

def td_from_t(t_hr, k_mD=10, phi=0.2, mu_cP=1, ct_1atm=1e-5, rw_m=0.1):
    return 0.00036 * t_hr * k_mD / (phi * mu_cP * ct_1atm * rw_m * rw_m)

def p_from_pd_atma(pd, k_mD=10, h_m=10, q_sm3day=20, b_m3m3=1.2, mu_cP=1, pi_atma=250):
    return pi_atma - pd * 18.41 * q_sm3day * b_m3m3 * mu_cP / k_mD / h_m

def pd_from_p(p_atma, k_mD=10, h_m=10, q_sm3day=20, b_m3m3=1.2, mu_cP=1, pi_atma=250):
    return (pi_atma - p_atma) / (18.41 * q_sm3day * b_m3m3 * mu_cP) * k_mD * h_m


# Координаты скважин
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
xv, yv = np.meshgrid(x, y)

xwells_prod = [10, 10, 90, 90]
ywells_prod = [10, 90, 90, 10]
xwells_inj = [50]
ywells_inj = [50]
t = 1

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15)

# Тепловая карта
p_mesh_i_prod=[]
i = 0
# рассчитаем значение давлений во всех точках сетки для скважины i
for xi in xwells_prod:
    p_mesh_i_ = p_r_t(rd_from_r(xv-xi), rd_from_r(yv-ywells_prod[i]), td_from_t(t))
    i = i + 1
    p_mesh_i_prod.append(p_mesh_i_)
p_mesh_i_inj=[]
k = 0
for xi in xwells_inj:
    p_mesh_i_ = - p_r_t(rd_from_r(xv-xi), rd_from_r(yv-ywells_inj[k]), td_from_t(t))
    k = k + 1
    p_mesh_i_inj.append(p_mesh_i_)
#Найдем сумму депрессий/репрессий
p_mesh_sum = 0
p_mesh_sum_inj = 0
for p_mesh_i_ in p_mesh_i_prod:
    p_mesh_sum = p_mesh_i_ + p_mesh_sum
for p_mesh_i_ in p_mesh_i_inj:
    p_mesh_sum_inj = p_mesh_i_ + p_mesh_sum_inj
p_mesh_sum_inj = p_from_pd_atma(p_mesh_sum_inj, q_sm3day=70)
p_mesh_sum = p_from_pd_atma(p_mesh_sum) + p_mesh_sum_inj - 250
im_h = ax.imshow(p_mesh_sum, cmap='rainbow', interpolation='nearest')
ax.invert_yaxis()

# Тепловая шкала
fig.colorbar(im_h, ax = ax)

# Слайдер
ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'Время', 1, 1000, valinit=t)

# Вызов функции изменения положения слайдера
slider_depth.on_changed(update_depth)
print(p_mesh_sum)
plt.show()