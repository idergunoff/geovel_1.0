from math import sin, cos, tan, pi
from pygeoguz.transform import bl2xy
from pygeoguz.objects import PointBL


xi, yi = 55.039575, 51.3209101

ro = 206264.8062 # Число угловых секунд в радиане

# Эллипсоид Красовского
aP = 6378245 # Большая полуось
alP = 1 / 298.3 # Сжатие
e2P = 2 * alP - alP ** 2 # Квадрат эксцентриситета

# Эллипсоид WGS84 (GRS80, эти два эллипсоида сходны по большинству параметров)
aW = 6378137 # Большая полуось
alW = 1 / 298.257223563 # Сжатие
e2W = 2 * alW - alW ** 2 # Квадрат эксцентриситета

# Вспомогательные значения для преобразования эллипсоидов
a = (aP + aW) / 2
e2 = (e2P + e2W) / 2
da = aW - aP
de2 = e2W - e2P

# Линейные элементы трансформирования, в метрах
# (коэф-ты из ГОСТ скорректированы до получения значений из геокалькулятора)
dx = 23.57
dy = -140.95
dz = -79.80

# Угловые элементы трансформирования, в секундах
# (коэф-ты из ГОСТ скорректированы до получения значений из геокалькулятора)
wx = 0
wy = -0.4012
wz = -0.8565

# Дифференциальное различие масштабов
ms = -0.22 * 1e-6

def dB(Bd, Ld, H):
    # Dim B, L, M, N
    B = Bd * pi / 180
    L = Ld * pi / 180
    M = a * (1 - e2) / (1 - e2 * sin(B) ** 2) ** 1.5
    N = a * (1 - e2 * sin(B) ** 2) ** -0.5
    dB_val = ro / (M + H) * (N / a * e2 * sin(B) * cos(B) * da + (N ** 2 / a ** 2 + 1) * N * sin(B) * cos(B) * de2 / 2 - (dx * cos(L) + dy * sin(L)) * sin(B) + dz * cos(B)) - wx * sin(L) * (1 + e2 * cos(2 * B)) + wy * cos(L) * (1 + e2 * cos(2 * B)) - ro * ms * e2 * sin(B) * cos(B)
    return dB_val


def dL(Bd, Ld, H):
    # Dim B, L, N
    B = Bd * pi / 180
    L = Ld * pi / 180
    N = a * (1 - e2 * sin(B) ** 2) ** -0.5
    dL_val = ro / ((N + H) * cos(B)) * (-dx * sin(L) + dy * cos(L)) + tan(B) * (1 - e2) * (wx * cos(L) + wy * sin(L)) - wz
    return dL_val


def WGS84_SK42_Lat(Bd, Ld, H):
    WGS84_SK42_Lat_val = Bd - dB(Bd, Ld, H) / 3600
    return WGS84_SK42_Lat_val


def WGS84_SK42_Long(Bd, Ld, H):
    WGS84_SK42_Long_val = Ld - dL(Bd, Ld, H) / 3600
    return WGS84_SK42_Long_val


def wgs84_to_pulc42(xi, yi):
    """
    перевод координат из географических WGS84 в метрические Pulcovo(CK42)
    """
    dLat = WGS84_SK42_Lat(xi, yi, 0)
    dLon = WGS84_SK42_Long(xi, yi, 0)
    point = PointBL(dLat, dLon)
    point_m_pulc = bl2xy(point)
    return point_m_pulc.x, point_m_pulc.y


# print(wgs84_to_pulc42(55.039575, 51.3209101))
#
# print(wgs84_to_pulc42(80.039575, 51.3209101))
