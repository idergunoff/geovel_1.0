import struct

fi = 'Ю22 15С П1 5-6.geo'


def read_signal_geo(file_name, n_izm):
    with open(file_name, 'rb') as file:
        radarogram = file.read()[15:]
        signal = radarogram[0 + n_izm * 526:526 + n_izm * 526]
        bin_format = '7h512b'   # 7 двухбитных чисел и 512 однобитных чисел
        list_signal = list(struct.unpack(bin_format, signal))[7:]
        return list_signal


def calc_len_izm_geo(file_name):
    with open(file_name, 'rb') as f:
        a = f.read()[16:]
        count_izm = int(len(a) / 526)
        return count_izm


for i in range(calc_len_izm_geo(fi)):
    print(len(read_signal_geo(fi, i)))