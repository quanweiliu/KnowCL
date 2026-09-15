import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        """
        origin data
        """
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth.astype(np.int64)

    @property
    def normal_cube(self):
        """
        normalization data: range(0, 1)
        """
        return (self.data_cube-np.min(self.data_cube)) / (np.max(self.data_cube)-np.min(self.data_cube))

class PaviaURaw(DataReader):
    def __init__(self, path_data=None, type_data=None):
        super(PaviaURaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "PaviaU.mat")
        self.data_cube = raw_data_package["paviaU"].astype(np.float32)
        
        if type_data == None:
            truth = sio.loadmat(path_data + "PaviaU_gt.mat")
            self.g_truth = truth["groundT"]

        elif type_data == "TRLabel":
            truth = sio.loadmat(path_data + "PaviaU_TR.mat")
            self.g_truth = truth["PaviaU_GT_TR"]

        elif type_data == "TSLabel":
            truth = sio.loadmat(path_data + "PaviaU_TE.mat")
            self.g_truth = truth["PaviaU_GT_TE"]
            # train_mat_num = Counter(self.g_truth.flatten())
            # print(train_mat_num)

class PaviaCRaw(DataReader):
    def __init__(self, path_data=None):
        super(PaviaCRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "PaviaC.mat")
        self.data_cube = raw_data_package["pavia"].astype(np.float32)
        
        # raw_data_package = sio.loadmat(path_data + "paviaU.mat")
        # self.data_cube = raw_data_package["data"].astype(np.float32)

        truth = sio.loadmat(path_data + "PaviaC_gt.mat")
        self.g_truth = truth["pavia_gt"]

class IndianRaw(DataReader):
    def __init__(self, path_data=None):
        super(IndianRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "Indian_pines_corrected.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(path_data + "Indian_pines_gt.mat")
        self.g_truth = truth["groundT"]


class KSCRaw(DataReader):
    def __init__(self, path_data=None):
        super(KSCRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "KSC.mat")
        self.data_cube = raw_data_package["KSC"].astype(np.float32)
        truth = sio.loadmat(path_data + "KSC_gt.mat")
        self.g_truth = truth["KSC_gt"]


class SalinasRaw(DataReader):
    def __init__(self, path_data=None, type_data=None):
        super(SalinasRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "Salinas_corrected.mat")
        self.data_cube = raw_data_package["salinas_corrected"].astype(np.float32)
        
        if type_data == None:
            truth = sio.loadmat(path_data + "Salinas_gt.mat")
            self.g_truth = truth["salinas_gt"]

        elif type_data == "TRLabel":
            truth = sio.loadmat(path_data + "Salinas_TR.mat")
            self.g_truth = truth["Salinas_GT_TR"]

        elif type_data == "TSLabel":
            truth = sio.loadmat(path_data + "Salinas_TE.mat")
            self.g_truth = truth["Salinas_GT_TE"]


class Houston_2013Raw(DataReader):
    def __init__(self, path_data=None, type_data=None):
        super(Houston_2013Raw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "Houston_2013.mat")
        self.data_cube = raw_data_package["Houston"].astype(np.float32)
        if type_data == "Houston" or type_data == None:
            truth = sio.loadmat(path_data + "Houston_GT_2013.mat")
            self.g_truth = truth["Houston_GT"]
        elif type_data == "TRLabel":
            truth = sio.loadmat(path_data + "HoustonTRLabel2013.mat")
            self.g_truth = truth["TRLabel"]
        elif type_data == "TSLabel":
            truth = sio.loadmat(path_data + "HoustonTSLabel2013.mat")
            self.g_truth = truth["TSLabel"]
        else:
            raise KeyError("Please select among ['Houston', 'TRLabel', 'TSLabel']")

class Houston_2018Raw(DataReader):
    def __init__(self, path_data=None, type_data=None):
        super(Houston_2018Raw, self).__init__()
        # print(type_data)

        raw_data_package = sio.loadmat(path_data + "Houston2018.mat")
        self.data_cube = raw_data_package["hu2018_48"].astype(np.float32)
        if type_data == "Houston" or type_data == None:
            truth = sio.loadmat(path_data + "hu2018_gt.mat")
            self.g_truth = truth["hu2018_gt"]
        elif type_data == "TRLabel":
            truth = sio.loadmat(path_data + "HoustonTRLabel2018.mat")
            self.g_truth = truth["TRLabel"]
        elif type_data == "TSLabel":
            truth = sio.loadmat(path_data + "HoustonTSLabel2018.mat")
            self.g_truth = truth["TSLabel"]
        else:
            raise KeyError("Please select among ['Houston', 'TRLabel', 'TSLabel']")

class BotswanaRaw(DataReader):
    def __init__(self, path_data=None):
        super(BotswanaRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "Botswana.mat")
        self.data_cube = raw_data_package["Botswana"].astype(np.float32)
        truth = sio.loadmat(path_data + "Botswana_gt.mat")
        self.g_truth = truth["Botswana_gt"]


class DCRaw(DataReader):
    def __init__(self, path_data=None):
        super(DCRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "DC.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(path_data + "DC_gt2.mat")
        self.g_truth = truth['groundT']


class DioniRaw(DataReader):
    def __init__(self, path_data=None, type_data=None):
        super(DioniRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Dioni.mat")
        self.data_cube = raw_data_package["Dioni"].astype(np.float32)

        if type_data == None:
            truth = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Dioni_GT.mat")
            self.g_truth = truth["Dioni_GT"]

        elif type_data == "TRLabel":
            truth = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Dioni_TR.mat")
            self.g_truth = truth["Dioni_GT_TR"]

        elif type_data == "TSLabel":
            truth = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Dioni_TE.mat")
            self.g_truth = truth["Dioni_GT_TE"]


class LoukiaRaw(DataReader):
    def __init__(self, path_data=None):
        super(LoukiaRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Loukia.mat")
        self.data_cube = raw_data_package["Loukia"].astype(np.float32)
        truth = sio.loadmat(path_data + "HyRANK_satellite/TrainingSet/Loukia_GT.mat")
        self.g_truth = truth['Loukia_GT']


class LongKouRaw(DataReader):
    def __init__(self, path_data=None):
        super(LongKouRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou.mat")
        self.data_cube = raw_data_package["WHU_Hi_LongKou"].astype(np.float32)
        truth = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat")
        self.g_truth = truth["WHU_Hi_LongKou_gt"]


class HongHuRaw(DataReader):
    def __init__(self, path_data=None):
        super(HongHuRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-HongHu/WHU_Hi_HongHu.mat")
        self.data_cube = raw_data_package["WHU_Hi_HongHu"].astype(np.float32)
        truth = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat")
        self.g_truth = truth["WHU_Hi_HongHu_gt"]

class HongHu_subRaw(DataReader):
    def __init__(self, path_data=None):
        super(HongHu_subRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "WHU-Hi\WHU-Hi-HongHu\honghu_sub.mat")
        self.data_cube = raw_data_package["honghu_sub"].astype(np.float32)
        truth = sio.loadmat(path_data + "WHU-Hi\WHU-Hi-HongHu\honghu_sub_gt.mat")
        self.g_truth = truth["honghu_sub_gt"]


class HanChuanRaw(DataReader):
    def __init__(self, path_data=None):
        super(HanChuanRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat")
        self.data_cube = raw_data_package["WHU_Hi_HanChuan"].astype(np.float32)
        truth = sio.loadmat(path_data + "WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat")
        self.g_truth = truth["WHU_Hi_HanChuan_gt"]

class CuonadongRaw(DataReader):
    def __init__(self, path_data=None):
        super(CuonadongRaw, self).__init__()

        raw_data_package = sio.loadmat(path_data + "Cuo\cuonadong_corrected.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(path_data + "Cuo\cuonadong_gt.mat")
        self.g_truth = truth["groundT"]

# Load the dataset
def load_data(dataset = "IndianPines", path_data=None, type_data=None):
    if dataset == "IndianPines":
        data = IndianRaw(path_data).normal_cube
        data_gt = IndianRaw(path_data).truth

    elif dataset == "PaviaU":
        data = PaviaURaw(path_data, type_data).normal_cube
        data_gt = PaviaURaw(path_data, type_data).truth

    elif dataset == "PaviaC":
        data = PaviaCRaw(path_data).normal_cube
        data_gt = PaviaCRaw(path_data).truth

    elif dataset == "KSC":
        data = KSCRaw(path_data).normal_cube
        data_gt = KSCRaw(path_data).truth

    elif dataset == "Salinas":
        data = SalinasRaw(path_data, type_data).normal_cube
        data_gt = SalinasRaw(path_data, type_data).truth

    elif dataset == "Houston_2013":
        data = Houston_2013Raw(path_data, type_data).normal_cube
        data_gt = Houston_2013Raw(path_data, type_data).truth

    elif dataset == "Houston_2018":
        data = Houston_2018Raw(path_data, type_data).normal_cube
        data_gt = Houston_2018Raw(path_data, type_data).truth

    elif dataset == "Botswana":
        data = BotswanaRaw(path_data).normal_cube
        data_gt = BotswanaRaw(path_data).truth

    elif dataset == "DC":
        data = DCRaw(path_data).normal_cube
        data_gt = DCRaw(path_data).truth

    elif dataset == "Dioni":
        data = DioniRaw(path_data, type_data).normal_cube
        data_gt = DioniRaw(path_data, type_data).truth
        
    elif dataset == "Loukia":
        data = LoukiaRaw(path_data).normal_cube
        data_gt = LoukiaRaw(path_data).truth

    elif dataset == "LongKou":
        data =LongKouRaw(path_data).normal_cube
        data_gt = LongKouRaw(path_data).truth

    elif dataset == "HongHu":
        data = HongHuRaw(path_data).normal_cube
        data_gt = HongHuRaw(path_data).truth

    elif dataset == "HongHu_sub":
        data = HongHu_subRaw(path_data).normal_cube
        data_gt = HongHu_subRaw(path_data).truth

    elif dataset == "HanChuan":
        data =HanChuanRaw(path_data).normal_cube
        data_gt = HanChuanRaw(path_data).truth

    elif dataset == "Cuonadong":
        data =CuonadongRaw(path_data).normal_cube
        data_gt = CuonadongRaw(path_data).truth
    
    else: 
        raise ValueError("IndianPines", 
                        "PaviaU", 
                        "PaviaC"
                        "KSC",
                        "Salinas",
                        "Houston_2013",
                        "Houston_2018",
                        "Botswana",
                        "DC",
                        "Dioni",
                        "Loukia",
                        "LongKou",
                        "HongHu",
                        "HongHu_sub",
                        "HanChuan",
                        "Cuonadong")
    return data, data_gt



# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca

# same to split_data
def data_info(train_label=None, val_label=None, test_label=None, start=1):
    class_num = np.max(train_label)

    if train_label is not None and val_label is not None and test_label is not None:
        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i],"\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)
    
    elif train_label is not None and val_label is not None:
        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)
    
    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)
        
    else:
        raise ValueError("labels are None")

def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 400, save_img=None):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)

if __name__ == "__main__":
    data = IndianRaw().cube
    data_gt = IndianRaw().truth
    IndianRaw().data_info(data_gt)
    IndianRaw().draw(data_gt, save_img=None)
    print(data.shape)
    print(data_gt.shape)





















