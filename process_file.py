import os, glob
import cv2
import numpy as np
import random
import shutil
import random

THREAD_TYPE_RB = ['SSP', 'SS', 'SBL', 'SBLT', 'SWN', 'SPWN', 'TE', 'BLT', 'SSPWN', 'STE']
THREAD_TYPE_B = ['NISIII', 'DSIII', 'DI', 'DSIMPLE', 'IMPLANTIUM', 'IMPANTIUM']
THREAD_TYPE_V = ['OTSIII', 'OSS', 'DNRL', 'DNR', 'SHLN', 'OSEV', 'OGSIII']

DENT_TYPE_MX_M = ['14','15','16','17','18','24','25','26','27','28']
DENT_TYPE_MN_M = ['44','45','46','47','48','34','35','36','37','38']
DENT_TYPE_MX_I = ['11','12','13','21','22','23']
DENT_TYPE_MN_I = ['31','32','33','41','42','43']


# dataset/train_splited/ 에 10-fold를 위한 폴더생성
# 0 - Buttress, Reverse_buttress, v-shaped 하위에 B, RB, V
# 1 - Buttress, Reverse_buttress, v-shaped
# ... 총 10x3 폴더생성
def create_kfold_directory():
    org_path = 'dataset/train_splited/kfold'
    
    for i in range(10):
        os.makedirs(os.path.join(org_path, str(i)))
        os.makedirs(os.path.join(org_path, str(i), 'train'))
        os.makedirs(os.path.join(org_path, str(i), 'test'))
        os.makedirs(os.path.join(org_path, str(i), 'train', 'Buttress'))
        os.makedirs(os.path.join(org_path, str(i), 'train', 'Reverse_buttress'))
        os.makedirs(os.path.join(org_path, str(i), 'train', 'v-shaped'))
        os.makedirs(os.path.join(org_path, str(i), 'test', 'Buttress'))
        os.makedirs(os.path.join(org_path, str(i), 'test', 'Reverse_buttress'))
        os.makedirs(os.path.join(org_path, str(i), 'test', 'v-shaped'))
        
# 1000개의 사진을 800개, 200개로 나눔
def train_val_split():
    org_path = 'dataset/raw'
    lst_file = os.listdir(org_path)
    
    val_file = random.sample(lst_file, 200)
    train_file = set(lst_file) - set(val_file)
    
    print(len(val_file))
    print(len(train_file))
    
    for file in val_file:
        shutil.copy(os.path.join(org_path, file), os.path.join('dataset/val_splited'))
        
    for file in train_file:
        shutil.copy(os.path.join(org_path, file), os.path.join('dataset/train_splited/all'))
        
def all_2_kfold():
    org_path = 'dataset/train_splited/all'
    lst_file = os.listdir(org_path)
    
    random.shuffle(lst_file)
    
    for i in range(10):
        lst_test = lst_file[i*80 : (i+1)*80]
        lst_train = set(lst_file) - set(lst_test)
        
        for img_name in lst_test:
            save_path = os.path.join('dataset/train_splited/kfold', str(i), 'test')
            img_cls = img_name.split(',')[-1].split('.')[0]
            
            if img_cls.upper() in THREAD_TYPE_B:
                path = os.path.join(save_path, "Buttress")
            elif img_cls.upper() in THREAD_TYPE_RB:
                path = os.path.join(save_path, "Reverse_buttress")
            elif img_cls.upper() in THREAD_TYPE_V:
                path = os.path.join(save_path, "v-shaped")
            else:
                print(img_name , "에 대한 class가 없음!")
                continue
                
            shutil.copy(os.path.join(org_path, img_name), path)
            
        for img_name in lst_train:
            save_path = os.path.join('dataset/train_splited/kfold', str(i), 'train')
            img_cls = img_name.split(',')[-1].split('.')[0]
            
            if img_cls.upper() in THREAD_TYPE_B:
                path = os.path.join(save_path, "Buttress")
            elif img_cls.upper() in THREAD_TYPE_RB:
                path = os.path.join(save_path, "Reverse_buttress")
            elif img_cls.upper() in THREAD_TYPE_V:
                path = os.path.join(save_path, "v-shaped")
            else:
                print(img_name , "에 대한 class가 없음!")
                continue
                
            shutil.copy(os.path.join(org_path, img_name), path)

def create_model_kfold_folder():
    for i in range(10):
        os.makedirs(os.path.join('model', str(i)))
            
            
            
            
            
            
            
            
            
            
            
            
def classify_file_to_thread_type(org_path, save_path):
    for img_path in glob.iglob(os.path.join(org_path, '**', "*.jpg"), recursive=True):

        # 파일 이름을 구한다
        img_name = os.path.basename(img_path)

        # 임플란트 종류를 얻음
        img_cls = img_name.split(",")[-1].split(".")[0]

        path = ""
        if img_cls.upper() in THREAD_TYPE_B:
            path = os.path.join(save_path, "Buttress")
        elif img_cls.upper() in THREAD_TYPE_RB:
            path = os.path.join(save_path, "Reverse_buttress")
        elif img_cls.upper() in THREAD_TYPE_V:
            path = os.path.join(save_path, "v-shaped")
        else:
            print(img_name , "에 대한 class가 없음!")
            continue

        shutil.copy(img_path, path)
        print(img_name , " 작업 완료")

def train_test_val_split(org_path, save_path, test_portion, val_portion):
    list_cls_path = glob.glob(os.path.join(org_path, "*"))

    # 먼저 save_path에 train, test, val 폴더와 각 폴더에 해당하는 cls를 만듦
    list_part = ['train', 'test', 'val']

    for part_name in list_part:
        create_path(os.path.join(save_path, part_name))

        for cls_path in list_cls_path:
            cls_name = cls_path.split('\\')[-1]
            create_path(os.path.join(save_path, part_name, cls_name))

    # 폴더는 다 만들었으니 이제 옮기기만 하면 됨
    for cls_path in list_cls_path:
        cls_name = cls_path.split('\\')[-1]
        list_img_path = glob.glob(os.path.join(cls_path, "*.jpg"))

        img_len = len(list_img_path)
        rnd_index = [x for x in range(0, img_len)]

        test_count = int(img_len * test_portion)
        val_count = int(img_len * val_portion)
        train_count = img_len - test_count - val_count

        train_index = random.sample(rnd_index, train_count)
        rnd_index = list(set(rnd_index) - set(train_index))

        test_index = random.sample(rnd_index, test_count)
        rnd_index = list(set(rnd_index) - set(test_index))

        val_index = rnd_index

        print("\n\n\nTrain : {}, Test : {}, Validation : {}".format(len(train_index), len(test_index), len(val_index)))
        print("Train : {}, Test : {}, Validation : {} (Count)".format(train_count, test_count, val_count))

        for i, img_path in enumerate(list_img_path):

            file_name = os.path.basename(img_path)
            real_save_path = ''

            if i in train_index:
                real_save_path = os.path.join(save_path, 'train', cls_name, file_name)
            elif i in test_index:
                real_save_path = os.path.join(save_path, 'test', cls_name, file_name)
            elif i in val_index:
                real_save_path = os.path.join(save_path, 'val', cls_name, file_name)
            else:
                pass

            shutil.copy(img_path, real_save_path)
            print(img_path, " ----> ", real_save_path)


def create_path(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)


def Raw2Processed(org_path, save_path):
    list_img_path = glob.glob(os.path.join(org_path, '*.jpg'))
    print("총 이미지 갯수 : ", len(list_img_path))

    t = 1
    for img_path in list_img_path:

        img_name = os.path.basename(img_path)

        # 이미지 형식 : 422188,15,DI.jgp
        # 여기서 DI만 뽑아오기
        pt_num = img_name.split(",")[0]
        dental_num = img_name.split(",")[1]
        implant_name = img_name.split(",")[2].split(".")[0]

        print("\n\n", t, "\nFile Name : " , img_name, "\nImplant Class : " , implant_name)
        print("Image Path : ", img_path)

        if(implant_name.lower() == "implantium"):
            implant_name = "DI"

        # 이미지 형식을 1234567,12,True,DI.jpg로 바꾼다
        real_name = "{},{},{},{}.jpg".format(pt_num, dental_num, "None", implant_name)
        print("Real name : ", real_name)

        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if(im is None):
            print("Image Read Failed!")
            t += 1
            continue

        real_save_path = os.path.join(save_path, real_name)

        print("Save path : ", real_save_path)

        cv2.imwrite(real_save_path, im)

        print("Saved!")

        t += 1

def make_dic_implant_name(org_path):
    list_img_path = glob.glob(os.path.join(org_path, '*.jpg'))
    print("총 이미지 갯수 : ", len(list_img_path))

    dic = {} # 임플란트 종류별로
    dic_thread = {'B':0, 'RB':0, 'V':0} # 임플란트 thread type 별로
    dic_dent = {'MX_I':0, 'MX_M':0, 'MN_I':0, 'MN_M':0} # 임플란트 치식 별로

    for img_path in list_img_path:
        img_name = os.path.basename(img_path)
        implant_name = img_name.split(",")[-1].split(".")[0].upper()
        implant_num = img_name.split(",")[1].split(".")[0]

        if implant_name in dic.keys():
            dic[implant_name] += 1
        else:
            dic[implant_name] = 1

        if implant_name in THREAD_TYPE_B:
            dic_thread['B'] += 1
        elif implant_name in THREAD_TYPE_RB:
            dic_thread['RB'] += 1
        elif implant_name in THREAD_TYPE_V:
            dic_thread['V'] += 1
        else:
            print(implant_name)

        if implant_num in DENT_TYPE_MX_I:
            dic_dent['MX_I'] += 1
        elif implant_num in DENT_TYPE_MX_M:
            dic_dent['MX_M'] += 1
        elif implant_num in DENT_TYPE_MN_I:
            dic_dent['MN_I'] += 1
        elif implant_num in DENT_TYPE_MN_M:
            dic_dent['MN_M'] += 1
        else:
            print('cannot classify : {}'.format(implant_num))

    print(dic)
    print(dic_thread)
    print(dic_dent)

if __name__ == '__main__':
    create_model_kfold_folder()