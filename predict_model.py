from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import argparse
import imutils
import pickle
import cv2
import os
import glob
import math

def test_network():

    IMAGE_PATH = ['new_model/classified_thread/val/Buttress', 'new_model/classified_thread/val/Reverse buttress', 'new_model/classified_thread/val/v-shaped']
    MODEL_PATH = 'check_VGG_weight_thread.hdf5'

    IMPLANT_LABEL = ['Buttress', 'Reverse buttress', 'v-shaped']

    print("[INFO] Loading network...")
    model = load_model(MODEL_PATH)

    c_right = 0
    c_wrong = 0

    for path in IMAGE_PATH:
        saved_img = np.zeros((224,224), dtype='uint8')
        for list_img in glob.glob(os.path.join(path, '*.jpg')):
            output = cv2.imread(list_img, cv2.IMREAD_GRAYSCALE)

            # pre-process the image for classification
            image = cv2.resize(output, dsize=(224, 224))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # classify the input image then find the indexes of the two class labels with the largest probability
            probability = model.predict(image)[0]

            # putting the text on the image
            predicted = IMPLANT_LABEL[int(np.argmax(probability))]
            prob = str(probability[np.argmax(probability)])[:4]
            ans = path.split('/')[-1]

            shown = cv2.resize(output, (224, 224))
            shown = cv2.putText(shown,
                                '%s %s' % (predicted, prob),
                                (10,20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255,0,0),
                                2)

            saved_img = cv2.hconcat([saved_img, shown])

            #cv2.imshow('predict', shown)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

            print('{} {}, {}'.format(predicted, prob, ans))

            if predicted == ans:
                c_right += 1
            else:
                c_wrong += 1

    cv2.imwrite('saved.jpg', saved_img)
    print('total : {}\nright : {}\nwrong : {}'.format(c_right+c_wrong, c_right, c_wrong))
    print('percent : {}%'.format(c_right / (c_right+c_wrong) * 100))

def test_model_generator(p_model):
    MODEL_PATH = r'new_model\checkpoint\checkpoint_{}.hdf5'.format(p_model)
    IMAGE_PATH = r'new_model\test_val'
    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle="false")

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    print(filenames)
    print('filenum : {}'.format(nb_samples))

    model = load_model(MODEL_PATH)
    loss, acc = model.evaluate_generator(test_generator)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    print('--{}--'.format(p_model))
    print('\n\n--loss--')
    print(loss)

    print('\n\n--acc--')
    print(acc)

# predict 한뒤 numpy 저장
def test_model_with_dental(p_model):
    MODEL_PATH = r'new_model\checkpoint\checkpoint_{}.hdf5'.format(p_model)
    IMAGE_PATH = r'new_model\test_val' # test + val 다 합친 경로
    PICKLE_PATH = r'new_model\prediction\predict_{}.pkl'

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle=False)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    print(filenames)
    print('filenum : {}'.format(nb_samples))
    print(test_generator.class_indices)

    model = load_model(MODEL_PATH)
    predict = model.predict_generator(test_generator)

    print(predict.shape)
    print(predict)

    with open(PICKLE_PATH.format(p_model), 'wb') as f:
        pickle.dump(predict, f)

    print(np.argmax(predict, 1))


def test_by_dentition(p_model):
    U_A = [11, 12, 13, 21, 22, 23]
    U_M = [18, 17, 16, 15, 14, 24, 25, 26, 27, 28]
    L_A = [31, 32, 33, 41, 42, 43]
    L_M = [48, 47, 46, 45, 44, 34, 35, 36, 37, 38]

    PICKLE_PATH = r'new_model\prediction\predict_{}.pkl'.format(p_model)

    predict = None
    with open(PICKLE_PATH, 'rb') as f:
        predict = pickle.load(f)

    predict = np.argmax(predict, 1).tolist()

    IMAGE_PATH = r'new_model\test_val'
    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle=False)
    filenames = test_generator.filenames

    x = [x.split('\\')[0] for x in filenames]
    real = []
    for i in x:
        if(i=="Buttress"):
            real.append(0)
        elif(i=="Reverse_buttress"):
            real.append(1)
        elif(i=="v-shaped"):
            real.append((2))

    dental_num = [int(x.split('\\')[1].split(',')[1]) for x in filenames]

    print(len(filenames), filenames)
    print(len(predict), predict)
    print(len(real), real)
    print(len(dental_num), dental_num)

    dic_acc = {'U_A' : {'correct':0, 'wrong':0}, 'U_M' : {'correct':0, 'wrong':0}, 'L_A' : {'correct':0, 'wrong':0}, 'L_M': {'correct':0, 'wrong':0}}

    for i in range(len(filenames)):
        if(predict[i]==real[i]):
            if(dental_num[i] in U_A):
                dic_acc['U_A']['correct'] += 1
            elif(dental_num[i] in U_M):
                dic_acc['U_M']['correct'] += 1
            elif (dental_num[i] in L_A):
                dic_acc['L_A']['correct'] += 1
            elif (dental_num[i] in L_M):
                dic_acc['L_M']['correct'] += 1
        else:
            if(dental_num[i] in U_A):
                dic_acc['U_A']['wrong'] += 1
            elif(dental_num[i] in U_M):
                dic_acc['U_M']['wrong'] += 1
            elif (dental_num[i] in L_A):
                dic_acc['L_A']['wrong'] += 1
            elif (dental_num[i] in L_M):
                dic_acc['L_M']['wrong'] += 1

    print('--{}--'.format(p_model))
    print('U_A : {}'.format(dic_acc['U_A']['correct'] / (dic_acc['U_A']['correct'] + dic_acc['U_A']['wrong'])))
    print('U_M : {}'.format(dic_acc['U_M']['correct'] / (dic_acc['U_M']['correct'] + dic_acc['U_M']['wrong'])))
    print('L_A : {}'.format(dic_acc['L_A']['correct'] / (dic_acc['L_A']['correct'] + dic_acc['L_A']['wrong'])))
    print('L_M : {}'.format(dic_acc['L_M']['correct'] / (dic_acc['L_M']['correct'] + dic_acc['L_M']['wrong'])))

    print('all correct : {}'.format(dic_acc['U_A']['correct'] + dic_acc['L_A']['correct'] + dic_acc['U_M']['correct'] + dic_acc['L_M']['correct']))


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    ax.tight_layout()
    ax.ylabel('True diagnosis')
    ax.xlabel('Predicted diagnosis')



def generate_roc_curve(p_model):
    PICKLE_PATH = r'new_model\prediction\predict_{}.pkl'.format(p_model)

    predict = None
    with open(PICKLE_PATH, 'rb') as f:
        predict = pickle.load(f)

    predict = np.argmax(predict, 1).tolist()

    IMAGE_PATH = r'new_model\test_val'
    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle=False)
    filenames = test_generator.filenames

    x = [x.split('\\')[0] for x in filenames]
    real = []
    for i in x:
        if(i=="Buttress"):
            real.append(0)
        elif(i=="Reverse_buttress"):
            real.append(1)
        elif(i=="v-shaped"):
            real.append((2))

    lb = LabelBinarizer()
    lb_predict = lb.fit_transform(predict)
    lb_real = lb.fit_transform(real)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label = ['Buttress', 'Reverse buttress', 'V-shaped']
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0

    for i, name in enumerate(label):
        thresh = 0
        fpr[i], tpr[i], thresh = roc_curve(lb_real[:, i], lb_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        tn, fp, fn, tp = confusion_matrix(lb_real[:, i], lb_predict[:, i]).ravel()
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1_score = 2 * recall * precision / (recall + precision)
        print('{}----------------'.format(name))
        print(precision, recall, f1_score)

        sum_precision += precision
        sum_recall += recall
        sum_f1_score += f1_score

    print('{} modal !!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(p_model))
    print('precision : {}'.format(sum_precision/3))
    print('recall : {}'.format(sum_recall / 3))
    print('f1_score : {}'.format(sum_f1_score / 3))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresh = roc_curve(lb_real.ravel(), lb_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print('-- roc for all')
    print(roc_auc[0], roc_auc[1], roc_auc[2])

    print('-- roc_micro')
    print(roc_auc['micro'])

    print('-- roc_macro')
    print(roc_auc['macro'])

# confusion matrix
def draw_cm_by_model(p_model):
    PICKLE_PATH = r'new_model\prediction\predict_{}.pkl'.format(p_model)

    predict = None
    with open(PICKLE_PATH, 'rb') as f:
        predict = pickle.load(f)

    predict = np.argmax(predict, 1).tolist()

    IMAGE_PATH = r'new_model\test_val'
    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle=False)
    filenames = test_generator.filenames

    x = [x.split('\\')[0] for x in filenames]
    real = []
    for i in x:
        if(i=="Buttress"):
            real.append(0)
        elif(i=="Reverse_buttress"):
            real.append(1)
        elif(i=="v-shaped"):
            real.append((2))

    cm = confusion_matrix(y_true=real, y_pred=predict)

    plot_confusion_matrix(cm, ['Buttress', 'Reverse Buttress', 'V-shaped'], title='{} without normalization'.format(p_model))
    plt.savefig('new_model\prediction\plot\{}_cm.png'.format(p_model))

def draw_cm_by_dentition(p_model):
    U_A = [11, 12, 13, 21, 22, 23]
    U_M = [18, 17, 16, 15, 14, 24, 25, 26, 27, 28]
    L_A = [31, 32, 33, 41, 42, 43]
    L_M = [48, 47, 46, 45, 44, 34, 35, 36, 37, 38]

    PICKLE_PATH = r'new_model\prediction\predict_{}.pkl'.format(p_model)

    predict = None
    with open(PICKLE_PATH, 'rb') as f:
        predict = pickle.load(f)

    predict = np.argmax(predict, 1).tolist()

    IMAGE_PATH = r'new_model\test_val'
    pic_size = (224, 224)

    if (p_model.lower() == 'inceptionv3'):
        pic_size = (299, 299)
    elif (p_model.lower() == 'xception'):
        pic_size = (229, 229)
    elif (p_model.lower() == 'resnet50v2'):
        pic_size = (224, 224)
    elif (p_model.lower() == 'resnet101v2'):
        pic_size = (224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=pic_size,
        color_mode="rgb",
        shuffle=False)
    filenames = test_generator.filenames

    x = [x.split('\\')[0] for x in filenames]
    real = []
    for i in x:
        if(i=="Buttress"):
            real.append(0)
        elif(i=="Reverse_buttress"):
            real.append(1)
        elif(i=="v-shaped"):
            real.append((2))

    dental_num = [int(x.split('\\')[1].split(',')[1]) for x in filenames]

    print(len(predict), predict)
    print(len(real), real)
    print(len(dental_num), dental_num)

    dic_predict = {'U_A' : [], 'U_M' : [], 'L_A' : [], 'L_M': []}
    dic_real = {'U_A' : [], 'U_M' : [], 'L_A' : [], 'L_M': []}

    for i in range(len(filenames)):
        if (dental_num[i] in U_A):
            dic_predict['U_A'].append(predict[i])
            dic_real['U_A'].append(real[i])
        elif (dental_num[i] in U_M):
            dic_predict['U_M'].append(predict[i])
            dic_real['U_M'].append(real[i])
        elif (dental_num[i] in L_A):
            dic_predict['L_A'].append(predict[i])
            dic_real['L_A'].append(real[i])
        elif (dental_num[i] in L_M):
            dic_predict['L_M'].append(predict[i])
            dic_real['L_M'].append(real[i])

    plt.figure()
    for i, name in enumerate(dic_predict.keys()):
        cm = confusion_matrix(y_true=dic_real[name], y_pred=dic_predict[name])
        plot_confusion_matrix(cm, classes=['Buttress', 'Reverse Buttress', 'V-shaped'],
                              title='{} without normalization\n({})'.format(p_model, name))
        plt.savefig('new_model\prediction\plot\{}_dental_cm.png'.format(p_model))

def calc_ci():
    data = [0.9205, 0.9276, 0.9435, 0.9538]
    auc = [0.9403, 0.9307, 0.9576, 0.9653]
    loss = [0.000003973, 0.001072, 0.22817, 1.001842]

    # 0.000003575-0.000004370

    site = [0.7500, 0.9426, 0.6923, 0.9427, 0.7500, 0.9490, 0.6923, 0.9114, 0.8571, 0.9490, 0.9230, 0.9531, 0.7500, 0.9808, 0.8461, 0.9687]
    len = [28, 157, 13, 192]

    loss = [0.000003973, 0.001072, 0.22817, 1.001842]

    print('loss')
    for i, d in enumerate(loss):
        min = d - math.sqrt(d / 3000)
        max = d + math.sqrt(d / 3000)
        print('%f-%f' % (min, max))