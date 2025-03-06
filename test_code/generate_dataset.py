import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import random
import numpy as np

# sets_train_NPS = ['Clip_02', 'Clip_08', 'Clip_09', 'Clip_11', 'Clip_14', 'Clip_15', 'Clip_16', 'Clip_17', 'Clip_18',
#                   'Clip_20', 'Clip_21', 'Clip_22', 'Clip_23', 'Clip_24', 'Clip_25', 'Clip_27', 'Clip_28', 'Clip_29',
#                   'Clip_30',
#                   'Clip_31', 'Clip_32', 'Clip_33', 'Clip_34', 'Clip_36', 'Clip_37', 'Clip_38', 'Clip_39', 'Clip_40',
#                   'Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49',
#                   'Clip_50']
#
# sets_test_NPS = ['Clip_01', 'Clip_03', 'Clip_04', 'Clip_05', 'Clip_06', 'Clip_07', 'Clip_10', 'Clip_12', 'Clip_13',
#                  'Clip_19', 'Clip_26', 'Clip_35']

# sets_DvB_test = ['00_10_09_to_00_10_40', '2019_08_19_GOPR5869_1530_phantom', '2019_10_16_C0003_1700_matrice',
#                  '2019_11_14_C0001_3922_matrice', 'distant_parrot_with_birds', 'dji_matrice_210_mountain',
#                  'dji_mavick_close_buildings', 'dji_phantom_4_hillside_cross', 'fixed_wing_over_hill_2', 'gopro_004',
#                  'GOPR5842_002', 'GOPR5845_001', 'GOPR5848_004', 'gopro_008', 'parrot_clear_birds_med_range']
#
# sets_DvB_train = ['2019_08_19_GP015869_1520_inspire', 'GOPR5842_005', 'dji_mavick_mountain_cruise',
#                   'dji_phantom_4_mountain_hover',
#                   'GOPR5847_004', 'GOPR5847_003', 'off_focus_parrot_birds', 'dji_matrice_210_off_focus',
#                   'parrot_disco_distant_cross_3',
#                   'gopro_006', 'parrot_disco_zoomin_zoomout', 'GOPR5846_002', 'GOPR5845_004', 'GOPR5843_005',
#                   'dji_mavick_distant_hillside', 'GOPR5844_002', 'parrot_clear_birds',
#                   'dji_pantom_landing_custom_fixed_takeoff',
#                   'distant_parrot_2', 'gopro_000', '2019_10_16_C0003_5043_mavic', 'gopro_002', 'GOPR5848_002',
#                   'parrot_disco_distant_cross', 'gopro_007', 'dji_mavick_mountain', 'dji_mavick_hillside_off_focus',
#                   'swarm_dji_phantom',
#                   'GOPR5844_004', 'dji_phantom_4_swarm_noon', 'GOPR5846_005', 'custom_fixed_wing_1',
#                   'parrot_disco_midrange_cross', 'two_parrot_disco_1', 'GOPR5843_002', 'swarm_dji_phantom4_2',
#                   'gopro_003', 'matrice_600_2',
#                   'matrice_600_3', '2019_09_02_GOPR5871_1058_solo', 'dji_matrice_210_hillside',
#                   'dji_phantom_mountain_cross',
#                   'dji_matrice_210_sky', 'GOPR5842_007', '00_06_10_to_00_06_27', 'gopro_005', 'two_uavs_plus_airplane',
#                   '00_09_30_to_00_10_09', '00_01_52_to_00_01_58', 'fixed_wing_over_hill_1', 'custom_fixed_wing_2',
#                   'dji_phantom_4_long_takeoff', 'parot_disco_takeoff', '2019_10_16_C0003_4613_mavic',
#                   '2019_08_19_C0001_5319_phantom',
#                   'parrot_disco_long_session', '2019_09_02_C0002_3700_mavic', 'gopro_001',
#                   '2019_09_02_C0002_2527_inspire', '00_02_45_to_00_03_10_cut']

# # all dataset
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom09', 'phantom10', 'phantom14',
        'phantom17', 'phantom19',
        'phantom20', 'phantom22', 'phantom28', 'phantom29', 'phantom30', 'phantom32', 'phantom36', 'phantom39',
        'phantom40', 'phantom41',
        'phantom42', 'phantom43', 'phantom44', 'phantom45', 'phantom46', 'phantom47', 'phantom50', 'phantom54',
        'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom63', 'phantom64', 'phantom65', 'phantom66',
        'phantom68', 'phantom70',
        'phantom71', 'phantom73', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom79',
        'phantom80', 'phantom81',
        'phantom82', 'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom92',
        'phantom93', 'phantom94',
        'phantom95', 'phantom97', 'phantom101', 'phantom102', 'phantom103', 'phantom104', 'phantom105', 'phantom106',
        'phantom107', 'phantom108',
        'phantom109', 'phantom110', 'phantom111', 'phantom112', 'phantom113', 'phantom114', 'phantom115', 'phantom116',
        'phantom117', 'phantom118',
        'phantom119', 'phantom120', 'phantom132', 'phantom133', 'phantom135', 'phantom136', 'phantom137', 'phantom138',
        'phantom139', 'phantom140',
        'phantom141', 'phantom142', 'phantom143', 'phantom144', 'phantom145', 'phantom146', 'phantom147', 'phantom148',
        'phantom149', 'phantom150']

# train dataset
sets = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29',
        'phantom30', 'phantom32',
        'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65',
        'phantom66', 'phantom68',
        'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80',
        'phantom81', 'phantom82',
        'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103',
        'phantom104', 'phantom105',
        'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115',
        'phantom116', 'phantom117',
        'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142',
        'phantom143', 'phantom145',
        'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

sets5 = ['phantom101']

sets_NPS = ['Clip_01', 'Clip_02', 'Clip_03', 'Clip_04', 'Clip_05', 'Clip_06', 'Clip_07', 'Clip_08', 'Clip_09', 'Clip_10',
            'Clip_11', 'Clip_12', 'Clip_13', 'Clip_14', 'Clip_15', 'Clip_16', 'Clip_17', 'Clip_18', 'Clip_19', 'Clip_20',
            'Clip_21', 'Clip_22', 'Clip_23', 'Clip_24', 'Clip_25', 'Clip_26', 'Clip_27', 'Clip_28', 'Clip_29', 'Clip_30',
            'Clip_31', 'Clip_32', 'Clip_33', 'Clip_34', 'Clip_35', 'Clip_36', 'Clip_37', 'Clip_38', 'Clip_39', 'Clip_40']

sets_NPS_test = ['Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49', 'Clip_50']

# test videos
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

# domain adaptation
# new scenes
sets_new_scenes = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
                   'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# low light adaptation
sets_low = ['phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

small_num = 0
set_es = ['phantom04', 'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom50', 'phantom54', 'phantom55', 'phantom61', 'phantom64', 'phantom73', 'phantom94']  # smaller than 144
set_rs = ['phantom02', 'phantom56', 'phantom57', 'phantom58', 'phantom60', 'phantom79', 'phantom92', 'phantom102', 'phantom110', 'phantom113', 'phantom119', 'phantom141', 'phantom144']  # 144~400
set_gs = ['phantom03', 'phantom05', 'phantom47', 'phantom93']  # 400~1024
set_viusal = 'phantom05'

Drone_name = ['DvsB27', 'DvsB56', 'DvsB38', 'DvsB44', 'DvsB36', 'DvsB08',
              'DvsB46', 'DvsB18', 'DvsB47', 'DvsB04', 'DvsB68', 'DvsB60']

for video_sets in Drone_name:
    id = video_sets
    # imgdir = "/home/user-guo/data/drone-dataset/phantom-dataset/images/" + id + "/"
    # annodir = '/home/user-guo/data/drone-dataset/phantom-dataset/Annotations/' + id + '/'
    # maskdir = '/home/user-guo/data/drone-dataset/phantom-dataset/mask32/' + id + '/'

    imgdir = "/home/user-guo/data/drone-videos/Drone-vs-Bird/DvsB/images/" + id + "/"
    annodir = '/home/user-guo/data/drone-videos/Drone-vs-Bird/DvsB/Annotations/' + id + '/'
    maskdir = '/home/user-guo/data/drone-videos/Drone-vs-Bird/DvsB/mask22/' + id + '/'

    imgdest = '/home/user-guo/Documents/YOLOMG/datasets/DvsB_mask22/images/'
    annodest = '/home/user-guo/Documents/YOLOMG/datasets/DvsB_mask22/Annotations/'
    maskdest = '/home/user-guo/Documents/YOLOMG/datasets/DvsB_mask22/mask22/'

    if not os.path.exists(imgdest):
        os.makedirs(imgdest)

    if not os.path.exists(annodest):
        os.makedirs(annodest)

    if not os.path.exists(maskdest):
        os.makedirs(maskdest)

    # end_index = int(len(image_list)/3)

    # for image in image_list:
    #     img = cv2.imread(imgdir + image)
    #     img_prefix = image.split('_')[0]
    #     # pic_height, pic_width, pic_depth = img.shape[0], img.shape[1], img.shape[2]
    #     break

    image_list = os.listdir(maskdir)
    num_of_image = len(image_list)
    end_index = int(num_of_image/1)
    # num_of_train = int(num_of_image * 0.4)
    # train_list = np.sort(random.sample(image_list, num_of_train))

    # for image in train_list:
    for i in range(end_index):
        image = id + '_' + str(i*1 + 2).zfill(4)
        name = image.split(".")
        imgname = name[0] + '.jpg'
        xmlname = name[0] + '.xml'

        img_path = os.path.join(imgdir, imgname)
        mask_path = os.path.join(maskdir, imgname)
        xml_path = os.path.join(annodir, xmlname)

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.find('object') is None:
            continue

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b1 = float(xmlbox.find('xmin').text)
            b2 = float(xmlbox.find('xmax').text)
            b3 = float(xmlbox.find('ymin').text)
            b4 = float(xmlbox.find('ymax').text)
            area = (b2 - b1) * (b4 - b3)

        # 对于global detector, area_thresh = 12 * 12, 对于local detector, area_thresh = 25
        if area >= 25:
            shutil.copy(img_path, imgdest)
            shutil.copy(mask_path, maskdest)
            shutil.copy(xml_path, annodest)
            print(xmlname)
        else:
            print(xmlname, end=' ')
            small_num = small_num + 1
            print('too small object: ', area)

print('small object num: ', small_num)

# for i in range(end_index):
#     # for image in image_list:
#     num = str(i * 3 + 1)
#     # num = str(i)
#     imgname = id + '_' + num.zfill(4) + '.jpg'
#     xmlname = id + '_' + num.zfill(4) + '.xml'
#
#     # image_pre, ext = os.path.splitext(image)
#     # imgname = image_pre + '.jpg'
#     # xmlname = image_pre + '.xml'
#
#     img_path = os.path.join(imgdir, imgname)
#     xml_path = os.path.join(annodir, xmlname)
#
#     if not os.path.exists(xml_path):
#         continue
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     if root.find('object') is None:
#         continue
#
#     # if not os.path.exists(img_path):
#     #     continue
#
#     shutil.copy(img_path, imgdest)
#     shutil.copy(xml_path, annodest)
#     print(xmlname)
