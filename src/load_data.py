from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.const.load_data_const import SEED_TRAIN, SEED_TEST, SEED_VAL
from src.const.general_const import BASE_DATA_PATH, IMG_SIZE
from src.centerpoint import get_keypoints, get_gaussian
import pandas as pd
import glob
import os

def get_data():
    ################################################
    # Save all the path of the images to data frames
    ################################################
    # TODO: write test that checks that all the df are correlated
    # Input
    train_X_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"leftImg8bit/train/*/*"))})
    test_X_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"leftImg8bit/test/*/*"))})
    val_X_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"leftImg8bit/val/*/*"))})

    # GT Train
    train_gtFine_color_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/train/*/*color*"))})
    train_gtFine_instanceIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/train/*/*instanceIds*"))})
    train_gtFine_labelIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/train/*/*labelIds*"))})

    # GT Test
    test_gtFine_color_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/test/*/*color*"))})
    test_gtFine_instanceIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/test/*/*instanceIds*"))})
    test_gtFine_labelIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/test/*/*labelIds*"))})

    # GT Validation
    val_gtFine_color_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/val/*/*color*"))})
    val_gtFine_instanceIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/val/*/*instanceIds*"))})
    val_gtFine_labelIds_path_df = pd.DataFrame({"filename":glob.glob(os.path.join(BASE_DATA_PATH,"gtFine/val/*/*labelIds*"))})



    #############################################################
    # create ImageDataGenerator instances with the same arguments
    #############################################################
    data_gen_args = dict(zoom_range=[0.5, 2.0],
                        horizontal_flip=True)

    # Input
    train_X_datagen = ImageDataGenerator(**data_gen_args)
    test_X_datagen = ImageDataGenerator(**data_gen_args)
    val_X_datagen = ImageDataGenerator(**data_gen_args)

    # GT Train
    train_gtFine_color_datagen = ImageDataGenerator(**data_gen_args)
    train_gtFine_instanceIds_datagen = ImageDataGenerator(**data_gen_args,
                                                          preprocessing_function=get_keypoints)
    train_gtFine_labelIds_datagen = ImageDataGenerator(**data_gen_args,
                                                       preprocessing_function=get_gaussian)

    # GT Test
    test_gtFine_color_datagen = ImageDataGenerator(**data_gen_args)
    test_gtFine_instanceIds_datagen = ImageDataGenerator(**data_gen_args,
                                                         preprocessing_function=get_keypoints)
    test_gtFine_labelIds_datagen = ImageDataGenerator(**data_gen_args,
                                                      preprocessing_function=get_gaussian)

    # GT Validation
    val_gtFine_color_datagen = ImageDataGenerator(**data_gen_args)
    val_gtFine_instanceIds_datagen = ImageDataGenerator(**data_gen_args,
                                                        preprocessing_function=get_keypoints)
    val_gtFine_labelIds_datagen = ImageDataGenerator(**data_gen_args,
                                                     preprocessing_function=get_gaussian)


    #############################################################
    # Create flow_from_dataframe generators
    #############################################################

    # Input
    train_X_generator = train_X_datagen.flow_from_dataframe(
        train_X_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
        batch_size=2, seed=SEED_TRAIN,weight_col=None)

    test_X_generator = test_X_datagen.flow_from_dataframe(
        test_X_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
        batch_size=2, seed=SEED_TEST,weight_col=None)

    val_X_generator = val_X_datagen.flow_from_dataframe(
        val_X_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
        batch_size=2, seed=SEED_VAL,weight_col=None)


    # GT Train
    train_gtFine_color_generator = train_gtFine_color_datagen.flow_from_dataframe(
    train_gtFine_color_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TRAIN,weight_col=None)

    train_gtFine_keypointTargets_generator = train_gtFine_instanceIds_datagen.flow_from_dataframe(
    train_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TRAIN,weight_col=None)

    train_gtFine_gaussianTargets_generator = train_gtFine_labelIds_datagen.flow_from_dataframe(
    train_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TRAIN,weight_col=None)

    # GT Test
    test_gtFine_color_generator = test_gtFine_color_datagen.flow_from_dataframe(
    test_gtFine_color_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TEST,weight_col=None)

    test_gtFine_keypointTargets_generator = test_gtFine_instanceIds_datagen.flow_from_dataframe(
    test_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TEST,weight_col=None)

    test_gtFine_gaussianTargets_generator = test_gtFine_labelIds_datagen.flow_from_dataframe(
    test_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_TEST,weight_col=None)

    # GT Validation
    val_gtFine_color_generator = val_gtFine_color_datagen.flow_from_dataframe(
    val_gtFine_color_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_VAL,weight_col=None)

    val_gtFine_keypointTargets_generator = val_gtFine_instanceIds_datagen.flow_from_dataframe(
    val_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_VAL,weight_col=None)

    val_gtFine_gaussianTargets_generator = val_gtFine_labelIds_datagen.flow_from_dataframe(
    val_gtFine_instanceIds_path_df, y_col=None, target_size=IMG_SIZE,class_mode=None,
    batch_size=2, seed=SEED_VAL,weight_col=None)


    ################################################################
    # combine generators of GT into one which yields 3 target images
    ################################################################
    # TODO: instead of using zip change it to dictionary based on Simon's key
    train_gt_generator = zip(train_gtFine_color_generator, train_gtFine_keypointTargets_generator, train_gtFine_gaussianTargets_generator)
    test_gt_generator = zip(test_gtFine_color_generator, test_gtFine_keypointTargets_generator, test_gtFine_gaussianTargets_generator)
    val_gt_generator = zip(val_gtFine_color_generator, val_gtFine_keypointTargets_generator, val_gtFine_gaussianTargets_generator)


    ##########################################################
    # combine generators into one which yields image and masks
    ##########################################################
    train_generator = zip(train_X_generator, train_gt_generator)
    test_generator = zip(test_X_generator, test_gt_generator)
    val_generator = zip(val_X_generator, val_gt_generator)

    return train_generator, test_generator, val_generator
