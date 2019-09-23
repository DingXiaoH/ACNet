from base_model.resnet import *
from base_model.wrn import *

IMAGENET_MODEL_MAP = {
    'resnet34':create_ResNet34,
    'resnet50':create_ResNet50,
    'resnet101':create_ResNet101,
    'resnet152':create_ResNet152
}


CIFAR10_MODEL_MAP = {
    'rc56':create_RC56,
    'rc110':create_RC110,
    'rc164':create_RC164,

    'wrnc16plain':create_wrnc16plain,
    'wrnc16drop':create_wrnc16drop,
    'wrnc28plain':create_wrnc28plain,
    'wrnc28drop':create_wrnc28drop,
    'wrnc40plain':create_wrnc40plain,
    'wrnc40drop':create_wrnc40drop,

}

CH_MODEL_MAP = {
    'wrnh16plain':create_wrnh16plain,
    'wrnh16drop':create_wrnh16drop,
    'wrnh28plain':create_wrnh28plain,
    'wrnh28drop':create_wrnh28drop,
    'wrnh40plain':create_wrnh40plain,
    'wrnh40drop':create_wrnh40drop,
}

SVHN_MODEL_MAP = {

}

DATASET_TO_MODEL_MAP = {
    'imagenet': IMAGENET_MODEL_MAP,
    'cifar10': CIFAR10_MODEL_MAP,
    'ch': CH_MODEL_MAP,           #ch for cifar-100
    'svhn': SVHN_MODEL_MAP
}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    return DATASET_TO_MODEL_MAP[dataset_name][model_name]
