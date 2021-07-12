from src.models.backbone_encoder import create_backbone_model
from .const.const import INPUT_SHAPE





if __name__ == '__main__':

    backbone_model = create_backbone_model(input_shape=INPUT_SHAPE)

