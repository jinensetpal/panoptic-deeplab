from src.models import backbone_encoder, decoder, heads
from tensorflow.keras import Input, Sequential, Model
from ..const.general_const import INPUT_SHAPE

def get_model(input_shape=None):
    if not input_shape:
        input_shape = INPUT_SHAPE

    inp = Input(shape=input_shape)
    backbone, res2, res3 = backbone_encoder.create_backbone_model()
    print(type(backbone))
    print(backbone)
    backbone = Sequential([backbone])
    sem_decoder, inst_decoder = decoder.get_decoder('semantic_decoder'), decoder.get_decoder('instance_decoder')
    sem_head, inst_ctr_head, inst_rgr_head = Sequential([heads.get_semantic_head()]), Sequential([heads.get_instance_center_head()]), Sequential([heads.get_instance_regression_head()])
    
    latent = backbone(inp)
    print(type(latent))
    print(latent)
    
    latent = {'res5': latent,
              'res2': res2,
              'res3': res3}
    
    sem_latent, inst_latent = sem_decoder(latent), inst_decoder(latent)
    sem_output, inst_ctr_output, inst_rgr_output = sem_head(sem_latent), inst_ctr_head(inst_latent), inst_rgr_head(inst_latent)
    
    model = Model(inputs=inp, outputs=[sem_output, inst_ctr_output, inst_rgr_output])
    return model
