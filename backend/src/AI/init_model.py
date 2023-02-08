from src.AI.config import MODEL_CONFIG
from src.AI.image_processing import image_to_tensor
from src.AI.image32_processing import image32_to_tensor
from src.AI.inference import SimpleMCNInference, AEInference

import os
import asyncio

MODEL = str(os.getenv("USE_MODEL")).lower()
is_load = True
if is_load:
    image_tensors, item_data = asyncio.run(image_to_tensor())
    dummy = item_data[item_data["category"] == "dummy"]

    if MODEL == "newmf":
        newMF = InferenceNewMF(model_config=MODEL_CONFIG["newMF"])
        asyncio.run(newMF.load_model())
    elif MODEL == "mcn":
        mcn = MCNInference(
            model_config=MODEL_CONFIG["SimpleMCN"], image_tensors=image_tensors, dummy=dummy
        )
        asyncio.run(mcn.load_model())
    elif MODEL =="simplemcn":
        simple_mcn = SimpleMCNInference(
            model_config=MODEL_CONFIG["SimpleMCN"], image_tensors=image_tensors, dummy=dummy
        )
        asyncio.run(simple_mcn.load_model())
    elif MODEL == "autoencoder":
        image_tensors, item_data = asyncio.run(image32_to_tensor())
        ae_predictor = AEInference(
            model_config=MODEL_CONFIG["AutoEncoderPredictor"], image_tensors=image_tensors, dummy=dummy
        )
        asyncio.run(ae_predictor.load_model())
    is_load = False


async def get_model():
    if MODEL == "newmf":
        yield newMF
    elif MODEL == "mcn":
        yield mcn
    elif MODEL =="simplemcn":
        yield simple_mcn
    elif MODEL == "autoencoder":
        yield ae_predictor
