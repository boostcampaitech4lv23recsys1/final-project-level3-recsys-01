from src.AI.config import MODEL_CONFIG
from src.AI.image_processing import image_to_tensor
from src.AI.inference import SimpleMCNInference


import asyncio


is_load = True
if is_load:
    image_tensors, item_data = asyncio.run(image_to_tensor())
    dummy = item_data[item_data["category"] == "dummy"]

    # 필요에 따라 모델 가져다가 쓰세요~
    # newMF = InferenceNewMF(model_config=MODEL_CONFIG["newMF"])
    # asyncio.run(newMF.load_model())

    # mcn = MCNInference(
    #     model_config=MODEL_CONFIG["SimpleMCN"], image_tensors=image_tensors, dummy=dummy
    # )
    # asyncio.run(mcn.load_model())

    simple_mcn = SimpleMCNInference(
        model_config=MODEL_CONFIG["SimpleMCN"], image_tensors=image_tensors, dummy=dummy
    )
    asyncio.run(simple_mcn.load_model())
    is_load = False


async def get_model():
    yield simple_mcn
