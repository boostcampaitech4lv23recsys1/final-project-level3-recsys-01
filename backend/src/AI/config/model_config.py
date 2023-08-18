MODEL_CONFIG = {
    "newMF": {
        "n_factors": 20,
        "model_path": "src/AI/save_model/newMF/NewMF_latest.pt",
        "top_k": 3,
    },
    "MCN": {
        "model_path": "src/AI/save_model/MCN/MCN_latest.pt",
        "embed_size": 128,
        "pe_off": False,
        "pretrained": False,
        "resnet_layer_num": 18,
        "hidden_sizes": [128],
        "item_num": 7
        "top_k": 10,
        "batch_size": 32,
    },
    "SimpleMCN": {
        "model_path": "src/AI/save_model/SimpleMCN/SimpleMCN_latest.pt",
        "batch_size": 32,
        "top_k": 10,
    },
    "AutoEncoderPredictor": {
        "model_path": "src/AI/save_model/AutoEncoderPredictor/AutoEncoderPredictor_latest.pt",
        "pretraind_model_dir": "src/AI/save_model/AutoEncoder",
        "dropout_prop": 0.2,
        "top_k": 10,
        "batch_size": 32
    }
}
