MODEL_CONFIG = {
    "newMF": {
        "n_factors": 20,
        "model_path": "src/AI/save_model/newMF/NewMF_latest.pt",
        "top_k": 3,
    },
    "MCN": {
        "model_path": "src/AI/save_model/MCN/MCN_latest.pt",
        "embed_size": 512,
        "need_rep": True,
        "vse_off": True,
        "pe_off": False,
        "mlp_layers": 2,
        "conv_feats": "1234",
        "pretrained": True,
        "resnet_layer_num": 18,
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
