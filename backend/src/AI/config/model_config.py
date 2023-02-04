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
        "top_k": 5,
        "batch_size": 16
    },
}
