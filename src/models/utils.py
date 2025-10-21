
import torch

def get_in_channels(data_code):
    in_ch = -1
    if data_code == "cmnist":
        in_ch = 1
    elif data_code == 'coco-animals':
        in_ch = 3
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    raise ValueError("Invalid or not supported dataset [{}]".format(data_code))

def get_embedded_data(pl_model, dl):
    embs = []
    for batch in dl():
        emb = pl_model.model.encode(batch[0].to(pl_model.device))
        embs.append(emb.detach().cpu())
    embs = torch.cat(embs)
    return embs

