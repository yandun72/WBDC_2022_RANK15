import torch
from torch.utils.data import SequentialSampler, DataLoader

from config_inference import parse_args
from data_helper_inference import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model_inference import *
import json
import pandas as pd
from tqdm import tqdm
def inference():
    args = parse_args()
    # 1. load data
    with open(args.test_annotation, 'r', encoding='utf8') as f:
         json_anns = json.load(f)
    anns = pd.DataFrame(json_anns)

    dataset = MultiModalDataset(args, args.test_zip_feats,anns, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # 2. load model
    model = Clip_Vit_Cross_Model(args,is_inference=True)
    checkpoint = torch.load('./inference_model/model_macbert_seed_2012_ema0.999_epoch_2_2812_mean_f1_0.7135.bin', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    len_dataloader = len(dataloader)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            with autocast():
                pred_label_id = model(batch, inference=True)    
            predictions.extend(pred_label_id.cpu().numpy())


    # 4. dump results
    with open('./result.csv', 'w') as f:
        for pred_label_id, ids in zip(predictions, anns.id.to_list()):
            video_id = ids
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    

if __name__ == '__main__':
    inference()
