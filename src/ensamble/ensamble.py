import numpy as np
import json 
from category_id_map import lv2id_to_category_id
import pandas as pd

double_2_0 = np.load('./double_2_device_0.npy')#双流有预训练albef式
double_2_1 = np.load('./double_2_device_1.npy')


single_1_0 = np.load('./single_1_device_0.npy') #单流有预训练 qq浏览器式
single_1_1 = np.load('./single_1_device_1.npy')

single_2_0 = np.load('./single_2_device_0.npy') #单流有预训练 albef式
single_2_1 = np.load('./single_2_device_1.npy')


first_half = double_2_0 +  single_1_0 + single_2_0
final_half = double_2_1 +  single_1_1 + single_2_1

final_array = np.concatenate((first_half,final_half),axis=0)
predictions = np.argmax(final_array,axis=1)

with open('/opt/ml/input/data/annotations/test.json', 'r', encoding='utf8') as f:
     json_anns = json.load(f)
        
anns = pd.DataFrame(json_anns)
with open('/opt/ml/output/result.csv', 'w') as f:
    for pred_label_id, ids in zip(predictions, anns.id.to_list()):
        video_id = ids
        category_id = lv2id_to_category_id(pred_label_id)
        f.write(f'{video_id},{category_id}\n')
