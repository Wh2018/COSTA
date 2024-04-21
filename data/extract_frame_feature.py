import cv2
import os
import pdb
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel 

def pick_arange(arange, num):

    if num > len(arange):
        return arange

    else:
        output = np.array([])
        seg = len(arange) / num
        for n in range(num):
            if int(seg * (n+1)) >= len(arange):
                output = np.append(output, arange[int(seg * n)])
                #output = np.append(output, arange[-1])
            else:
                output = np.append(output, arange[int(seg * n)])
        return output
    
def frame_dump():
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

    errs = []
    base_dir = 'Charades_videos/'
    videos_path = os.listdir(base_dir)
    videos_path.sort(reverse=False)
    
    for video in tqdm(videos_path):
        print(video)
        output_dir = 'Charades_frames/' + video[:5] + '/'
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = base_dir + video
        
        video_cv = cv2.VideoCapture(video_path)
        frame_all = video_cv.get(cv2.CAP_PROP_FRAME_COUNT)
        

        frame_indexs = []
        for i in range(int(frame_all)):
            frame_indexs.append(i)
        frame_indexs = pick_arange(frame_indexs, 32)
        
        if len(frame_indexs) != 32:
            err = video
            errs.append(err)
            continue
        
        images = []
        for frame_idx in frame_indexs:
            frame_idx = int(frame_idx)
            video_cv.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            rval, frame = video_cv.read()
            
            frame_path = output_dir + str(frame_idx) + '.jpg'
            cv2.imwrite(frame_path, frame)
            
            image = Image.open(frame_path)
            images.append(image)
            
            os.remove(frame_path)

        output_feature_dir = 'Charades_ViT-B16/' + video[:5]
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        
        output_features = np.array(last_hidden_state.detach().cpu().numpy())
        np.save(output_feature_dir, output_features)
        

    with open('err.txt', 'w') as f:
        for err in errs:
            f.write(err + '\n')
        
frame_dump()