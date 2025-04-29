from imaris_ims_file_reader.ims import ims
import dask.array as da
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from tqdm import tqdm
import pickle as pkl
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


device = torch.device("cuda")

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


im_path = r'/h20/CBI/Jonathan/CLEM/Birder/88EM87C 25x25_ashlar.ome.tif'
output_dir = r'/h20/CBI/Kasia/data_analysis/2025_Birder_mito/C_00_analysis'
output_sub_dir = 'mitos_sam'
prefix_save = '88EM87C_'

df_path = os.path.join(output_dir,f'{prefix_save}axons.pkl')

axon_res = 3
mitos_res = 0
row_offset = 0 # used if df was created as a test on a smaller image
col_offset = 0 # used if df was created as a test on a smaller image

sam2_checkpoint = "/h20/CBI/Kasia/sam_install/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "/h20/CBI/Kasia/sam_install/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

os.makedirs(os.path.join(output_dir,output_sub_dir),exist_ok=True)

# get image
store = imread(im_path, aszarr=True)
im = da.from_zarr(store,mitos_res)

# import df
df = pd.read_pickle(df_path)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=2,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1000,
    use_m2m=False,
)

res_adjust = axon_res - mitos_res
mito_props = ['area', 'predicted_iou', 'stability_score']

for ind, row in tqdm(df.loc[70:, :].iterrows(), total=len(df.loc[70:, :])):

    # get the cell interior image of high res
    row_start = (int(row['inside_bbox-0']) + row_offset)*2**res_adjust
    row_end = (int(row['inside_bbox-2']) + row_offset)*2**res_adjust
    col_start = (int(row['inside_bbox-1']) + col_offset)*2**res_adjust
    col_end = (int(row['inside_bbox-3']) + col_offset)*2**res_adjust

    im_test = im[row_start:row_end, col_start:col_end]
    
    # run the segmentation
    im_rgb = gray2rgb(im_test).compute()
    masks_org = mask_generator.generate(im_rgb)

    # saving the masks
    file_name = f'{prefix_save}{str(row.label).zfill(6)}_mito.pkl'
    pkl.dump(masks_org, open(os.path.join(output_dir,output_sub_dir,file_name), 'wb')) 
