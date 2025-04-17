import dask.array as da
import pandas as pd
import numpy as np
from itertools import combinations

from skimage.measure import label,regionprops_table, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects, disk
from skimage.morphology import remove_small_holes
from scipy.ndimage import binary_fill_holes

def extract_regionprops(row,properties,small_size = 100):
    '''
    Function to transform output of SAM into separate objects and compute regionprops.
    The function takes a row of the dataframe, removes small objects and holes, labels the image (separation of objects if a single SAM mask contains multiple) and calculates regionprops.
    Returns a dataframe with regionprops for each object.
    '''
    
    # get bounding box
    pad_size = 5
    bbox = [int(x) for x in row['bbox']]
    im_mask = row['segmentation'][bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    im_mask_1 = remove_small_objects(im_mask, min_size=small_size, connectivity=1)
    padded = np.pad(im_mask_1, pad_width=pad_size, mode='constant', constant_values=False)
    im_mask_2 = remove_small_holes(padded, area_threshold=small_size, connectivity=1)
    im_mask_2 = im_mask_2[5:-5, 5:-5]
    im_mask_clean = label(im_mask_2)
    
    props = pd.DataFrame(regionprops_table(im_mask_clean, properties=properties))

    props['bbox-0'] = props['bbox-0'] + bbox[1]
    props['bbox-1'] = props['bbox-1'] + bbox[0]
    props['bbox-2'] = props['bbox-2'] + bbox[1]
    props['bbox-3'] = props['bbox-3'] + bbox[0]

    props['centroid-0'] = props['centroid-0'] + bbox[1]
    props['centroid-1'] = props['centroid-1'] + bbox[0]

    bbox_cols = ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']
    props[bbox_cols] = props[bbox_cols].astype(int)
    
    # Each regionprops_table output is a dictionary 
    return props

def compute_iou_array(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """

    # Format: (min_row, min_col, max_row, max_col)
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_area = max(0, inter_ymax - inter_ymin) * max(0, inter_xmax - inter_xmin)

    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def suppress_by_iou(df, iou_threshold = 0.5):
    """
    Function to suppress overlapping objects based on IoU and solidity.
    It is used for detection of myelinated axons. 
    Therefore ring-shaped objects cannot by suppressed by the solid objects.
    """
    
    df = df.copy()
    df['keep'] = 1  # Default to keep everything

    bboxes = df[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values
    scores = df['solidity'].values
    eulers = df['euler_number'].values

    for i, j in combinations(range(len(df)), 2):
        iou = compute_iou_array(bboxes[i], bboxes[j])
        
        if iou > iou_threshold:
            i_ring = eulers[i] < 1
            j_ring = eulers[j] < 1

            if i_ring and not j_ring:
                df.at[j, 'keep'] = 0
            elif j_ring and not i_ring:
                df.at[i, 'keep'] = 0
            else:
                if scores[i] >= scores[j]:
                    df.at[i, 'keep'] = 0
                else:
                    df.at[j, 'keep'] = 0

    return df

def suppress_by_iou_mitos(df, iou_threshold = 0.5):
    """
    Function to suppress overlapping objects based on IoU and eccentricity.
    It is used for detection of mitochondria.
    """
    
    df = df.copy()
    df['keep'] = 1  # Default to keep everything

    bboxes = df[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values
    scores = df['eccentricity'].values

    for i, j in combinations(range(len(df)), 2):
        iou = compute_iou_array(bboxes[i], bboxes[j])
        
        if iou > iou_threshold:

            if scores[i] <= scores[j]:
                df.at[i, 'keep'] = 0
            else:
                df.at[j, 'keep'] = 0

    return df

def mask_from_df(df, shape, prefix=''):
    '''
    Function to create a mask from the dataframe containing regionprops style info.
    '''
    
    mask = np.zeros(shape, dtype=np.uint16)
    for i, row in df.iterrows():
        row_start = int(row[f'{prefix}bbox-0'])
        row_stop = int(row[f'{prefix}bbox-2'])
        col_start = int(row[f'{prefix}bbox-1'])
        col_stop = int(row[f'{prefix}bbox-3'])
        patch = mask[row_start:row_stop, col_start:col_stop]
        mask[row_start:row_stop, col_start:col_stop] = patch + row[f'{prefix}image']*row[f'{prefix}label']
    return mask

def find_edge_df(df_res, im_shape, skip_labels = [], pad_size = 25):
    '''
    Function to find edge regions in the dataframe.
    '''

    df_edge = df_res.loc[df_res.keep == 1,:]
    df_edge.loc[df_edge['bbox-0'] < pad_size,'edge_side'] = 'top'
    df_edge.loc[df_edge['bbox-1'] < pad_size,'edge_side'] = 'left'
    df_edge.loc[df_edge['bbox-2'] > (im_shape[0]-pad_size),'edge_side'] = 'bottom'
    df_edge.loc[df_edge['bbox-3'] > (im_shape[1]-pad_size),'edge_side'] = 'right'

    # drop non-edge
    df_edge = df_edge.loc[df_edge['edge_side'].notnull(),:]
    # drop if appears in df_sel
    df_edge = df_edge.loc[~df_edge['label'].isin(skip_labels),:]

    # drop if the area doesn't resemble a cut ring
    for ind,row in df_edge.iterrows():

        candidate_mask = row.image.astype(np.uint8).copy()

        if row.edge_side == 'left':
            candidate_mask[:,:5] = 1
        elif row.edge_side == 'top':
            candidate_mask[:5,:] = 1
        elif row.edge_side == 'right':
            candidate_mask[:,-5:] = 1    
        elif row.edge_side == 'bottom':
            candidate_mask[-5:,:] = 1
            
        prop = regionprops(candidate_mask)
        df_edge.loc[ind, 'edge_ring'] = ((prop[0]['euler_number'] < 1) & (prop[0]['area']/prop[0]['area_filled'] < 0.9) & (prop[0]['area_filled'] - prop[0]['area'] > 500))

    df_edge = df_edge.loc[df_edge.edge_ring == True,:]

    return df_edge

def add_soma_data(df, inside_props, pad = 3):
    """
    Function adds axon soma parameters to the dataframe.
    """

    # empty column to keep images
    df['inside_image'] = None
    df['inside_image'] = df['inside_image'].astype(object)

    for ind,row in df.iterrows():
        
        mask = row['image'].copy()
        # pre-process the edge objects
        if row['edge_ring']:
            if row['edge_side'] == 'top':
                mask[:pad,:] = 1
            elif row['edge_side'] == 'bottom':
                mask[-pad:,:] = 1
            elif row['edge_side'] == 'left':
                mask[:,:pad] = 1
            elif row['edge_side'] == 'right':
                mask[:,-pad:] = 1

        
        mask_full = binary_fill_holes(mask)
        mask_inside = mask_full ^ mask

        if row['edge_ring']:
            if row['edge_side'] == 'top':
                mask_inside[:pad,:] = 0
            elif row['edge_side'] == 'bottom':
                mask_inside[-pad:,:] = 0
            elif row['edge_side'] == 'left':
                mask_inside[:,:pad] = 0
            elif row['edge_side'] == 'right':
                mask_inside[:,-pad:] = 0
        
        # collect the inside mask properties
        props = regionprops(label(mask_inside))
        if len(props) > 1:

            props = [max(props, key=lambda x: x.area)]

        df.at[ind,'inside_image'] = props[0].image

        df.loc[ind,'inside_bbox-0'] = int(props[0].bbox[0] + row['bbox-0'])
        df.loc[ind,'inside_bbox-1'] = int(props[0].bbox[1] + row['bbox-1'])
        df.loc[ind,'inside_bbox-2'] = int(props[0].bbox[2] + row['bbox-0'])
        df.loc[ind,'inside_bbox-3'] = int(props[0].bbox[3] + row['bbox-1'])

        for col in inside_props:
            df.loc[ind, f'inside_{col}'] = props[0][col]

    return df
    
