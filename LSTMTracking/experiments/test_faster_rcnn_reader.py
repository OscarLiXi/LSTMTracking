# Copyright (c) <2017> <Xi Li>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

'''
test read image features and bbox from faster r-cnn dectection, hlcv project ss17
'''

import faster_rcnn_reader as reader
import numpy as np
import os

def read_bbox_all(GT_path):
    bbox_path = os.path.dirname(GT_path) + "/../Estimated_BBox/"
    #fts_path = os.path.dirname(GT_path) + "/../Extracted_Features/"
    info = np.genfromtxt(GT_path, dtype = 'unicode', delimiter=',',max_rows = 1)
    
    bbox_path = bbox_path + info[1] + '-bbox.txt'
    #fts_path = fts_path + info[1] + '-fts.txt'
    
    GT_data = np.genfromtxt(GT_path, delimiter=',',skip_header = 1)
    bbox_data = np.genfromtxt(bbox_path, delimiter=',')                
    #f = open(fts_path, 'rb')
    #fts_data = np.array([], dtype='string')
    #for line in f:
    #    fts_data = np.append(fts_data, line)
          
    return GT_data, bbox_data

def test_faster_rcnn_reader(path):
    gt_list, obj_num = reader.read_GT_list(path)
    
    print gt_list[1], obj_num
    
    GT_data, bbox_data, fts_data = reader.read_concatenate_all('../../py-faster-rcnn/data/MOT2016/GT/MOT16-02-gt-023.txt')
    
    print GT_data[0], bbox_data[0]
    
    batch_xs, frame_id = reader.extractSingleSample(0, bbox_data, fts_data, 1, 4100)
    
    print batch_xs[0][4096], batch_xs[0][4096], frame_id
    
    batch_ys = reader.extractSingleGTBBox(frame_id, GT_data, 1)
    
    print batch_ys[0]
    
def prune_gt_list(path):
    gt_list, obj_num = reader.read_GT_list(path)
    sum_length = 0
    count = 0
    for i in range(len(gt_list)):
        GT_data, bbox_data = read_bbox_all(gt_list[i])
        if len(np.shape(bbox_data)) > 1:
            n = len(bbox_data)
            if (n > 100 and n < 600):
                print gt_list[i]
                count += 1
                sum_length += n
                with open(GT_list_path+"1", 'ab') as f:                        
                    f.write(os.path.basename(gt_list[i]) + ',')      
                if sum_length > 15000:
                    break 

    print sum_length, count
if __name__ == "__main__":
    GT_list_path = '../../py-faster-rcnn/data/MOT2016/GT_list.txt'
    #test_faster_rcnn_reader(GT_list_path)
    prune_gt_list(GT_list_path)