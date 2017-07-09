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

def test_faster_rcnn_reader(path):
    gt_list, obj_num = reader.read_GT_list(path)
    
    print gt_list[1], obj_num
    
    GT_data, bbox_data, fts_data = reader.read_concatenate_all(gt_list[0])
    
    print GT_data[0], bbox_data[0]
    
    batch_xs, frame_id = reader.extractSingleSample(285, bbox_data, fts_data, 3, 4100)
    
    print batch_xs[1][4096], batch_xs[2][4096], frame_id
    
    batch_ys = reader.extractSingleGTBBox(frame_id, GT_data, 3)
    
    print batch_ys[0]

if __name__ == "__main__":
    GT_list_path = '../../py-faster-rcnn/data/MOT2016/GT_list.txt'
    test_faster_rcnn_reader(GT_list_path)