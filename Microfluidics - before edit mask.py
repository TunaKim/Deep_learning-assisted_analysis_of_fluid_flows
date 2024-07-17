"""
Mask R-CNN
Train on the toy Microfluidics dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 microfluidics.py train --dataset=/path/to/microfluidics/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 microfluidics.py train --dataset=/path/to/microfluidics/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 microfluidics.py train --dataset=/path/to/microfluidics/dataset --weights=imagenet

    # Apply color splash to an image
    python3 microfluidics.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 microfluidics.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import random
import colorsys
import skimage.draw
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/programing-pjh/python/microfluidics")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = os.path.join("D:/Microfluidics", "logs")


############################################################
#  Configurations
############################################################


class MicrofluidicsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + microfluidics(Channel, Fluid Volume at Outlet)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200 # MAXIMUN(integer) : [number of image] / [Batch size]

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class MicrofluidicsDataset(utils.Dataset):

    def load_microfluidics(self, dataset_dir, subset):
        """Load a subset of the Microfluidics dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Object", 1, "Fluid Volume at Outlet")
        
        #self.add_class("Object", 1, "Channel")
        #self.add_class("Object", 2, "Fluid Volume at Outlet")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons    = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons    = [r['shape_attributes'] for r in a['regions']] 

            if type(a['regions']) is dict:
                names       = [r['region_attributes'] for r in a['regions'].values()]
                #print(names)
            else:
                names       = [r['region_attributes'] for r in a['regions']] 
                #print(names)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a microfluidics dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #"Object" is the attributes name decided when labeling, etc. 'region_attributes': {Object:'a'}
        class_names = info["names"]
        #print("class_names : ", class_names)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

         # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if p['Object'] == 'Fluid Volume at Outlet':
                class_ids[i] = 1
            #elif p['Object'] == 'Fluid Volume at Outlet':
            #    class_ids[i] = 2
            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MicrofluidicsDataset()
    dataset_train.load_microfluidics(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MicrofluidicsDataset()
    dataset_val.load_microfluidics(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                #epochs=30,
                epochs=2000,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    #class_names = ['BG', 'Channel', 'Fluid Volume at Outlet']

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


random.seed(0)
N=90
brightness = 1.0
hsv = [(i / N, 1, brightness) for i in range(N)]
random.shuffle(hsv)
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    all_colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return all_colors
def class_color(id,prob):
    _hsv = list(hsv[id])
    # _hsv[2]=random.uniform(0.8, 1)
    _hsv[2]=prob
    color = colorsys.hsv_to_rgb(*_hsv)
    return color

def mask_cover(image, boxes, 
               masks, class_ids,
               class_names, count,
               frame_total,
               scores=None, title="",
               figsize=(16, 16), ax=None,
               show_mask=True, show_bbox=True,
               colors=None, captions=None):
    """
    Apply put on a mask.
    Edited by junhong.
    """
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("frame: {0:5d} / {1:5d}     No instances" .format(int(count), int(frame_total)))
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        print("frame: {0:5d} / {1:5d}" .format(int(count), int(frame_total)))

    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.copy()
    for i in range(N):
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        # color = colors[i]
        color = class_color(class_id,score*score*score*score)
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        # left(x) right(o)
        if x1 > int(width / 2):
            # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
            #                       alpha=0.7, linestyle="dashed",
            #                       edgecolor=color, facecolor='none')

            #Instance's rectangle
            #cv2.rectangle(masked_image, (x1, y1),(x2, y2), [int(x*255) for x in (color)],1)

            # Label
            label = class_names[class_id]

            #Counting Mask Pixels
            number_of_mask_pixels = int(np.count_nonzero(np.where(masks[:,:,i] == True)) / 2)
        
            x = random.randint(x1, (x1 + x2) // 2)
            #caption = "%s %d%%"%(label, int(score*100)) if score else label
            caption1    = "%.2f %%"%((score*100)) if score else label
            #caption2    = "%s"%(label) if score else label
            caption2    = "%d px"%(number_of_mask_pixels) if score else label
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")

            yyy1 = y1-10
            if yyy1 < 0:
                yyy1 = 0

            yyy2 = y2+16
            if yyy2 < 0:
                yyy2 = 0

            #if len(label) > 8 :
            #    cv2.putText(masked_image, caption, (x1, yyy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [int(x*255) for x in (color)],1)
            #else :
            #    cv2.putText(masked_image, caption, (x1, yyy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [int(x*255) for x in (color)],1)
            cv2.putText(masked_image, caption1, (x1, yyy1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [int(x*255) for x in (color)],1)
            cv2.putText(masked_image, caption2, (x1, yyy2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [int(x*255) for x in (color)],1)
            # Mask
            mask = masks[:, :, i]
            masked_image = visualize.apply_mask(masked_image, mask, color)


            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                # ax.add_patch(p)
                pts = np.array(verts.tolist(), np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(masked_image,[pts],True,[int(x*255) for x in (color)],1)
    return masked_image.astype(np.uint8)

# def detect_and_mask
# def detect_and_mask
# def detect_and_mask
def detect_and_mask(model, image_path=None, video_path=None): #### Edited by Junhong.
    assert image_path or video_path

    class_names = ['BG', 'Fluid Volume at Outlet']
    rows=1
    cols=1
    size=16
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        mask = mask_cover(image, r['rois'],
                          r['masks'], r['class_ids'],
                          class_names, False,
                          False,
                          r['scores'], ax=ax,
                          title="Predictions")
        ###
        #print(r['masks'])

        # Save output
        file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, mask)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        frame_total = vcapture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Define codec and create video writer
        file_name = "mask_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        ########################################################## start_pos >>
        # Set the detection point while playing in reverse. To save time.
        # 
        vcapture_copy = vcapture
        # frame_info : [[frmae number, nubmer of mask pixels, bos_x1], [...], ....]
        frame_info = []
        d = 10
        pos = int(frame_total-11)
        i_no_detection = 0
        i_detection = 0
        i_detection_prev = False
        i_detection_max = False
        start_pos = 0
        end_pos = 0
        while pos > 0:
            vcapture_copy.set(cv2.CAP_PROP_POS_FRAMES, pos)
            print("{0}              i_no_detecionint:{1:3d}, i_detection:{2:3d}, \
            i_detection_prev:{3}, i_detection_max:{4}".format(int(pos),
                                                            i_no_detection,
                                                            i_detection,
                                                            i_detection_prev,
                                                            i_detection_max))
            
            ret, image_copy = vcapture_copy.read()
            if ret:
                image_copy = image_copy[..., ::-1]
                rr=model.detect([image_copy], verbose=0)[0]
                N = rr['rois'].shape[0]
                if N > 0:
                    i_no_detection = 0
                    if i_detection_prev==True :
                        i_detection += 1
                    i_detection_prev = True

                    number_of_mask_pixels = int(np.count_nonzero(np.where(rr['masks'][:,:,0] == True)) / 2)
                    x1 = rr['rois'][0][1]
                    frame_info.append([int(pos), number_of_mask_pixels, x1])
                    pos = pos - d
                else:
                    i_detection = 0
                    if i_detection_prev==False :
                        i_no_detection += 1
                    i_detection_prev = False
                    frame_info.append([int(pos), -1, -1])
                    if i_detection_max==False :
                        pos = pos - d*5 # if no mask, skip faster
                    else :
                        pos = pos - d
                    
            
            if i_detection > 5 :
                i_detection_max = True
            if i_no_detection >= 3 and i_detection_max :
                start_pos = pos + d*3
                break

        for i in range(len(frame_info)):
            print(frame_info[i])

        #for i in range(len(frame_info)):
        #    if i == 0 :
        #        start_pos = frame_info[i][0]
        #    if start_pos > frame_info[i][0]:
        #        start_pos = frame_info[i][0]
        print("start_pos : ", start_pos)
        #
        ########################################################## start_pos //

        i_no_detection = 0
        i_detection = 0
        i_detection_prev = False
        i_detection_max = False
        #volume = [] #avg || array : number_of_mask_pixels

        # end_pos
        # to define end_pos
        #x2_of_box = [] 
        x2_max = 0
        x2_no_move = 0

        count = start_pos
        success = True
        #success = False
        vcapture.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        while success:
            #print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Mask cover
                mask = mask_cover(image, r['rois'],
                                  r['masks'], r['class_ids'],
                                  class_names, count,
                                  int(frame_total),
                                  r['scores'], ax=ax,
                                  title="Predictions")
                # RGB -> BGR to save image to video
                mask = mask[..., ::-1]
                # Add image to video writer
                vwriter.write(mask)
                count += 1

                N = r['rois'].shape[0]
                if N > 0:
                    if i_detection_max == False:
                        i_no_detection = 0
                    if i_detection_prev==True :
                        i_detection += 1
                    i_detection_prev = True

                    # end_pos
                    # very right x_pos of box
                    x2_now = r['rois'][0][3]
                    if x2_now <= x2_max and i_detection_max:
                        x2_no_move += 1
                    elif x2_now > x2_max:
                        x2_max = x2_now

                    ##avg
                    #number_of_mask_pixels = int(np.count_nonzero(np.where(r['masks'][:,:,0] == True)) / 2)
                    #volume.insert(0, number_of_mask_pixels)
                else:
                    i_detection = 0
                    if i_detection_prev==False or i_detection_max:
                        i_no_detection += 1
                    i_detection_prev = False

                    ###avg
                    ###number_of_mask_pixels = 0
                    ###volume.insert(0, number_of_mask_pixels)
                    
            if i_detection > 10 :
                i_detection_max = True
            if i_no_detection >= 10 and i_detection_max :
                break

            # end_pos
            if x2_no_move > 9:
                break;

            ##avg
            #avg_width = 15
            #if i_detection_max :
            #    avg = sum(volume[0:avg_width]) / avg_width
            #    avg0 = sum(volume[1:avg_width+1]) / avg_width
            #    if avg - avg0 < 0:
            #        print("avg - avg0 < 0")
            #        break;

        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect microfluidics.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/microfluidics/dataset/",
                        help='Directory of the Microfluidics dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "mask":
        assert args.image or args.video,\
               "Provide --image or --video to apply mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MicrofluidicsConfig()
    else:
        class InferenceConfig(MicrofluidicsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "mask":
        detect_and_mask(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash' or 'mask'".format(args.command))
