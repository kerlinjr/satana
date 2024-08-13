import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches



def find_valid_boxes_by_area(boxes, width, height, area_threshold=0.05):
    box_idx = []
    for i in range(len(boxes)):
        box = boxes[i]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < width * height * area_threshold:
            box_idx.append(1)
        else:
            box_idx.append(0)
    return np.array(box_idx)


def calculate_iou_not_polygon(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    # Return the intersection over union value
    return iou


def is_box1_in_box2(box1, box2):
    if box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]:
        return True
    else:
        return False


def remove_skinny_boxes(boxes, threshold=0.05):
    box_idx = np.array([True for x in range(len(boxes))])
    for i in range(len(boxes)):
        box = boxes[i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        if (width / height < threshold) or (height / width < threshold):
            box_idx[i] = False
    return box_idx


def remove_overlapping_boxes(boxes, scores, threshold=0.5):
    box_idx = np.array([True for x in range(len(boxes))])
    for i in range(len(boxes)):
        box1 = boxes[i]
        for j in range(len(boxes)):
            if i != j:
                box2 = boxes[j]
                iou = calculate_iou_not_polygon(box1, box2)
                # print(iou)
                # if is_box1_in_box2(box1, box2):
                #     box_idx[j] = False
                if (iou > threshold):
                    if scores[i] > scores[j]:
                        box_idx[j] = False
                    else:
                        box_idx[i] = False

    return box_idx


def find_indices(lst, target):
    return [i for i, x in enumerate(lst) if x == target]


def display_image_with_labels(image, category_bboxes, category):
    # Load the image

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image)

    # Display the averaged polygons
    for cnt, bbox in enumerate(category_bboxes):
        x1, y1, x2, y2 = bbox
        if category[cnt] == 'AC Unit':
            color = 'blue'
        elif category[cnt] == 'AC leaking':
            color = 'red'
        else:
            ValueError('Category not recognized')

        patch = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='None')
        ax.add_patch(patch)


def filter_model_bbox_outputs(results, outputs, image, params):
    params['area_threshold'] = 0.05

    width, height = image.size
    boxes = np.array(results[0]['boxes'].tolist())
    passing_area_idx = find_valid_boxes_by_area(boxes, width, height, params['area_threshold']
                                                )
    logits = outputs['logits'][0]
    probs = torch.sigmoid(logits)
    scores = np.array(torch.max(probs, dim=-1)[0].tolist())

    passing_score_idx = (scores > params['score_threshold']).astype(int)

    qualified_box_idx = find_indices(passing_area_idx & passing_score_idx, 1)

    skinny_filter = find_indices(remove_skinny_boxes(boxes[qualified_box_idx], threshold=params['skinny_threshold']),
                                 True)
    qualified_box_idx = [qualified_box_idx[i] for i in skinny_filter]

    overlap_filter = find_indices(remove_overlapping_boxes(boxes[qualified_box_idx], scores[qualified_box_idx],
                                                           threshold=params['overlap_threshold']), True)
    qualified_box_idx = [qualified_box_idx[i] for i in overlap_filter]

    if params['display_image'] == True:
        labels = np.array(results[0]['labels'])
        category_bboxes = boxes[qualified_box_idx]
        cat_labels = ['AC Unit', 'AC leaking']
        category = ['AC Unit' for x in range(len(labels[qualified_box_idx]))]
        display_image_with_labels(image, category_bboxes, category)
    return qualified_box_idx


def human_labels_to_df(human_labels, category_idx):
    df = pd.DataFrame(columns=['annot_id', 'category', 'area', 'human_boxes'])
    for cnt in range(len(human_labels['category'])):
        if human_labels['category'][cnt] == category_idx:
            bbox = human_labels['bbox'][cnt]
            bbox = [int(x) for x in [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]
            new_row = pd.DataFrame({'annot_id': [human_labels['id'][cnt]],
                                    'category': [human_labels['category'][cnt]],
                                    'area': [human_labels['area'][cnt]],
                                    'human_boxes': [bbox]})

            df = pd.concat([df, new_row], ignore_index=True)
    if len(df) == 0:
        df = pd.DataFrame({'annot_id': [-1],
                           'category': [category_idx],
                           'area': [0],
                           'human_boxes': [[0, 0, 0, 0]]})
    return df


def auc_scoring(y_actual, y_score):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_actual, y_score)
    auc = auc(fpr, tpr)
    return auc


def f1_scoring(y_actual, y_score):
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import f1_score
    fpr, tpr, thresholds = roc_curve(y_actual, y_score)
    J = tpr - fpr
    # Find the index of the threshold with the greatest Youden's J statistic
    ix = np.argmax(J)
    # Find the optimal threshold
    optimal_threshold = thresholds[ix]
    y_pred_binary = (y_score > optimal_threshold).astype(int)

    f1 = f1_score(y_actual, y_pred_binary)
    return f1
