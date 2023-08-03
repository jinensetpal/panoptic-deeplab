from typing import Any, List, Mapping, Optional, Tuple, Collection
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from src import const, coco_tools
import tensorflow as tf
import numpy as np
import logging


def get_metrics():
    metrics = [
        tf.keras.metrics.MeanIoU(const.N_CLASSES, 'IoU'),
        PanopticQuality(const.N_CLASSES, 0, const.MAX_INSTANCE_PER_CATEGORY, 0),
        PanopticInstanceAveragePrecision(const.N_CLASSES, const.LABELS, const.PANOPTIC_LABEL_DIVISOR, 0)
    ]
    return metrics


def _ids_to_counts(id_array: np.ndarray) -> Mapping[int, int]:
    """Given a numpy array, a mapping from each unique entry to its count."""
    ids, counts = np.unique(id_array, return_counts=True)
    return dict(zip(ids, counts))


class PanopticInstanceAveragePrecision(tf.keras.metrics.Metric):
    """Computes instance segmentation AP of panoptic segmentations.

    Panoptic segmentation includes both "thing" and "stuff" classes. This class
    ignores the "stuff" classes to report metrics on only the "thing" classes
    that have discrete instances. It computes a series of AP-based metrics using
    the COCO evaluation scripts.
    """

    def __init__(self,
                 num_classes: int,
                 things_list: Collection[int],
                 label_divisor: int,
                 ignored_label: int,
                 name: str = 'panoptic_instance_ap',
                 **kwargs):
        """Constructs panoptic instance segmentation evaluation class."""
        super(PanopticInstanceAveragePrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.stuff_list = set(range(num_classes)).difference(things_list)
        self.label_divisor = label_divisor
        self.ignored_label = ignored_label
        self.detection_metric = InstanceAveragePrecision()
        self.reset_states()

    def reset_states(self) -> None:
        self.detection_metric.reset_states()

    def result(self) -> np.ndarray:
        return self.detection_metric.result()

    def update_state(self,
                     groundtruth_panoptic: tf.Tensor,
                     predicted_panoptic: tf.Tensor,
                     semantic_probability: tf.Tensor,
                     instance_score_map: tf.Tensor,
                     is_crowd_map: Optional[tf.Tensor] = None) -> None:
        """Adds the results from a new image to be computed by the metric.

        Args:
          groundtruth_panoptic: A 2D integer tensor, with the true panoptic label at
            each pixel.
          predicted_panoptic: 2D integer tensor with predicted panoptic labels to be
            evaluated.
          semantic_probability: An float tensor of shape `[image_height,
            image_width, num_classes]`. Specifies at each pixel the estimated
            probability distribution that that pixel belongs to each semantic class.
          instance_score_map: A 2D float tensor, where the pixels for an instance
            will have the probability of that being an instance.
          is_crowd_map: A 2D boolean tensor. Where it is True, the instance in that
        """
        pass


class PanopticQuality(tf.keras.metrics.Metric):
    """Metric class for Panoptic Quality.

    "Panoptic Segmentation" by Alexander Kirillov, Kaiming He, Ross Girshick,
    Carsten Rother, Piotr Dollar.
    https://arxiv.org/abs/1801.00868

    Stand-alone usage:

    pq_obj = panoptic_quality.PanopticQuality(num_classes,
      max_instances_per_category, ignored_label)
    pq_obj.update_state(y_true_1, y_pred_1)
    pq_obj.update_state(y_true_2, y_pred_2)
    ...
    result = pq_obj.result().numpy()
    """

    def __init__(self,
                 num_classes: int,
                 ignored_label: int,
                 max_instances_per_category: int,
                 offset: int,
                 name: str = 'panoptic_quality',
                 **kwargs):
        """Initialization of the PanopticQuality metric.

        Args:
          num_classes: Number of classes in the dataset as an integer.
          ignored_label: The class id to be ignored in evaluation as an integer or
            integer tensor.
          max_instances_per_category: The maximum number of instances for each class
            as an integer or integer tensor.
          offset: The maximum number of unique labels as an integer or integer
            tensor.
          name: An optional variable_scope name. (default: 'panoptic_quality')
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super(PanopticQuality, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignored_label = ignored_label
        self.max_instances_per_category = max_instances_per_category
        self.total_iou = self.add_weight(
            'total_iou', shape=(num_classes,), initializer=tf.zeros_initializer)
        self.total_tp = self.add_weight(
            'total_tp', shape=(num_classes,), initializer=tf.zeros_initializer)
        self.total_fn = self.add_weight(
            'total_fn', shape=(num_classes,), initializer=tf.zeros_initializer)
        self.total_fp = self.add_weight(
            'total_fp', shape=(num_classes,), initializer=tf.zeros_initializer)
        self.offset = offset

    def compare_and_accumulate(
            self, gt_panoptic_label: tf.Tensor, pred_panoptic_label: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compares predicted segmentation with groundtruth, accumulates its metric.

        It is not assumed that instance ids are unique across different categories.
        See for example combine_semantic_and_instance_predictions.py in official
        PanopticAPI evaluation code for issues to consider when fusing category
        and instance labels.

        Instances ids of the ignored category have the meaning that id 0 is "void"
        and remaining ones are crowd instances.

        Args:
          gt_panoptic_label: A tensor that combines label array from categories and
            instances for ground truth.
          pred_panoptic_label: A tensor that combines label array from categories
            and instances for the prediction.

        Returns:
          The value of the metrics (iou, tp, fn, fp) over all comparisons, as a
          float scalar.
        """
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        tp_per_class = np.zeros(self.num_classes, dtype=np.float64)
        fn_per_class = np.zeros(self.num_classes, dtype=np.float64)
        fp_per_class = np.zeros(self.num_classes, dtype=np.float64)

        # Pre-calculate areas for all groundtruth and predicted segments.
        gt_segment_areas = _ids_to_counts(gt_panoptic_label.numpy())
        pred_segment_areas = _ids_to_counts(pred_panoptic_label.numpy())

        # We assume the ignored segment has instance id = 0.
        ignored_panoptic_id = self.ignored_label * self.max_instances_per_category

        # Next, combine the groundtruth and predicted labels. Dividing up the pixels
        # based on which groundtruth segment and which predicted segment they belong
        # to, this will assign a different 64-bit integer label to each choice
        # of (groundtruth segment, predicted segment), encoded as
        #   gt_panoptic_label * offset + pred_panoptic_label.
        intersection_id_array = tf.cast(gt_panoptic_label,
                                        tf.int64) * self.offset + tf.cast(
            pred_panoptic_label, tf.int64)

        # For every combination of (groundtruth segment, predicted segment) with a
        # non-empty intersection, this counts the number of pixels in that
        # intersection.
        intersection_areas = _ids_to_counts(intersection_id_array.numpy())

        # Compute overall ignored overlap.
        def prediction_ignored_overlap(pred_panoptic_label):
            intersection_id = ignored_panoptic_id * self.offset + pred_panoptic_label
            return intersection_areas.get(intersection_id, 0)

        # Sets that are populated with which segments groundtruth/predicted segments
        # have been matched with overlapping predicted/groundtruth segments
        # respectively.
        gt_matched = set()
        pred_matched = set()

        # Calculate IoU per pair of intersecting segments of the same category.
        for intersection_id, intersection_area in intersection_areas.items():
            gt_panoptic_label = intersection_id // self.offset
            pred_panoptic_label = intersection_id % self.offset

            gt_category = gt_panoptic_label // self.max_instances_per_category
            pred_category = pred_panoptic_label // self.max_instances_per_category
            if gt_category != pred_category:
                continue
            if pred_category == self.ignored_label:
                continue

            # Union between the groundtruth and predicted segments being compared does
            # not include the portion of the predicted segment that consists of
            # groundtruth "void" pixels.
            union = (
                    gt_segment_areas[gt_panoptic_label] +
                    pred_segment_areas[pred_panoptic_label] - intersection_area -
                    prediction_ignored_overlap(pred_panoptic_label))
            iou = intersection_area / union
            if iou > 0.5:
                tp_per_class[gt_category] += 1
                iou_per_class[gt_category] += iou
                gt_matched.add(gt_panoptic_label)
                pred_matched.add(pred_panoptic_label)

        # Count false negatives for each category.
        for gt_panoptic_label in gt_segment_areas:
            if gt_panoptic_label in gt_matched:
                continue
            category = gt_panoptic_label // self.max_instances_per_category
            # Failing to detect a void segment is not a false negative.
            if category == self.ignored_label:
                continue
            fn_per_class[category] += 1

        # Count false positives for each category.
        for pred_panoptic_label in pred_segment_areas:
            if pred_panoptic_label in pred_matched:
                continue
            # A false positive is not penalized if is mostly ignored in the
            # groundtruth.
            if (prediction_ignored_overlap(pred_panoptic_label) / pred_segment_areas[pred_panoptic_label]) > 0.5:
                continue
            category = pred_panoptic_label // self.max_instances_per_category
            if category == self.ignored_label:
                continue
            fp_per_class[category] += 1
        return iou_per_class, tp_per_class, fn_per_class, fp_per_class

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> List[tf.Operation]:
        """Accumulates the panoptic quality statistics.

        Args:
          y_true: The ground truth panoptic label map (defined as semantic_map *
            max_instances_per_category + instance_map).
          y_pred: The predicted panoptic label map (defined as semantic_map *
            max_instances_per_category + instance_map).
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update ops for iou, tp, fn, fp.
        """
        result = self.compare_and_accumulate(y_true, y_pred)
        iou, tp, fn, fp = tuple(result)
        update_iou_op = self.total_iou.assign_add(iou)
        update_tp_op = self.total_tp.assign_add(tp)
        update_fn_op = self.total_fn.assign_add(fn)
        update_fp_op = self.total_fp.assign_add(fp)
        return [update_iou_op, update_tp_op, update_fn_op, update_fp_op]

    def result(self) -> tf.Tensor:
        """Computes the panoptic quality."""
        sq = tf.math.divide_no_nan(self.total_iou, self.total_tp)
        rq = tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + 0.5 * self.total_fn + 0.5 * self.total_fp)
        pq = tf.math.multiply(sq, rq)

        # Find the valid classes that will be used for evaluation. We will
        # ignore classes which have (tp + fn + fp) equal to 0.
        # The "ignore" label will be included in this based on logic that skips
        # counting those instances/regions.
        valid_classes = tf.not_equal(self.total_tp + self.total_fn + self.total_fp,
                                     0)

        # Compute averages over classes.
        qualities = tf.stack(
            [pq, sq, rq, self.total_tp, self.total_fn, self.total_fp], axis=0)
        summarized_qualities = tf.math.reduce_mean(
            tf.boolean_mask(qualities, valid_classes, axis=1), axis=1)

        return summarized_qualities

    def reset_states(self) -> None:
        """See base class."""
        tf.keras.backend.set_value(self.total_iou, np.zeros(self.num_classes))
        tf.keras.backend.set_value(self.total_tp, np.zeros(self.num_classes))
        tf.keras.backend.set_value(self.total_fn, np.zeros(self.num_classes))
        tf.keras.backend.set_value(self.total_fp, np.zeros(self.num_classes))

    def get_config(self) -> Mapping[str, Any]:
        """See base class."""
        config = {
            'num_classes': self.num_classes,
            'ignored_label': self.ignored_label,
            'max_instances_per_category': self.max_instances_per_category,
            'offset': self.offset,
        }
        base_config = super(PanopticQuality, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InstanceAveragePrecision(tf.keras.metrics.Metric):
    """COCO evaluation metric class."""

    def __init__(self, name: str = 'instance_ap', **kwargs):
        """Constructs COCO evaluation class."""
        super(InstanceAveragePrecision, self).__init__(name=name, **kwargs)
        self.reset_states()

    def reset_states(self) -> None:
        """Reset COCO API object."""
        self.detections = []
        self.dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        self.image_id = 1
        self.next_groundtruth_annotation_id = 1
        self.category_ids = set()
        self.metric_values = None

    def evaluate(self) -> np.ndarray:
        """Evaluates with detections from all images with COCO API.

        Returns:
          coco_metric: float numpy array with shape [12] representing the
            coco-style evaluation metrics.
        """
        if not self.detections:
            logging.warn('No detections to evaluate.')
            return np.zeros([12], dtype=np.float32)

        self.dataset['categories'] = [{
            'id': int(category_id)
        } for category_id in self.category_ids]

        # Creates "unwrapped" copies of COCO json-style objects.
        dataset = {
            'images': self.dataset['images'],
            'categories': self.dataset['categories']
        }
        dataset['annotations'] = [
            _unwrap_annotation(ann) for ann in self.dataset['annotations']
        ]
        detections = [_unwrap_annotation(ann) for ann in self.detections]

        logging.info('Creating COCO objects for AP eval...')
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(detections)

        logging.info('Running COCO evaluation...')
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        return np.array(coco_metrics, dtype=np.float32)

    def result(self) -> np.ndarray:
        """Return the instance segmentation metric values, computing them if needed.

        Returns:
          A float vector of 12 elements. The meaning of each element is (in order):

           0. AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
           1. AP @[ IoU=0.50      | area=   all | maxDets=100 ]
           2. AP @[ IoU=0.75      | area=   all | maxDets=100 ]
           3. AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
           4. AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
           5. AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
           6. AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
           7. AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
           8. AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
           9. AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
          10. AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
          11, AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]

          Where: AP = Average Precision
                 AR = Average Recall
                 IoU = Intersection over Union. IoU=0.50:0.95 is the average of the
                   metric over thresholds of 0.5 to 0.95 with increments of 0.05.

          The area thresholds mean that, for those entries, ground truth annotation
          with area outside the range is ignored.
            small:  [0**2, 32**2],
            medium: [32**2, 96**2]
            large:  [96**2, 1e5**2]
        """
        if not self.metric_values:
            self.metric_values = self.evaluate()
        return self.metric_values

    def update_state(self, groundtruth_boxes: tf.Tensor,
                     groundtruth_classes: tf.Tensor, groundtruth_masks: tf.Tensor,
                     groundtruth_is_crowd: tf.Tensor, detection_masks: tf.Tensor,
                     detection_scores: tf.Tensor,
                     detection_classes: tf.Tensor) -> None:
        """Update detection results and groundtruth data.

        Append detection results to self.detections to the aggregate results from
        all of the validation set. The groundtruth_data is parsed and added into a
        dictionary with the same format as COCO dataset, which can be used for
        evaluation.

        Args:
          groundtruth_boxes: tensor (float32) with shape [num_gt_annos, 4]
          groundtruth_classes: tensor (int) with shape [num_gt_annos]
          groundtruth_masks: tensor (uint8) with shape [num_gt_annos, image_height,
            image_width]
          groundtruth_is_crowd: tensor (bool) with shape [num_gt_annos]
          detection_masks: tensor (uint8) with shape [num_detections, image_height,
            image_width]
          detection_scores: tensor (float32) with shape [num_detections]
          detection_classes: tensor (int) with shape [num_detections]
        """
        # Reset the caching of result values.
        self.metric_values = None

        # Update known category ids.
        self.category_ids.update(groundtruth_classes.numpy())
        self.category_ids.update(detection_classes.numpy())

        # Add ground-truth annotations.
        groundtruth_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
            self.image_id,
            self.next_groundtruth_annotation_id,
            self.category_ids,
            groundtruth_boxes.numpy(),
            groundtruth_classes.numpy(),
            groundtruth_masks=groundtruth_masks.numpy(),
            groundtruth_is_crowd=groundtruth_is_crowd.numpy())
        self.next_groundtruth_annotation_id += len(groundtruth_annotations)

        # Add to set of images for which there are gt & detections
        # Infers image size from groundtruth masks.
        _, height, width = groundtruth_masks.shape
        self.dataset['images'].append({
            'id': self.image_id,
            'height': height,
            'width': width,
        })
        self.dataset['annotations'].extend(groundtruth_annotations)

        # Add predictions/detections.
        detection_annotations = coco_tools.ExportSingleImageDetectionMasksToCoco(
            self.image_id, self.category_ids, detection_masks.numpy(),
            detection_scores.numpy(), detection_classes.numpy())
        self.detections.extend(detection_annotations)

        self.image_id += 1


def _instance_masks(panoptic_label_map: tf.Tensor,
                    instance_panoptic_labels: tf.Tensor) -> tf.Tensor:
    """Constructs an array of masks for each instance in a panoptic label map.

    Args:
      panoptic_label_map: An integer tensor of shape `[image_height, image_width]`
        specifying the panoptic label at each pixel.
      instance_panoptic_labels: An integer tensor of shape `[num_instances]` that
        gives the label for each unique instance for which to compute masks.

    Returns:
      A boolean tensor of shape `[num_instances, image_height, image_width]` where
      each slice in the first dimension gives the mask for a single instance over
      the entire image.
    """
    return tf.math.equal(
        tf.expand_dims(panoptic_label_map, 0),
        tf.reshape(instance_panoptic_labels,
                   [tf.size(instance_panoptic_labels), 1, 1]))


def _unwrap_segmentation(seg):
    return {'size': list(seg['size']),
            'counts': seg['counts']}


_ANNOTATION_CONVERSION = {'bbox': list,
                          'segmentation': _unwrap_segmentation}


def _unwrap_annotation(ann: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwraps the objects in an COCO-style annotation dictionary.

    Logic within the Keras metric class wraps the objects within the ground-truth
    and detection annotations in ListWrapper and DictWrapper classes. On the other
    hand, the COCO API does strict type checking as part of determining which
    branch to use in comparing detections and segmentations. We therefore have
    to coerce the types from the wrapper to the built-in types that COCO is
    expecting.

    Args:
      ann: A COCO-style annotation dictionary that may contain ListWrapper and
        DictWrapper objects.

    Returns:
      The same annotation information, but with wrappers reduced to built-in
      types.
    """
    unwrapped_ann = {}
    for k in ann:
        if k in _ANNOTATION_CONVERSION: unwrapped_ann[k] = _ANNOTATION_CONVERSION[k](ann[k])
        else: unwrapped_ann[k] = ann[k]
    return unwrapped_ann
