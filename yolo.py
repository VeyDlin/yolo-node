from ultralyticsplus import YOLO
import torch

from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    ImageField,
    BoundingBoxField
)
from invokeai.app.invocations.primitives import BoundingBoxCollectionOutput


@invocation(
    "yolo",
    title="YOLO",
    tags=["image", "YOLO", "detection"],
    category="image",
    version="1.0.0",
)
class YoloInvocation(BaseInvocation):
    """YOLO (You Only Look Once) is a real-time object detection algorithm that identifies objects and their locations in a single pass"""
    image: ImageField = InputField(default=None, description="Input image")
    model: str = InputField(
        description="The model ID",
        default="ultralyticsplus/yolov8s",
    )
    confidence_threshold: float = InputField(
        description="Confidence threshold for YOLO model. Only bounding boxes with confidence scores above this threshold will be returned",
        ge=0.0,
        le=1.0,
        default=0.25,
    )
    iou_threshold: float = InputField(
        description="Intersection over Union (IoU) threshold for YOLO model. Overlapping bounding boxes with IoU above this threshold will be suppressed",
        ge=0.0,
        le=1.0,
        default=0.45,
    )
    max_detections: int = InputField(
        description="Maximum number of detections for YOLO model. Limits the number of objects detected in a single image",
        ge=1,
        default=300,
    )

    def invoke(self, context: InvocationContext) -> BoundingBoxCollectionOutput:
        if not self.model:
            raise ValueError("The model cannot be empty")
        
        image = context.images.get_pil(self.image.image_name)
        model = YOLO(self.model)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        results = model.predict(image)

        bounding_boxes: list[BoundingBoxField] = []
        boxes = results[0].boxes

        if boxes is not None:
            for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                bounding_box = BoundingBoxField(
                    x_min = int(round(xyxy[0].item())),
                    y_min = int(round(xyxy[1].item())),
                    x_max = int(round(xyxy[2].item())),
                    y_max = int(round(xyxy[3].item())),
                    score=conf.item()
                )
                bounding_boxes.append(bounding_box)

        return BoundingBoxCollectionOutput(collection=bounding_boxes)
