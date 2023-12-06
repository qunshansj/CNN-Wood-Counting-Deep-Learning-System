python
class YOLOv5:
    def __init__(self, weights):
        self.weights = weights
        self.model = self.load_model()

    def load_model(self):
        # Load the YOLOv5 model using the provided weights
        model = ...
        return model

    def detect(self, image):
        # Perform object detection on the input image
        detections = ...
        return detections
