import fastdeploy as fd
import cv2
option = fd.RuntimeOption()
option.backend = fd.Backend.TRT
option.device = fd.Device.GPU
option.trt_max_workspace_size = 107374182400
option.trt_min_shape = {"images": [1, 3, 640, 640]}
option.trt_opt_shape = {"images": [4, 3, 640, 640]}
option.trt_max_shape = {"images": [8, 3, 640, 640]}
model=fd.vision.ultralytics.YOLOv5("/huangjianhui/fastdeploy_test/models/yolov5s/yolov5s.onnx", "None", option)
im = cv2.imread("/huangjianhui/fastdeploy_test/data/coco/images/val2017/000000182611.jpg")
result = model.predict(im, 0.01, 0.5)
print(result)
