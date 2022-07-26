import fastdeploy as fd

option = fd.RuntimeOption()
option.device = fd.Device.GPU
model = fd.vision.ultralytics.YOLOv5("../models/yolov5s/yolov5s.onnx", runtime_option=option)


result = fd.vision.evaluation.eval_detection(model, 0.01, 0.65, "../data/coco/val2017.txt") 

print(result)
