from util import *
import fastdeploy as fd
import os

class TestYolov5Test(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="coco", model_dir_name="yolov5s", model_name="yolov5s")
        self.util.redirect_err_out()
        self.onnxmodel = os.path.join(self.util.model_path, "yolov5s.onnx")
        self.image_file_path = self.util.data_path
        self.label_file_path = os.path.join(self.util.data_path, "val2017.txt")
        self.option = fd.RuntimeOption()
        
    def teardown_method(self):
        print_log(["stderr.log", "stdout.log"], iden="after predict")
    
    def test_cpu(self):
        self.option.backend = fd.Backend.ORT
        self.option.device = fd.Device.CPU
        model = fd.vision.ultralytics.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, 0.001, 0.65, self.label_file_path)
        check_result(result, self.util.ground_truth)

    def test_gpu(self):
        self.option.backend = fd.Backend.ORT
        self.option.device = fd.Device.GPU
        model = fd.vision.ultralytics.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, 0.001, 0.65, self.label_file_path)
        check_result(result, self.util.ground_truth)


    def test_trt(self):
        self.option.backend = fd.Backend.TRT
        self.option.device = fd.Device.GPU
        self.option.trt_max_workspace_size = 107374182400
        self.option.trt_min_shape = {"images": [1, 3, 640, 640]}
        self.option.trt_opt_shape = {"images": [4, 3, 640, 640]}
        self.option.trt_max_shape = {"images": [8, 3, 640, 640]}
        model = fd.vision.ultralytics.YOLOv5(self.onnxmodel, "None", self.option)
        result = fd.vision.evaluation.eval_detection(model, 0.001, 0.65, self.label_file_path)
        check_result(result, self.util.ground_truth)
