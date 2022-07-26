from util import *
import fastdeploy as fd
import os

class TestResnet50vdTest(object):
    def setup_class(self):
        self.util = FastdeployTest(data_dir_name="imagenet", model_dir_name="resnet50_vd", model_name="resnet50_vd")
        self.util.redirect_err_out()
        self.pdiparams = os.path.join(self.util.model_path, "inference.pdiparams")
        self.pdmodel = os.path.join(self.util.model_path, "inference.pdmodel")
        self.yaml_file = os.path.join(self.util.model_path, "inference_cls.yaml")
        self.image_file_path = self.util.data_path
        #self.label_file_path = os.path.join(self.util.data_path, "val_list_test.txt")
        self.label_file_path = os.path.join(self.util.data_path, "val_list.txt")
        self.option = fd.RuntimeOption()
        
    def teardown_method(self):
        print_log(["stderr.log", "stdout.log"], iden="after predict")
    
    def test_cpu(self):
        self.option.backend = fd.Backend.ORT
        self.option.device = fd.Device.CPU
        model = fd.vision.ppcls.Model(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = {}
        tok1_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=1)
        result["topk1"] = tok1_result
        tok5_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=5)
        result["topk5"] = tok5_result
        check_result(result, self.util.ground_truth)

    def test_gpu(self):
        self.option.backend = fd.Backend.ORT
        self.option.device = fd.Device.GPU
        model = fd.vision.ppcls.Model(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = {}
        tok1_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=1)
        result["topk1"] = tok1_result
        tok5_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=5)
        result["topk5"] = tok5_result
        check_result(result, self.util.ground_truth)

    def test_trt(self):
        self.option.backend = fd.Backend.TRT
        self.option.device = fd.Device.GPU
        self.option.trt_min_shape = {"x": [1, 3, 224, 224]}
        self.option.trt_opt_shape = {"x": [4, 3, 224, 224]}
        self.option.trt_max_shape = {"x": [8, 3, 224, 224]}
        model = fd.vision.ppcls.Model(self.pdmodel, self.pdiparams, self.yaml_file, self.option)
        result = {}
        tok1_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=1)
        result["topk1"] = tok1_result
        tok5_result = fd.vision.evaluation.eval_classify(model, self.image_file_path, self.label_file_path, topk=5)
        result["topk5"] = tok5_result
        check_result(result, self.util.ground_truth)
