import yaml
import os

class FastdeployTest(object):
    def __init__(self, data_dir_name: str, model_dir_name: str, model_name: str):
        """
        需设置环境变量
        MODEL_PATH: 模型根目录
        DATA_PATH: 数据集根目录
        py_version: python版本 
        """
        self.py_version = os.environ.get("py_version")
        self.data_path = f"{os.environ.get('DATA_PATH')}/{data_dir_name}/"
        self.model_path = f"{os.environ.get('MODEL_PATH')}/{model_dir_name}/"
        print(model_name)
        self.ground_truth = self.get_ground_truth(model_name)
        

    def get_ground_truth(self, model_name):
        f = open('ground_truth.yaml', 'r', encoding="utf-8")
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data[model_name]

    @staticmethod
    def redirect_err_out(err="stderr.log", out="stdout.log"):
        err = open(err, "w")
        out = open(out, "w")

def check_result(result_data: dict, ground_truth_data: dict, delta=2e-3):
    for key, result_value in result_data.items():
        assert key in ground_truth_data, "The key:{} in result_data is not in the ground_truth_data".format(key)
        ground_truth_val = ground_truth_data[key]
        diff = abs(result_value - ground_truth_val)
        assert diff <= delta, "The diff of {} between result_data and ground_truth_data is {} is bigger than {}".format(key, diff, delta)

def print_log(file_list, iden=""):
    for file in file_list:
        print(f"======================{file} {iden}=====================")
        if os.path.exists(file):
            with open(file, "r") as f:
                print(f.read())
            if file.startswith("log"):
                os.remove(file)
        else:
            print(f"{file} not exist")
        print("======================================================")
