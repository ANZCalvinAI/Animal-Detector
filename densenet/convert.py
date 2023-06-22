import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
 
 
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
 
    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的, this is orignally for detection.
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})

        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores 
 
 
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
 

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load("./models/model-0.pth") # pytorch模型加载
    batch_size = 1  #批处理大小
    input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape
    
    # #set the model to inference mode
    model.eval()
    
    x = torch.randn(batch_size, *input_shape)   # 生成张量
    x = x.to(device)

    if os.path.exists("./onnx") is False:
        os.makedirs("./onnx")

    export_onnx_file = os.path.join("./onnx", "densenet161.onnx")
    torch.onnx.export(model,
        x,
        export_onnx_file,
        opset_version=10,
        do_constant_folding=True,   # 是否执行常量折叠优化
        input_names=["input"],      # 输入名
        output_names=["output"],    # 输出名
        dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                        "output":{0:"batch_size"}})
    
    r_model_path=export_onnx_file
     
    img = cv2.imread("./dog.jpg")
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
     
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img = img.unsqueeze_(0)
     
    rnet1 = ONNXModel(r_model_path)
    out = rnet1.forward(to_numpy(img))
    print(out)
    
