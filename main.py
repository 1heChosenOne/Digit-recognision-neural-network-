from fastapi import FastAPI, UploadFile
from utils import my_neural_net
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO


app=FastAPI()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda device avaliable?:",torch.cuda.is_available())
neuralnet=my_neural_net(input_layer=784,hl_1=128,hl_2=32,output_layer=10).to(device)
neuralnet.load_state_dict(torch.load("neural_net_weights.pth"))



    
transformator=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.Resize((28,28)),
                                  transforms.ToTensor()])

norm_params=torch.load("norm_parameters.pth")
global_mean=norm_params["mean"].to(device)
global_std=norm_params["std"].to(device)

@app.post("/predict")
async def get_predict(image:UploadFile):
    img=Image.open(BytesIO(await image.read()))
    img_tensor=transformator(img).reshape(1,784).to(device)
    
    img_tensor=(img_tensor-(global_mean/255))/(global_std/255)
    with torch.no_grad():
        output=neuralnet(img_tensor)
        probs=torch.softmax(output,dim=1)
        probability,index=torch.max(probs,dim=1)
    return {"message":f"predicted number:{index.item()},probability:{probability.item()}"}
     