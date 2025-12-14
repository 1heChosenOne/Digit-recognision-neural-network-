
from torch import nn
from torch.utils.data import DataLoader
from utils import load_mnist_images , load_mnist_y, my_dataset, my_neural_net, my_test_dataset
import time,torch
from torchvision import transforms


path_x="mnist_dataset/train-images.idx3-ubyte"
path_y="mnist_dataset/train-labels.idx1-ubyte"
test_path_x="mnist_dataset/t10k-images.idx3-ubyte"
test_path_y="mnist_dataset/t10k-labels.idx1-ubyte"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda device avaliable?:",torch.cuda.is_available())
# device=torch.device("cpu")

transformator=transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize((28,28)),
                                  transforms.ToTensor()])

ds=my_dataset(load_mnist_images(path_x), load_mnist_y(path_y),transform=transformator)
loader=DataLoader(ds,batch_size=100)

# if never computed global mean or std , uncomment 4 lines below and don't forget about turning off transform above by simply deleting it
# global_mean=torch.tensor(ds.X.mean(),dtype=torch.float32).to(device)
# global_std=(torch.tensor(ds.X.std(),dtype=torch.float32)+1e-8).to(device)
# torch.save({"mean":global_mean,"std":global_std},"norm_parameters.pth")
# print("mean and std have been saved,mean:",global_mean,"std:",global_std)

norm_params=torch.load("norm_parameters.pth") #if there is no file such as that, comment this and 3 lines below and read instruction above
global_mean=norm_params["mean"].to(device)
global_std=norm_params["std"].to(device)
print("loaded mean:",global_mean,"loaded std:",global_std)

test_ds=my_test_dataset(load_mnist_images(test_path_x), load_mnist_y(test_path_y))
test_loader=DataLoader(test_ds,batch_size=100,shuffle=True)


neuralnet=my_neural_net(input_layer=784,hl_1=128,hl_2=32,output_layer=10).to(device)
optimizer=torch.optim.Adam(neuralnet.parameters(),lr=0.001)



criterion=nn.CrossEntropyLoss()

epochs=10
total_steps=len(loader)


neuralnet.load_state_dict(torch.load("neural_net_weights.pth", map_location=device))

start=time.time()

for epoch in range(epochs):
    for i,(x,y) in enumerate(loader):
        x=x.reshape(-1,784).to(device)
        y=y.to(device)
        x_norm=((x)-(global_mean/255))/(global_std/255)
        
        y_hat=neuralnet(x_norm)
        loss=criterion(y_hat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i+1) % (total_steps/10)==0:
            print(f" epochs:{epoch+1}/{epochs}",f"steps:{i+1}/{total_steps}","loss:",loss.item())
        # if ((epoch+1) % (epochs/10) == 0 and (i+1)==total_steps):
        #     print(f" epochs:{epoch+1}/{epochs}",f"steps:{i+1}/{total_steps}","loss:",loss.item())
        

the_time=time.time()-start
print("time to compute weights:",the_time)

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1,784).to(device)
        labels = labels.to(device)
        
        images_norm=((images/255)-(global_mean/255))/(global_std/255)

        outputs = neuralnet(images_norm)
        
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')
    
torch.save(neuralnet.state_dict(),"neural_net_weights.pth")
print("net weights have been saved")