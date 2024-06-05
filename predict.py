from nets.Siamese import Siamese
import utils.utils as utils

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   state_dict_path: The path to your model weights
    #   img_path: The path to the images you want to test
    #   threshold: Threshold to determine forgery or real
    # ----------------------------------------------------#
    state_dict_path = 'logs/model_loss_0.1228.pth'
    img_path_1 = 'images/高荣-60-4.jpg'
    img_path_2 = 'images/蔡洲_forg_2.jpg'
    threshold = 0.1

    # ----------------------------------------------------#
    #   Load the model and put it on GPU
    #   Remember to set "evaluation" mode
    # ----------------------------------------------------#
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Siamese()
    model.load_state_dict(torch.load(state_dict_path))

    model.to(device)
    model.eval()

    # ----------------------------------------------------#
    #   Set up transform -> don't change this
    # ----------------------------------------------------#
    pred_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # ----------------------------------------------------#
    #   Process the images
    # ----------------------------------------------------#
    img1 = Image.open(img_path_1).convert('L')
    img2 = Image.open(img_path_2).convert('L')

    # Apply the transformations
    img1 = pred_transform(img1)
    img2 = pred_transform(img2)

    # Add a batch dimension (model expects batches)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    img1, img2 = img1.to(device), img2.to(device)

    # ----------------------------------------------------#
    #   Start prediction
    # ----------------------------------------------------#
    with torch.no_grad():
        output = model(img1, img2)

    prob = F.sigmoid(output).item()
    print(f'Probability that the signatures are fraud: {prob:.2f}')

    img_demo_1, img_demo_2, label_demo = img1[0].cpu().numpy(), img2[0].cpu().numpy(), int(prob > threshold)

    # ----------------------------------------------------#
    #   Visualize the result
    # ----------------------------------------------------#
    utils.visualize_images(img_demo_1, img_demo_2, label_demo)
