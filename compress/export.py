from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights

def load_resnet():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    print(summary(model,
                  input_size=(1, 3, 224, 224),
                  col_names=["output_size", "num_params", "params_percent"]))
    return model

if __name__ == "__main__":
    resnet_model = load_resnet()