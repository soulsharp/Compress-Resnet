import torch
import argparse
import os

from profiler.profiling_scripts import ModelProfiler
from model.resnet_pl import Resnet50Module
from utils.utils import return_train_val_cfg

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Arguments for model profiling')
    parser.add_argument('--config', default='config/config.yaml', type=str, 
                        help="Path containing evaluation config")
    parser.add_argument("--num_classes", default=10, type=int, help="Num_classes in the dataset")
    parser.add_argument("--pretrained_path", default="model/weights/resnet50.pt", type=str,
                        help="Path containing pretrained Resnet50 Model weights")
    parser.add_argument("--prof_trace_save_path", default="logs/profiler", type=str,
                        help="Path to save the profiling trace")
    
    args = parser.parse_args()
    
    train_cfg, _ = return_train_val_cfg(args.config)
    model = Resnet50Module(cfg=train_cfg, num_classes=args.num_classes, pretrained_path=args.pretrained_path)
    input_res = (64, 3, 224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    profiler = ModelProfiler(model, input_res, device)
    backend = "pytorch"
    macs, params = profiler.get_theoretical_flops(backend)

    print(f"{'Computational complexity:':<30}  {macs:<8}")
    print(f"{'Number of parameters:':<30}  {params:<8}")

    profile_mem = True
    prof_results = profiler.get_runtime_reports(profile_mem)
    profiler.get_profiler_trace(prof_results, save_path=args.prof_trace_save_path) 