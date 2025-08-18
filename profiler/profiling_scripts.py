import torch
import os
from ptflops import get_model_complexity_info

from compress.export import load_resnet
from torch.profiler import profile, ProfilerActivity, record_function

class ModelProfiler():
    def __init__(self, model, input_res, device):
        self.model = model.to(device)
        self.input_shape = input_res
        self.device = device
    
    def get_theoretical_flops(self, backend):
        """
        Calculate the theoretical FLOPs and parameter count of the model.

        Args:
            backend (str): Backend for FLOPs calculation ('pytorch' or 'aten').

        Returns:
            tuple: (macs, params) where both are strings representing
                multiply-accumulate operations and parameter count.
        """
        macs, params = get_model_complexity_info(self.model, self.input_shape[1:],
                                                as_strings=True,
                                                backend=backend,
                                                print_per_layer_stat=False)
        
        return macs, params
    
    def get_profiler_trace(self, p, save_path):
        """
        Export the profiler's Chrome trace for visualization.

        Args:
            p (torch.profiler.profile): Active profiler instance.
            save_path (str): Directory to save the trace.json file.

        Returns:
            None
        """
        traced_prof_path = os.path.join(save_path, "trace.json")
        p.export_chrome_trace(traced_prof_path)
        print(f"Saved trace to {traced_prof_path}")
    
    def get_runtime_reports(self, profile_memory_flag):
        """
        Profile the model's runtime performance (CPU or CUDA) for a sample input.

        Args:
            profile_memory_flag (bool): If True, also profile memory usage.

        Returns:
            torch.profiler.profile: Profiler instance containing performance data.
        """
        sample_inputs = torch.randn(input_res).to(self.device)
        if self.device == torch.device('cuda'):
            activities = [ProfilerActivity.CUDA]
        else:
            activities = [ProfilerActivity.CPU]
        
        if profile_memory_flag:
            print("Profiling for memory consumption...")
            with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(sample_inputs)
        else:
            print("Profiling for execution speed...")
            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(sample_inputs)
        
        # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            
        return prof
     
if __name__ == "__main__":
    model = load_resnet()
    input_res = (64, 3, 224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    profiler = ModelProfiler(model, input_res, device)
    backend = "pytorch"
    macs, params = profiler.get_theoretical_flops(backend)

    print(f"{'Computational complexity:':<30}  {macs:<8}")
    print(f"{'Number of parameters:':<30}  {params:<8}")

    profile_mem = True
    prof_results = profiler.get_runtime_reports(profile_mem)
    profiler.get_profiler_trace(prof_results, save_path=os.path.join(os.curdir, "profiler"))     