
import numpy as np

def inspect_amps(ms_path):
    import casatools  # Import inside function per CASA best practices
    
    print(f"Inspecting Amplitudes: {ms_path}")
    ms = casatools.ms()
    ms.open(str(ms_path))
    
    try:
        # Check first spw
        ms.selectinit(datadescid=0)
        data_dict = ms.getdata(['data', 'model_data', 'flag'])
        data = data_dict['data']
        model = data_dict['model_data']
        flags = data_dict['flag']
        
        # Filter valid data
        valid_mask = ~flags
        if np.count_nonzero(valid_mask) == 0:
            print("No valid data found in SPW 0")
            return

        valid_data = data[valid_mask]
        valid_model = model[valid_mask]
        
        data_amp = np.abs(valid_data)
        model_amp = np.abs(valid_model)
        
        print(f"Data Amp: Mean={np.mean(data_amp):.4e}, Median={np.median(data_amp):.4e}, Std={np.std(data_amp):.4e}")
        print(f"Model Amp: Mean={np.mean(model_amp):.4e}, Median={np.median(model_amp):.4e}, Std={np.std(model_amp):.4e}")
        
        # Avoid division by zero
        model_amp_safe = np.where(model_amp == 0, 1e-9, model_amp)
        ratio = data_amp / model_amp_safe
        print(f"Data/Model Ratio: Mean={np.mean(ratio):.4e}, Median={np.median(ratio):.4e}")

    finally:
        ms.close()

if __name__ == "__main__":
    ms_path = "/data/jfaber/stage/dsa110-contimg/test_run_verify_transit/science/2026-01-22/2026-01-22T01:25:35_phaseshift_0137+331.ms"
    inspect_amps(ms_path)
