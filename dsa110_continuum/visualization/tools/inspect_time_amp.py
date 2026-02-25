
import numpy as np
import matplotlib.pyplot as plt

def inspect_time_amp(ms_path):
    import casatools  # Import inside function per CASA best practices
    
    print(f"Inspecting Amplitude vs Time: {ms_path}")
    ms = casatools.ms()
    ms.open(str(ms_path))
    
    try:
        # Select SPW 0 (safest bet)
        ms.selectinit(datadescid=0)
        
        # Read data and time
        data_dict = ms.getdata(['data', 'time', 'axis_info'])
        data = data_dict['data'] # [pol, chan, row]
        time = data_dict['time'] # [row]
        
        # Average across pols and chans for each row to get a single amp per visibility
        # Then bin by time
        
        # Amplitude of raw visibilities [2, 48, N] -> [N]
        amp = np.mean(np.abs(data), axis=(0,1))
        
        # Unique times
        unique_times = np.unique(time)
        print(f"Time span: {len(unique_times)} unique timestamps.")
        print(f"Duration: {unique_times[-1] - unique_times[0]} seconds.")
        
        # Bin averaging
        binned_amps = []
        timestamps = []
        
        for t in unique_times:
            # Mask for this time
            mask = (time == t)
            # Average amplitude for all baselines at this time
            # We want to see the array-wide response peak
            mean_amp = np.mean(amp[mask])
            binned_amps.append(mean_amp)
            timestamps.append(t)
            
        # simple ASCII plot
        amps = np.array(binned_amps)
        t0 = timestamps[0]
        rel_time = np.array(timestamps) - t0
        
        print("\nAmplitude Profile (Time vs Mean Amplitude):")
        # Normalize to 0-1 for plotting
        if np.max(amps) - np.min(amps) == 0:
             print("Flat line (const).")
        else:
            norm_amps = (amps - np.min(amps)) / (np.max(amps) - np.min(amps))
            for t, val, raw in zip(rel_time, norm_amps, amps):
                bar = "#" * int(val * 50)
                print(f"{t:6.1f}s | {bar} ({raw:.4e})")

    finally:
        ms.close()

if __name__ == "__main__":
    ms_path = "/data/jfaber/stage/dsa110-contimg/test_run_verify_transit/science/2026-01-22/2026-01-22T01:25:35_phaseshift_0137+331.ms"
    inspect_time_amp(ms_path)
