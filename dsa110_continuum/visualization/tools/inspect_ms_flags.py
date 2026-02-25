
import numpy as np
from pathlib import Path

def inspect_flags(ms_path):
    import casatools  # Import inside function per CASA best practices
    
    print(f"Inspecting MS: {ms_path}")
    ms = casatools.ms()
    ms.open(str(ms_path))
    
    try:
        # Get spectral window info
        spw_info = ms.getspectralwindowinfo()
        n_spw = len(spw_info)
        print(f"Found {n_spw} spectral windows.")
        
        for spw_id in range(n_spw):
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=spw_id)
            
            # Read flags and data
            # 'flag' column is shape [n_pol, n_chan, n_row]
            data_dict = ms.getdata(['flag', 'data'])
            flags = data_dict['flag']
            data = data_dict['data']
            
            n_total = flags.size
            n_flagged = np.count_nonzero(flags)
            percent_flagged = (n_flagged / n_total) * 100
            
            # Check for zeros in data (unflagged zeros are bad)
            # We only care about zeros if they are NOT flagged
            unflagged_mask = ~flags
            n_unflagged = np.count_nonzero(unflagged_mask)
            
            if n_unflagged > 0:
                unflagged_data = data[unflagged_mask]
                n_zeros = np.count_nonzero(np.abs(unflagged_data) == 0)
                percent_zeros = (n_zeros / n_unflagged) * 100
            else:
                percent_zeros = 0.0
            
            print(f"SPW {spw_id}: {percent_flagged:.2f}% flagged. Unflagged Zeros: {percent_zeros:.2f}%")
            
    finally:
        ms.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inspect_flags(sys.argv[1])
    else:
        # Default to the path from the user's logs
        ms_path = "/data/jfaber/stage/dsa110-contimg/test_run_verify_transit/science/2026-01-22/2026-01-22T01:25:35_phaseshift_0137+331.ms"
        inspect_flags(ms_path)
