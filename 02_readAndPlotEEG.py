import os
import re
import struct
import numpy as np
import mne
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import csv

# ------------------------------------------------
# Parameters
# ------------------------------------------------

EEG_FILE = "../demo_EEG/demofile.EEG"
CHANNEL_NAMES = [
    "Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
    "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
SFREQ = 256.0
N_BYTES_MARKERS = 8192
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.5
# ------------------------------------------------
# Functions
# ------------------------------------------------
def detect_binary_offset(filename, min_offset=200, max_scan=20000):
    """ Detects where the ASCII header ends and binary data begins """
    with open(filename, "rb") as f:
        data = f.read(max_scan)

    printable = np.array([(32 <= b <= 126 or b in (9, 10, 13)) for b in data])
    byte_values = np.frombuffer(data, dtype=np.uint8)
    window = 1024
    threshold = 0.2
    offset = None

    for i in range(min_offset, len(data) - window, 32):  # 32-Byte Schritte
        win = printable[i:i + window]
        frac_printable = np.mean(win)
        stddev = np.std(byte_values[i:i + window])
        # Text = high ASCII ratio, low varianc
        # Binary data = low ASCII ratio, high variance
        if frac_printable < threshold and stddev > 20:
            offset = i
            break

    if offset is None:
        print("No clear transition detected – use default 1024 bytes")
        offset = 1024

    print(f"\nBinary data probably starting at byte {offset}")
    with open(EEG_FILE, "rb") as f:
        header = f.read(offset)  # read first number of bytes
    return offset

def extract_markers(path, n_bytes=N_BYTES_MARKERS, sfreq=SFREQ):
    """ Extract markers from the last bytes of the EEG file. """
    with open(path, "rb") as f:
        f.seek(-n_bytes, os.SEEK_END)
        tail = f.read()
    marker_re = re.compile(rb"(Augen auf|Augen zu|HV Anfang|HV Ende|IGNORED)")
    events = []
    for m in marker_re.finditer(tail):
        text = m.group(0).decode("latin1")
        start = m.start()
        prefix = tail[start-8:start]
        if len(prefix) >= 4:
            idx = struct.unpack("<I", prefix[0:4])[0]
            onset = idx / sfreq
            events.append((onset, text))
    return events

def markers_to_events(markers, sfreq):
    """ Create MNE-compatible events array. """
    event_id = {}
    events = []
    for onset, desc in markers:
        if desc not in event_id:
            event_id[desc] = len(event_id) + 1
        sample_idx = int(onset * sfreq)
        events.append([sample_idx, 0, event_id[desc]])
    return np.array(events, dtype=int), event_id

def check_for_nans(evoked_dict):
    """ Checks for NaNs in ERP data. """
    valid_evokeds = {}
    for cond, evoked in evoked_dict.items():
        if np.isnan(evoked.data).any():
            print(f"{cond}: contains NaNs – skipped")
        else:
            valid_evokeds[cond] = evoked
    return valid_evokeds

# ------------------------------------------------
# main part
# ------------------------------------------------

def main():
    print(f"loading file: ", EEG_FILE)
    OFFSET = detect_binary_offset(EEG_FILE)
    data = np.fromfile(EEG_FILE, dtype=np.int16, offset=OFFSET)
    print("\nLength data: ",data.size)
    n_channels = len(CHANNEL_NAMES)
    n_samples = data.size // n_channels # rounds down to integer
    data = data[: n_samples * n_channels]
    data = data.reshape(n_samples, n_channels).T # creates a 2D array from data with samples on x- and channels on y-axis

    """ Optional: convert to µV """
    data = data.astype(np.float64) * 0.195  # Example scaling, depending on device
    info = mne.create_info(CHANNEL_NAMES, SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    raw = mne.io.RawArray(data, info)

    """ Extract markers """
    markers = extract_markers(EEG_FILE, sfreq=SFREQ)
    # print(f"\nfound Markers: {len(markers)}")
    # for onset, text in markers:
    #     print(f"{onset:.3f} s → {text}")

    """ Discard markers outside the valid range """
    valid_markers = [(on, tx) for on, tx in markers if 0 <= on <= raw.times[-1]]
    print(f"\nMarker within data range: {len(valid_markers)}")
    for onset, text in valid_markers:
        print(f"{onset:.3f} s → {text}")

    """ Annotations """
    annotations = mne.Annotations(
        onset=[on for on, _ in valid_markers],
        duration=[1.0] * len(valid_markers),
        description=[tx for _, tx in valid_markers]
    )
    raw.set_annotations(annotations)

    raw.plot(n_channels=19, duration=10.0, scalings="auto", block=True)
if __name__ == "__main__":
    main()
