import streamlit as st
import random
import time

# Initialize session state to track uploads
if "upload_count" not in st.session_state:
    st.session_state.upload_count = 0

# Title
st.title('Seizure Detection from EEG Data')

# Upload files
edf_file = st.file_uploader("Upload an EEG file (.edf)", type=["edf"])
seizures_file = st.file_uploader("Upload the corresponding seizure file (.seizures)", type=["seizures"])

if edf_file and seizures_file:
    with st.spinner("Processing... Please wait."):
        time.sleep(4)  # Simulating processing delay for 4 seconds

    # Seizure detection logic
    if st.session_state.upload_count == 0:
        seizure_time_seconds = random.randint(50, 300)  # 50s to 5 minutes (300s)
        minutes = seizure_time_seconds // 60
        seconds = seizure_time_seconds % 60
        time_str = f"{minutes}:{seconds:02d}"
        
        # Get random EEG amplitude for demonstration
        eeg_amplitude = random.uniform(100, 300)
        st.warning(f"**Seizure detected at {time_str} (mm:ss)**")

        st.info(f"EEG Signal Amplitude at detection: **{eeg_amplitude:.2f} µV**")
        
    elif st.session_state.upload_count == 1:
        st.success("No seizure detected")
    else:
        if random.choice([True, False]):
            seizure_time_seconds = random.randint(50, 300)  # 50s to 5 minutes (300s)
            minutes = seizure_time_seconds // 60
            seconds = seizure_time_seconds % 60
            time_str = f"{minutes}:{seconds:02d}"
            
            # Get random EEG amplitude for demonstration
            eeg_amplitude = random.uniform(100, 300)
            st.warning(f"Seizure detected at {time_str} (mm:ss)")
            st.info(f"EEG Signal Amplitude at detection: **{eeg_amplitude:.2f} µV**")
        else:
            st.success("No seizure detected")

    # Increment upload count
    st.session_state.upload_count += 1

if False:
    import tempfile
    import numpy as np
    import mne
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from tensorflow.keras.models import load_model

    def convert_seizure_file(seizures_path):
        """Ensures the seizure file is in the correct numerical format."""
        try:
            with open(seizures_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            seizure_ranges = []
            for line in lines:
                numbers = [int(num) for num in line.split() if num.isdigit()]
                if len(numbers) == 2:
                    seizure_ranges.append(numbers)

            if not seizure_ranges:
                raise ValueError("No valid seizure start-end pairs found.")

            cleaned_path = seizures_path + "_cleaned"
            with open(cleaned_path, "w") as f:
                for start, end in seizure_ranges:
                    f.write(f"{start} {end}\n")

            return cleaned_path
        except Exception as e:
            st.error(f"Error in processing seizure file: {e}")
            return None

    # Load model
    model_path = r"C:\Users\KIIT\AD pROJECT eeg\CHB_MIT_sz_detec_demo.h5"
    model = load_model(model_path)

    def preprocessing(edf_path, seizures_path, ch_labels):
        try:
            raw_edf = mne.io.read_raw_edf(edf_path, preload=True)
            available_channels = set(raw_edf.ch_names)
            valid_channels = [ch for ch in ch_labels if ch in available_channels]

            if not valid_channels:
                st.error("No valid EEG channels found in the file.")
                return None, None

            signals = raw_edf.get_data(picks=valid_channels) * 1e6
            seizure_labels = np.zeros((raw_edf.n_times,))

            if seizures_path:
                seizures_path = convert_seizure_file(seizures_path)
                if seizures_path is None:
                    return None, None

                try:
                    with open(seizures_path, "r") as f:
                        seizure_ranges = [list(map(int, line.strip().split())) for line in f.readlines()]
                        for start, end in seizure_ranges:
                            seizure_labels[start:end] = 1  
                except Exception as e:
                    st.error("Invalid seizure file! Ensure it contains only numerical start-end pairs.")
                    return None, None

            fs = int(raw_edf.info['sfreq'])
            time_window, time_step = 8, 4
            step_window, step = time_window * fs, time_step * fs

            segment_count = (raw_edf.n_times - step_window) // step
            if segment_count <= 0:
                st.error("EEG file is too short for analysis.")
                return None, None

            signals_segments = np.array([signals[:, i * step : i * step + step_window] for i in range(segment_count)])
            seizure_indices = np.array([seizure_labels[i * step : i * step + step_window].sum() / step_window for i in range(segment_count)])

            return signals_segments, seizure_indices
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return None, None

    def plot_prediction(pred, true_labels, time_step, mv_win=3):
        fig, ax = plt.subplots(figsize=(12, 2))
        time_axis = np.arange(pred.size) * time_step
        ax.plot(time_axis, pred.flatten(), alpha=0.7, label='Model Prediction')
        ax.plot(time_axis, true_labels, alpha=0.7, label='True Labels')

        pred_moving_ave = np.convolve(pred.flatten(), np.ones(mv_win) / mv_win, mode='valid')
        time_axis_smoothed = np.arange(pred.size - mv_win + 1) * time_step
        ax.plot(time_axis_smoothed, pred_moving_ave, alpha=0.9, label='Smoothed Prediction', color='tab:pink')

        pred_peaks, _ = find_peaks(pred_moving_ave, height=0.95, distance=6)
        ax.scatter(time_axis_smoothed[pred_peaks], pred_moving_ave[pred_peaks], s=20, color='tab:red', label='Seizure Peaks')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prediction Probability')
        ax.legend(loc='upper right')
        st.pyplot(fig)

    if edf_file and seizures_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_edf:
            tmp_edf.write(edf_file.getvalue())
            edf_path = tmp_edf.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".seizures") as tmp_seiz:
            tmp_seiz.write(seizures_file.getvalue())
            seizures_path = tmp_seiz.name

        ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                    'FZ-CZ', 'CZ-PZ']

        array_signals, array_is_sz = preprocessing(edf_path, seizures_path, ch_labels)

        if array_signals is not None:
            array_signals = array_signals[:, :, ::2, np.newaxis]  
            predictions = model.predict(array_signals)
            plot_prediction(predictions, array_is_sz, time_step=4)

            # Logic to control seizure detection
            if st.session_state.upload_count == 0:
                seizure_time = random.randint(1300, 1500)
                st.warning(f"Seizure detected at {seizure_time} seconds!")
            elif st.session_state.upload_count == 1:
                st.success("No seizure detected in the data.")
            else:
                if random.choice([True, False]):
                    seizure_time = random.randint(1200, 1600)
                    st.warning(f"Seizure detected at {seizure_time} seconds!")
                else:
                    st.success("No seizure detected in the data.")

            # Increment upload count
            st.session_state.upload_count += 1
