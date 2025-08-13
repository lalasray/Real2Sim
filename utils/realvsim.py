import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_npz_vs_csv(npz_path, csv_path):
    # Load original simulated data
    sim_data = np.load(npz_path)
    acc_sim = sim_data['accelerometer']  # shape (N, 3)
    gyro_sim = sim_data['gyroscope']     # shape (N, 3)

    # Remove singleton dimensions (1, N, 3) → (N, 3)
    acc_sim = np.squeeze(acc_sim)
    gyro_sim = np.squeeze(gyro_sim)

    # Load generated synthetic data
    synth_df = pd.read_csv(csv_path)
    acc_synth = synth_df[['synth_acc_x', 'synth_acc_y', 'synth_acc_z']].values
    gyro_synth = synth_df[['synth_gyro_x', 'synth_gyro_y', 'synth_gyro_z']].values

    # Make sure lengths match (pad synthetic if needed)
    N = len(acc_sim)
    if len(acc_synth) < N:
        pad_len = N - len(acc_synth)
        acc_synth = np.vstack([acc_synth, np.zeros((pad_len, 3))])
        gyro_synth = np.vstack([gyro_synth, np.zeros((pad_len, 3))])
    elif len(acc_synth) > N:
        acc_synth = acc_synth[:N]
        gyro_synth = gyro_synth[:N]

    t = np.arange(N)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # Accelerometer comparison
    axs[0, 0].plot(t, acc_sim)
    axs[0, 0].set_title("Simulated Accelerometer")
    axs[0, 0].legend(["X", "Y", "Z"])
    axs[0, 0].set_ylabel("m/s²")

    axs[0, 1].plot(t, acc_synth)
    axs[0, 1].set_title("Real2Sim Accelerometer")
    axs[0, 1].legend(["X", "Y", "Z"])

    # Gyroscope comparison
    axs[1, 0].plot(t, gyro_sim)
    axs[1, 0].set_title("Simulated Gyroscope")
    axs[1, 0].legend(["X", "Y", "Z"])
    axs[1, 0].set_ylabel("deg/s")

    axs[1, 1].plot(t, gyro_synth)
    axs[1, 1].set_title("Real2Sim Gyroscope")
    axs[1, 1].legend(["X", "Y", "Z"])

    plt.suptitle("Original Simulated vs Generated Synthetic IMU Data", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    npz_file = "/home/lala/Documents/Data/VQIMU/UTD_MHAD/a25_s8_t4_color/wham_output_right_wrist_imusim.npz"
    csv_file = "/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/real2sim/a25_s8_t4_inertial_synthetic.csv"
    plot_npz_vs_csv(npz_file, csv_file)
