import matplotlib.pyplot as plt

def plot_sample_signals(sample, window_idx=0):
    """
    Plot simulated and real accelerometer and gyroscope signals for one window.
    
    Args:
        sample (dict): a single sample dict from the dataset or dataloader batch.
        window_idx (int): which window to plot if sample contains multiple windows (default 0).
    """

    # Extract signals, handle batch dim if present
    def get_signal(tensor):
        # If tensor has 3 dims (batch, time, channels), select window_idx
        if tensor.dim() == 3:
            return tensor[window_idx].cpu().numpy()
        else:
            return tensor.cpu().numpy()

    acc_sim = get_signal(sample['acc_sim'])
    gyro_sim = get_signal(sample['gyro_sim'])

    has_real = ('acc_real' in sample) and ('gyro_real' in sample)
    if has_real:
        acc_real = get_signal(sample['acc_real'])
        gyro_real = get_signal(sample['gyro_real'])

    time_sim = range(acc_sim.shape[0])
    if has_real:
        time_real = range(acc_real.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    # Simulated Acc
    axs[0, 0].plot(time_sim, acc_sim[:, 0], label='X')
    axs[0, 0].plot(time_sim, acc_sim[:, 1], label='Y')
    axs[0, 0].plot(time_sim, acc_sim[:, 2], label='Z')
    axs[0, 0].set_title('Simulated Accelerometer')
    axs[0, 0].legend()

    # Simulated Gyro
    axs[0, 1].plot(time_sim, gyro_sim[:, 0], label='X')
    axs[0, 1].plot(time_sim, gyro_sim[:, 1], label='Y')
    axs[0, 1].plot(time_sim, gyro_sim[:, 2], label='Z')
    axs[0, 1].set_title('Simulated Gyroscope')
    axs[0, 1].legend()

    if has_real:
        # Real Acc
        axs[1, 0].plot(time_real, acc_real[:, 0], label='X')
        axs[1, 0].plot(time_real, acc_real[:, 1], label='Y')
        axs[1, 0].plot(time_real, acc_real[:, 2], label='Z')
        axs[1, 0].set_title('Real Accelerometer')
        axs[1, 0].legend()

        # Real Gyro
        axs[1, 1].plot(time_real, gyro_real[:, 0], label='X')
        axs[1, 1].plot(time_real, gyro_real[:, 1], label='Y')
        axs[1, 1].plot(time_real, gyro_real[:, 2], label='Z')
        axs[1, 1].set_title('Real Gyroscope')
        axs[1, 1].legend()
    else:
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
