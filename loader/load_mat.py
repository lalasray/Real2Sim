from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the .mat file
data = loadmat(r'/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial/a1_s1_t1_inertial.mat')

# data is a dictionary with variable names as keys
print(data.keys())
#print(data)

# Access a specific variable, for example 'variable_name'
variable = data['d_iner']
print(variable.shape)

acc = variable[:, 0:3]   # columns 0,1,2
gyro = variable[:, 3:6]  # columns 3,4,5

# Plot accelerometer
plt.figure()
plt.plot(acc)
plt.title('Accelerometer Data')
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')
plt.legend(['Acc X', 'Acc Y', 'Acc Z'])

# Plot gyroscope
plt.figure()
plt.plot(gyro)
plt.title('Gyroscope Data')
plt.xlabel('Sample')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend(['Gyro X', 'Gyro Y', 'Gyro Z'])

plt.show()