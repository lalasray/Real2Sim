from scipy.io import loadmat

# Load the .mat file
data = loadmat(r'/home/lala/Documents/Data/VQIMU/UTD_MHAD_Inertial/Inertial/a1_s1_t1_inertial.mat')

# data is a dictionary with variable names as keys
print(data.keys())
#print(data)

# Access a specific variable, for example 'variable_name'
variable = data['d_iner']
print(variable.shape)
