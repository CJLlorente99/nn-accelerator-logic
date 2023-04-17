from ttUtilities.isfRealization import DNFRealization

nNeurons = 100
filename = '../data/layersTT/2023_04_15_layer1_binaryNN100Epoch100NPLBlackAndWhite'

# Create DNFRealization object
dnf = DNFRealization(100)

# Load tt from feather file

dnf.loadTT(filename)

# Proceed with the realizations
# TODO. Impossible computational wise

# realization = dnf.realizeNeurons()

# Create PLA file for Espresso

dnf.createPLAFileEspresso('../data/espresso')

# Create PLA file for ABC

dnf.createPLAFileABC('../data/ABC')
