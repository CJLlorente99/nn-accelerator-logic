from ttUtilities.isfRealization import DNFRealization


nNeurons = 200
discriminated = True
filename = '../data/layersTTOptimized/layer0_binary20epoch200NPLBlackAndWhite'

# Create DNFRealization object
dnf = DNFRealization(nNeurons)

# Load tt from feather file

dnf.loadTT(filename)

# Proceed with the realizations
# TODO. Impossible computational wise

# realization = dnf.realizeNeurons()

# Create PLA file for Espresso

# TODO. Change functions so they're not overwritten
dnf.createPLAFileEspresso('../data/espressoOptimized', discriminated=discriminated)

# Create PLA file for ABC

# TODO. Change functions so they're not overwritten
dnf.createPLAFileABC('../data/ABCOptimized', discriminated=discriminated)

# Create file with binary representation of output
# TODO. Impossible computational wise

# dnf.createBinaryOutputRepresentation('../data/outputRepresentation/bin')
