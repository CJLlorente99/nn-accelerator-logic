from ttUtilities.isfRealization import DNFRealization


nNeurons = 100
discriminated = True
filename = '../data/layersTTOptimized/'
layerFilename = 'layer0_binary20epoch200NPLBlackAndWhite'

# Create DNFRealization object
dnf = DNFRealization(nNeurons)

# Load tt from feather file

dnf.loadTT(filename + layerFilename)

# Proceed with the realizations
# Impossible computational wise

# realization = dnf.realizeNeurons()

# Create PLA file for Espresso

# TODO. Change functions so they're not overwritten
dnf.createPLAFileEspresso(f'../data/espressoOptimized/{layerFilename}', discriminated=discriminated)

# Create PLA file for ABC

dnf.createPLAFileABC(f'../data/ABCOptimized/{layerFilename}', discriminated=discriminated)

# Create file with binary representation of output
# TODO. Impossible computational wise

# dnf.createBinaryOutputRepresentation('../data/outputRepresentation/bin')
