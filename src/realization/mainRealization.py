from ttUtilities.isfRealization import DNFRealization

# ! WARNING! There are different nNeurons depending on the layer
nNeurons = 100
discriminated = False
filename = './data/layersTT/'
layerFilename = 'layer3_MNISTSignbinNN100Epoch100NPLnllCriterion'

# Create DNFRealization object
dnf = DNFRealization(nNeurons)

# Load tt from feather file

dnf.loadTT(filename + layerFilename)

# Create PLA file for Espresso

dnf.createPLAFileEspresso(f'./data/espresso/{layerFilename}', discriminated=discriminated)

# Create PLA file for Espresso (all together)

dnf.createPLAFileEspresso(f'./data/espresso/grouped/{layerFilename}', discriminated=discriminated, joinOutput=True)

# Create PLA file for ABC

# dnf.createPLAFileABC(f'./data/ABC/{layerFilename}', discriminated=discriminated)

# Create PLA file for ABC (all together)

dnf.createPLAFileABC(f'./data/ABC/grouped/{layerFilename}', discriminated=discriminated, joinOutput=True)
