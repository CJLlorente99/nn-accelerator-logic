from ttUtilities.isfRealization import DNFRealization

modelName = 'eeb_100ep_100npl'
ttFolderName = f'./data/layersTT/{modelName}'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

# Create PLA file for Espresso

# dnf.createPLAFileEspresso(f'./data/plas/{modelName}/ESPRESSO')

# Create PLA file for ABC

dnf.createPLAFileABC(f'./data/plas/{modelName}/ABC')
