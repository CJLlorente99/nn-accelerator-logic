from ttUtilities.isfRealization import DNFRealization

modelName = 'eeb_100ep_100npl'
dataFolder = 'data'

'''
Not optimized
'''
print('CREATING NOT OPTIMIZED PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/notOptimized/{modelName}'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

# Create PLA file for ABC (not Optimized)
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCNotOptimized')

'''
Optimized per entry
'''
print('CREATING OPTIMIZED PER ENTRY PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/optimizedPerEntry/{modelName}'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

# Create PLA file for ABC (not Optimized)
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

'''
Optimized per class
'''
print('CREATING OPTIMIZED PER CLASS PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/notOptimized/{modelName}'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

# Create PLA file for ABC (not Optimized)
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCNotOptimizedPerClass')