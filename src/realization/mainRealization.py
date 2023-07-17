from ttUtilities.isfRealization import DNFRealization
import os

modelName = 'eeb_100ep_100npl'
dataFolder = 'data'

'''
Not optimized
'''
print('CREATING NOT OPTIMIZED PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCNotOptimized'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCNotOptimized')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimized'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimized')

# Create PLA file for ABC (not Optimized)
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCNotOptimized')

# Create PLA file for ESPRESSO (not Optimized)
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimized')

'''
Optimized per entry
'''
print('CREATING OPTIMIZED PER ENTRY PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

# Create PLA file for ESPRESSO
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry')

'''
Optimized per class
'''
print('CREATING OPTIMIZED PER CLASS PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCNotOptimizedPerClass'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCNotOptimizedPerClass')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimizedPerClass'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimizedPerClass')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCNotOptimizedPerClass')

# Create PLA file for ESPRESSO
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSONotOptimizedPerClass')