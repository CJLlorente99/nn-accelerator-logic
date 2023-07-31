from ttUtilities.isfRealization import DNFRealization
import os

modelName = 'bnn/bnn_prunedBT8_100ep_4096npl'
dataFolder = 'data'
pruned = True
bin = True
prunedBaseFilename = f'{dataFolder}/savedModels/{modelName}_prunedInfol'

'''
Not optimized
'''
print('CREATING NOT OPTIMIZED PLA FILES')
ttFolderName = f'./data/layersTT/{modelName}/notOptimized'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABC'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABC')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSO'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSO')

# Create PLA file for ABC (not Optimized)
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABC', pruned=pruned, prunedBaseFilename=prunedBaseFilename, bin=bin)

# Create PLA file for ESPRESSO (not Optimized)
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSO', pruned=pruned, prunedBaseFilename=prunedBaseFilename, bin=bin)

# '''
# Optimized per entry
# '''
print('CREATING OPTIMIZED PER ENTRY PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_0')
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_1')
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_2')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry', pruned=pruned, prunedBaseFilename=prunedBaseFilename)

# Create PLA file for ESPRESSO
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry', pruned=pruned, prunedBaseFilename=prunedBaseFilename, conflictMode=0)
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry', pruned=pruned, prunedBaseFilename=prunedBaseFilename, conflictMode=1)
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry', pruned=pruned, prunedBaseFilename=prunedBaseFilename, conflictMode=2)

'''
Optimized per class
'''
print('CREATING OPTIMIZED PER CLASS PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/optimizedPerClass'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass')

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass', pruned=pruned, prunedBaseFilename=prunedBaseFilename)

# Create PLA file for ESPRESSO
dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass', pruned=pruned, prunedBaseFilename=prunedBaseFilename)