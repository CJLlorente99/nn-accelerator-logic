from ttUtilities.isfRealization import DNFRealization
import os

modelName = 'eeb_100ep_100npl'
dataFolder = 'data'

'''
Not optimized
'''
# print('CREATING NOT OPTIMIZED PLA FILES')
# ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'

# # Create DNFRealization object
# dnf = DNFRealization(ttFolderName)

# if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABC'):
#     os.makedirs(f'{dataFolder}/plas/{modelName}/ABC')

# if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSO'):
#     os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSO')

# # Create PLA file for ABC (not Optimized)
# dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABC')

# # Create PLA file for ESPRESSO (not Optimized)
# dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSO')

'''
Optimized per entry
'''
print('CREATING OPTIMIZED PER ENTRY PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

# if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry'):
#     os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerEntry')

# Create PLA file for ESPRESSO
# dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry')

'''
Optimized per class
'''
print('CREATING OPTIMIZED PER CLASS PLA FILES')
ttFolderName = f'{dataFolder}/layersTT/{modelName}/optimizedPerClass'

# Create DNFRealization object
dnf = DNFRealization(ttFolderName)

if not os.path.exists(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass'):
    os.makedirs(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass')

# if not os.path.exists(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass'):
#     os.makedirs(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass')

# Create PLA file for ABC
dnf.createPLAFileABC(f'{dataFolder}/plas/{modelName}/ABCOptimizedPerClass')

# Create PLA file for ESPRESSO
# dnf.createPLAFileEspresso(f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass')