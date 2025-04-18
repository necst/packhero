# The script check if the packer is correct.

excluded_files = ['die_results.txt','upx_a81362b3427e0b8061a937245cff720c32e19a0f.upx.bin', 'upx_9bbfb7e070238e88107b8335190500b6d29bc80d.upx.bin', 'themida-v2_d756de22a8172c1560299ef3146e6c060fba210e.themida-v2.bin', 'themida-v2_b47c2247999b3d45297803313b598b17b544576f.themida-v2.bin', 'telock_7a3a873e5e940c04d11806d2bb874b03d140206f.telock.bin', 'telock_f9fb1623e84b55df84e987f28914756371a5ae37.telock.bin', 'upx_bb7b1bd42a3282a075a0ff61118f4cd0185c9434.upx.bin', 'petite_30058ff28fa1b96e9354285dd2a9eaa617d2d822.petite.bin', 'pecompact_a63e34ec6e7610352cce4cc2eea3ea164eb27bfc.pecompact.bin', 'telock_90dcdae6e283bc401f61ee75c7052ec80a713532.telock.bin']

packer_for_file = {}
filename = ""

def parse_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('/run'):
                filename = line.split('/')[-1][:-2]
                packer_for_file[filename] = []
            
            if 'Packer' in line or 'Protector' in line:
                if '(' in line:
                    packer = line.split(': ')[-1].split('(')[0]
                else:
                    packer = line.split(': ')[-1][:-1]
                if '/' in line:
                    packer = packer.split('/')[0]
                if '[' in line:
                    packer = packer.split('[')[0]

                if packer.lower() == 'themida':
                    packer = 'themida-v2'
                
                packer_for_file[filename].append(packer.lower())

parse_file("die_results.txt")

PACKERS = ['kkrunchy', 'mpress', 'obsidium', 'pecompact', 'pelock', 'petite', 'telock', 'themida-v2', 'upx']

recall = {packer: [] for packer in PACKERS}
fpr = {packer: [] for packer in PACKERS}

for file in packer_for_file:
    if file not in excluded_files:
        if file.split('_')[0] in packer_for_file[file]:
            recall[file.split('_')[0]].append(1)
        else:
            recall[file.split('_')[0]].append(0)

print("Recall")
for packer in recall:
    print(packer, round(sum(recall[packer])/len(recall[packer]), 2))

# evaluate if there are false positives
for file in packer_for_file:
    if file in excluded_files:
        continue

    if file.split('_')[0] not in packer_for_file[file]:
        for packer in packer_for_file[file]:
            if packer in PACKERS:
                fpr[packer].append(1)

print()
print("FPR")
n_files = len(packer_for_file) - len(excluded_files)
for packer in fpr:
    print(packer, round(sum(fpr[packer])/n_files, 4))
    
