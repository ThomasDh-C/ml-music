# %% Import libraries and json file
import json

json_file = open('data/chords.json')
pages_to_search = json.load(json_file)

# find all chords we have to find the sounds of
all_chords =set()
for page in pages_to_search:
    for chord_obj in page['chords']:
        all_chords.add(chord_obj['chord'])
print(all_chords)
# {'icon-rest', 'label-A_5', 'label-A_maj', 'label-A_min', 'label-A_sus4', 'label-Ab_maj', 'label-As_maj',
#  'label-B_5', 'label-B_maj', 'label-B_min', 'label-B_min7', 'label-Bb_maj', 'label-Bb_min', 'label-C_5',
#  'label-C_maj', 'label-C_maj7', 'label-C_min', 'label-C_sus4', 'label-Cs_maj', 'label-Cs_min', 'label-D_maj', 
#  'label-D_min', 'label-D_min7', 'label-D_sus4', 'label-Db_maj', 'label-Ds_maj', 'label-Ds_min', 'label-Ds_min7', 
#  'label-E_5', 'label-E_7', 'label-E_maj', 'label-E_min', 'label-E_min7', 'label-E_min9', 'label-Eb_maj', 
#  'label-F_maj', 'label-F_min', 'label-Fs_maj', 'label-Fs_min', 'label-G_5', 'label-G_maj', 'label-G_maj7', 
#  'label-G_min', 'label-G_min7', 'label-G_sus4', 'label-Gb_maj', 'label-Gs_min'}