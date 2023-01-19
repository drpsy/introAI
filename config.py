image_width=768
image_height=768
target_size=768
gaussian_heatmap_size=1024
start_position_region=0.15
size_of_heatmap_region=1-start_position_region*2
start_position_affinity=0.15
size_of_heatmap_affinity=1-start_position_affinity*2
height_of_box=64.0
# height_of_box=70
expand_small_box=5
batch_size_synthtext=1
batch_size_word=2
epochs_end=200
nb_epochs_change_lr=20
path_saved = "./from_craft_ori"
synth_data = "../vn_syn_data_craft"
expand_scale = 1 # hiep update
word_data = [#"../vn_syn_data_weak",
                # "../../../disk2/hiepnm/craft/data/data_23_5/"
                # "./data_to_craft/train/gcn_qsdd_mt", 
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                "./data_to_craft/train/add_data",
                
                "./data_to_craft/train/gcn_chungnhan",
                "./data_to_craft/train/gcn_nhao_dato"
                # "./data_to_craft/train/gcn_qsdd_mst",
                # "./data_to_craft/train/gcn_qsdd_tsgl_mc",
                # "./data_to_craft/train/gcn_qsdd_tsgl_ms",
                # "./data_to_craft/train/gcn_qsdd_tsgl_mt",
                # "./data_to_craft/train/vietnamese"
                # "../../../disk2/hiepnm/craft/data/Vin_dataset/vietnamese/"
            ]
char_data=[]