# bcn_based

1) Train ori full classifier: python3 main.py --gpu 2 --model_file_name 'full_ori.pth.tar'
2) Train slectors initialized with full_ori.pth.tar classifier: python3 imdb_main.py --gpu 2 --model_file_name modle_sparsity_0.0125_coherent_2.0.pth.tar --load_model 1 --classifier_file_name full_ori.pth.tar

