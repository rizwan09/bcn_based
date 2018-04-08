# bcn_based

1) Train ori full classifier: python3 main.py --gpu 2 --model_file_name 'full_ori.pth.tar' (must on margo 2 others donot work, nlp massive performance drop)
2) Train slectors initialized with full_ori.pth.tar classifier: python3 imdb_main.py --gpu 2 --model_file_name model_sparsity_0.00075_coherent_2.0.pth.tar --load_model 1 --classifier_file_name full_ori.pth.tar

 3) For sparsity_list  =  [ 0.5,  0.015, 0.01, 0.00075, 0.0005, 0.0001 ], coherent_list = [2.0] but for [0.00075, 0.0005], [1.0, 2.0], do: save selector outputs: python3 imdb_main.py --gpu 2 --model_file_name dummy_model.pth.tar --load_model 2 --classifier_file_name modle_sparsity_0.0005_coherent_1.0.pth.tar --sparsity 0.0005 --coherent 2.0
 --selector_file_name modle_sparsity_0.0005_coherent_1.0.pth.tar --save_selection 1 --batch_size 8 (NLP 2 checked, margo 2 may be, not others)

 4) Train WAG full classifier from this selection outputs: (only on NLP (specially 2 has been used): For out of memory issue, some twicks has been done in imdb_train_WAG_full_classifier.py and only [sp : 0.015, 0.01][coherent: 2.0], [0.00075, 1.0] and full selection is used. total 4 versions, each with batch size 8, total batch size = 32
 python3 imdb_main_WAG_full_classifier.py --gpu 2 --model_file_name full_WAG_classifier_nlp.pth.tar --load_model -1 --sparsity 0 --coherent 0 --print_every 500 --plot_every 500
 
 
 
 
 Sample code for spped up test; python3 imdb_test.py --gpu 2 --load_model 2 --classifier_file_name modle_sparsity_0.00075_coherent_1.0.pth.tar --selector_file_name modle_sparsity_0.00075_coherent_1.0.pth.tar






