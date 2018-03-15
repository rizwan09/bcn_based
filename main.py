###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, helper, data, train, os, sys, numpy, torch, pickle, json
from torch import optim
from model import BCN

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
print('{0}\n'.format(args))


###############################################################################
# Load data
###############################################################################

# load train and dev dataset
train_corpus = data.Corpus(args.tokenize)
dev_corpus = data.Corpus(args.tokenize)
test_corpus = data.Corpus(args.tokenize)



args.save_path = args.save_path+args.task+"/"
task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
for task in task_names:
    if 'IMDB' in task:
        ###############################################################################
        # Load Learning to Skim paper's Pickle file
        ###############################################################################
        train_d, dev_d, test_d = helper.get_splited_imdb_data('../IMDB/aclImdb/imdb.p')
        train_corpus.parse(train_d, task, args.max_example)
        dev_corpus.parse(dev_d, task, args.max_example)
        test_corpus.parse(test_d, task, args.max_example)

    elif 'tweet' in task:
        ###############################################################################
        # Load teet data for course 
        ###############################################################################
        train_corpus.parse('../'+args.task+'/train.txt', task, args.max_example)
        dev_corpus.parse('../'+args.task+'/dev.txt', task, args.max_example)
        test_corpus.parse('../'+args.task+'/test.txt', task, args.max_example)

    elif task == 'multinli':
        train_corpus.parse(args.data + task + '/train.txt', task, args.max_example)
        dev_corpus.parse(args.data + task + '/dev_matched.txt', task, args.tokenize)
        test_corpus.parse(args.data + task + '/test_matched.txt', task, args.tokenize)
    else:
        train_corpus.parse(args.data + task + '/train.txt', task, args.max_example)
        dev_corpus.parse(args.data + task + '/dev.txt', task, args.tokenize)
        test_corpus.parse(args.data + task + '/test.txt', task, args.tokenize)


print('train set size = ', len(train_corpus.data))
print('development set size = ', len(dev_corpus.data))
print('test set size = ', len(test_corpus.data))

dictionary = data.Dictionary()
dictionary.build_dict(train_corpus.data + dev_corpus.data + test_corpus.data, args.max_words)
# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')
print('vocabulary size = ', len(dictionary))

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file, dictionary.word2idx)
print('number of OOV words = ', len(dictionary) - len(embeddings_index))

# ###############################################################################
# # Build the model
# ###############################################################################

model = BCN(dictionary, embeddings_index, args)
print(model)
optim_fn, optim_params = helper.get_optimizer(args.optimizer)
optimizer = optim_fn(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
best_acc = 0

param_dict = helper.count_parameters(model)
print('number of trainable parameters = ', numpy.sum(list(param_dict.values())))

# for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if len(cuda_visible_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
# if 'IMDB' in args.task:model = torch.nn.DataParallel(model, device_ids=[0,1,2])
if args.cuda:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

# ###############################################################################
# # Train the model
# ###############################################################################
if args.resume:
    path = args.save_path +'Models/'+ args.resume
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = helper.load_checkpoint(path)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# ###############################################################################
# # Train the model
# ###############################################################################
train = train.Train(model, optimizer, dictionary, args, best_acc)
if 'IMDB' in args.task: last_epoch =8
if not args.no_train:
    last_epoch = train.train_epochs(train_corpus, dev_corpus, test_corpus, args.start_epoch, args.epochs)


# ###############################################################################
# # Test the model
# ###############################################################################
path_of_best_model = args.save_path +'Models/'
if args.no_train and 'epoch' not in args.save_model: path_of_best_model+='epoch_'+str(last_epoch)+'_' 
path_of_best_model += args.save_model 


if os.path.isfile(path_of_best_model):
    print("=> loading saved best model for testing '{}'".format(path_of_best_model))
    checkpoint = helper.load_checkpoint(path_of_best_model)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded saved best model '{}' "
          .format(path_of_best_model))
    test_acc = train.validate(test_corpus)
    print('Test acc = %.2f%%' % test_acc, ' best acc in saved model: ', best_acc)
else:
    print("=> no checkpoint found at '{}'".format(path_of_best_model))

