from LMDataset import *
from LMModel import *
from Dataloader import *
from argparse import Namespace
import torch
import torch.optim as optim
import sys
import time
from loadGlove import *


PATH = "savedmodel.tar"



def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)
    
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)

args = Namespace(
                 dataset_csv="data/LM_data.csv",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 reload_from_files=True,
                 expand_filepaths_to_save_dir=True,
                 glove_filepath='data/glove_data/glove.6B.100d.txt', 
                 use_glove=True,
                 cuda=False,
                 seed=1337,
                 learning_rate=5e-4,
                 batch_size=64,
                 num_epochs=100,
                 early_stopping_criteria=5,              
                 source_embedding_size=64, 
                 target_embedding_size=100,
                 decoding_size=64,
                 catch_keyboard_interrupt=True)


def make_train_state(args):
    return {'epoch_index':0,
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[],
            'test_loss':-1,
            'test_acc':-1}
train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# create dataset and vectorizer
dataset = LMDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
vectorizer = dataset.get_vectorizer()

if args.use_glove:
    words = vectorizer.target_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath, 
                                       words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None

model = LMModel(target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 decoding_size=args.decoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 pretrained_embeddings=embeddings)



    
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)
mask_index = vectorizer.target_vocab.mask_index


train_state = make_train_state(args)

                        
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index
    
    dataset.set_split('train')
    batch_generator = generate_LM_batches(dataset , batch_size = args.batch_size , device = args.device)
    running_loss = 0.0
    running_acc = 0.0
    correct = 0
    model.train()
    start = time.time()
    for batch_index,batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = model(batch_dict['x_target'])
        sys.stdout.write("batch number: %d/142   \r" % (batch_index ) )
        sys.stdout.flush()
        # print(y_pred)
        # print(batch_dict['y_target'].float())
        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)
        loss_batch = loss.item()
        running_loss += (loss_batch-running_loss)/(batch_index + 1)
        loss.backward()
        optimizer.step()
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'] , mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        # train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  # epoch=epoch_index)
        # train_bar.update()
    end = time.time()
    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)
    print("Epoch:" + str(epoch_index) + "  " + "running_loss:" + str(round(running_loss , 4)) + "         " + "running_acc:%" + str(round(running_acc , 2))+ "         " + "running_time: " + str(end-start))
    
    
    
    
    dataset.set_split('val')
    batch_generator = generate_LM_batches(dataset , batch_size = args.batch_size , device = args.device)
    running_loss = 0.0
    running_acc = 0.0
    model.eval()
    for batch_index,batch_dict in enumerate(batch_generator):
        # print("batch num: " + str(batch_index))
        y_pred = model(batch_dict['x_target'])
        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)
        loss_batch = loss.item()
        running_loss += (loss_batch-running_loss)/(batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'] , mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
         # train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  # # epoch=epoch_index)
        # train_bar.update()        
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)        
    print("Epoch:" + str(epoch_index) + "  " + "eval_running_loss:" + str(round(running_loss , 4)) + "    " + "eval_running_acc:%" + str(round(running_acc , 2)))
    print("saving checkpoint")
    torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)   


