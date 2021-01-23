
from utils.vocab import Vocab
from utils.util import initialize_weights, set_seed, count_parameters, save_params, build_vocab, bleu_nltk
import utils.constants as constants
from utils.loss import LabelSmoothing, LossCompute
from utils.optimizer import NoamOpt

from Dataloader import ParallelDataset, get_dataloader
from Translate import translate

from models.Multitask import Multitask
from layers.Encoder import Encoder
from layers.Decoder import Decoder

import torch
import torch.nn as nn

import math
import os
import time
import copy
import nltk

def improved(new_value, best_curr_value, criteria):
    if criteria > 1:
        if new_value > best_curr_value:
            return True
        else:
            return False
    else:
        if new_value < best_curr_value:
            return True
        else:
            return False


def build_dataset(source_files, target_files, batch_size, shuffle=False, \
        source_vocabs=None, target_vocabs=None, mtl=False, max_length=180, lower=False):
    '''
        This method builds a dataset and dataloader for all tasks
        source_files: path for each source file (each file represents a task)
        target_files: path for each target file (each file represents a task)
        batch_size: the size of the batch
        shuffle: shuffle the dataset
        source_vocabs: the source vocabulary for each file
        target_vocabs: the target vocabulary for each file
        mtl: if true an specific target vocabulary is used for each dataset sharing he source vocab, otherwise, all are built separately
        max_length: max length of the source/target lines
        lower: lowercase or not
    '''

    loaders = []

    for index, (source_file, target_file) in enumerate(zip(source_files, target_files)):
        if mtl is True:
            _set = ParallelDataset(source_file, target_file, max_length = max_length, \
                                    source_vocab = source_vocabs[0], target_vocab = target_vocabs[index], lower=lower)
        else:
            _set = ParallelDataset(source_file, target_file, max_length = max_length, \
                                    source_vocab = source_vocabs[0], target_vocab = target_vocabs[0], lower=lower)

        loader = get_dataloader(_set, batch_size, shuffle=shuffle)
        loaders.append(loader)
    return loaders


def load_model(args, source_vocabs, target_vocabs, device, max_length):
    '''
        This method loads a pre-trained model
        args: arguments for loading the model
        source_vocabs: the source vocabulary for each file
        target_vocabs: the target vocabulary for each file
        device: if use gpu or cpu
        max_length: max length of a sentence
    '''
    if args.load_encoder:
        from collections import OrderedDict
        encoder = OrderedDict()
        model = torch.load(args.model)
        for item in model:
            if item.startswith("encoder"):
                encoder[item.replace("encoder.","")] = model[item]
        print("Building an model using a pre-trained encoder ... ")
        current = build_model(args, source_vocabs, target_vocabs, device, max_length, encoder)
        return current
    else:
        mtl = build_model(args, source_vocabs, target_vocabs, device, max_length)
        mtl.load_state_dict(torch.load(args.model))
        print("Building an model using the encoder and the decoder ... ")
        return mtl


def build_model(args, source_vocabs, target_vocabs, device, max_length , encoder=None):
    '''
        This method builds a model from scratch or using the encoder of a pre-trained model
        args: arguments for loading the model
        source_vocabs: the source vocabulary for each file
        target_vocabs: the target vocabulary for each file
        device: if use gpu or cpu
        max_length: max length of a sentence
        encoder: if the encoder is passed as a pre-trained model
    '''

    input_dim = source_vocabs[0].len()
    enc = Encoder(input_dim, 
        args.hidden_size, 
        args.encoder_layer, 
        args.encoder_head, 
        args.encoder_ff_size, 
        args.encoder_dropout, 
        device,
            max_length=max_length).to(device)
    if encoder is None:
        enc.apply(initialize_weights);
    else:
        enc.load_state_dict(encoder)

    decs = []

    for target_vocab in target_vocabs:

        output_dim = target_vocab.len()
        dec = Decoder(output_dim, 
                args.hidden_size, 
                args.decoder_layer, 
                args.decoder_head, 
                args.decoder_ff_size, 
                args.decoder_dropout, 
                device,
        max_length=max_length).to(device)

        if args.tie_embeddings:
            dec.tok_embedding = enc.tok_embedding
            dec.fc_out.weight = enc.tok_embedding.weight

        dec.apply(initialize_weights);
        decs.append(dec)

    model = Multitask(enc, decs, constants.PAD_IDX, constants.PAD_IDX, device).to(device)

    return model

def train_step(model, loader, loss_compute, device, task_id = 0):
    '''
        This method performs training on a step (only one batch)
        model: the model being trained
        loader: dataloader that provides the batches
        loss_compute: function to compute the loss
        device: if use gpu or cpu
        task_id: task id that is being trained (0 as default)
    '''

    model.train()

    (src, tgt) = next(iter(loader))

    src = src.to(device)
    tgt = tgt.to(device)

    output, _ = model(src, tgt[:,:-1], task_id=task_id)        
    #output = [batch size, tgt len - 1, output dim]
    #tgt = [batch size, tgt len]
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    tgt = tgt[:,1:].contiguous().view(-1)
    #output = [batch size * tgt len - 1, output dim]
    #tgt = [batch size * tgt len - 1]

    loss = loss_compute(output, tgt)

    return loss


def evaluate(model, loader, loss_compute, device, task_id=0):
    '''
        This method performs an evaluation on all dataset
        model: the model being evaluated
        loader: dataloader that provides the batches
        loss_compute: function to compute the loss
        device: if use gpu or cpu
        task_id: task id that is being trained (0 as default)
    '''

    model.eval()  
    epoch_loss = 0
    total_tokens = 0
    with torch.no_grad():

        for i, (src, tgt) in enumerate(loader):

            src = src.to(device)
            tgt = tgt.to(device)
            output, _ = model(src, tgt[:,:-1], task_id=task_id)
            #output = [batch size, tgt len - 1, output dim]
            #tgt = [batch size, tgt len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:,1:].contiguous().view(-1)

            #output = [batch size * tgt len - 1, output dim]
            #tgt = [batch size * tgt len - 1]

            loss = loss_compute(output, tgt)
            epoch_loss += loss

        #if torch.equal(model.decoders[task_id].fc_out.weight, model.encoder.tok_embedding.weight):
        #	print("decoder output and encoder embeddings are the same")

    return epoch_loss / len(loader)


def run_translate(model, source_vocab, target_vocabs, save_dir, device, beam_size, filenames, max_length):
    '''
        This method builds a model from scratch or using the encoder of a pre-trained model
        model: the model being evaluated
        source_vocabs: the source vocabulary for each file
        target_vocabs: the target vocabulary for each file
        save_dir: path where the outpus will be saved
        beam_size: beam size during the translating
        filenames: filenames of triples to process
        max_length: max length of a sentence
    '''


    for index, eval_name in enumerate(filenames):
        n = len(eval_name.split("/"))
        name = eval_name.split("/")[n-1]
        print(f'Reading {eval_name}')
        fout = open(save_dir + name + "." + str(index) + ".out", "w")
        with open(eval_name, "r") as f:
            outputs = translate(model, index, f, source_vocab, target_vocabs[index], device, 
                            beam_size=beam_size, max_length=max_length)
            for output in outputs:
                fout.write(output.replace("<eos>","").strip() + "\n")
        fout.close()
        
def run_evaluation(model, source_vocab, target_vocabs, device, beam_size, filenames, ref_files, max_length, criteria):
    '''
        This method builds a model from scratch or using the encoder of a pre-trained model
        model: the model being evaluated
        source_vocabs: the source vocabulary for each file
        target_vocabs: the target vocabulary for each file
        beam_size: beam size during the translating
        filenames: filenames of triples to process
        ref_files: filenames with gold-standards for each process
        max_length: max length of a sentence
        criteria: accuracy or bleu
    '''

    accuracies = []
    bleus = []
    for index, eval_name in enumerate(filenames):
        eval_ref = ref_files[index]
        eval_ref, corpus = '/'.join(eval_ref.split('/')[:-1]), eval_ref.split('/')[-1]

        references = []
        for i, fname in enumerate(sorted(os.listdir(eval_ref))):
            if corpus in fname:
                path = os.path.join(eval_ref, fname)
                with open(path) as f:
                    doc = f.read().split('\n')
                    if i == 0:
                        references = [[w] for w in doc]
                    else:
                        for i, ref in enumerate(doc):
                            references[i].append(ref)

        # references tokenized
        references_tok = copy.copy(references)
        for i, refs in enumerate(references_tok):
            references_tok[i] = [(' '.join(nltk.word_tokenize(ref))).lower() for ref in refs]
        
        n = len(eval_name.split("/"))
        name = eval_name.split("/")[n-1]
        print(f'Reading {eval_name}')
        with open(eval_name, "r") as f:
            outputs = translate(model, index, f, source_vocab, target_vocabs[index], device, 
                            beam_size=beam_size, max_length=max_length)

        if criteria == 2: ## evaluating accuracy
            acc = 0.0
            for j, output in enumerate(outputs):
                if output.replace("<eos>","").strip().lower() in [w.lower() for w in references[j]]:
                    acc += 1
            acc /= len(outputs)
            accuracies.append(acc)            

        if criteria == 3: ## evaluating bleu
            #modifying output (fixing BPE)
            for i in range(len(outputs)):
                newline = ""
                for token in outputs[i].split():
                    if token.endswith("@@"):
                        newline += token.replace("@@","")
                    else:
                        newline += token + " "
                outputs[i] = newline.strip().replace("<eos>", "")
            corpus_bleu = bleu_nltk(references_tok[:len(references_tok)-1], outputs)
            bleus.append(corpus_bleu)

    return accuracies, bleus


def train(args):

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    batch_size = args.batch_size
    max_length = args.max_length
    mtl = args.mtl

    learning_rate = 0.0005
    if not args.learning_rate:
        learning_rate = args.learning_rate

    if len(args.train_source) != len(args.train_target):
        print("Error.Number of inputs in train are not the same")
        return

    if len(args.dev_source) != len(args.dev_target):
        print("Error: Number of inputs in dev are not the same")
        return

    if not args.tie_embeddings:
        print("Building Encoder vocabulary")
        source_vocabs = build_vocab(args.train_source, args.src_vocab, save_dir=args.save_dir, lower=args.lower)
        print("Building Decoder vocabulary")
        target_vocabs = build_vocab(args.train_target, args.tgt_vocab, mtl=mtl, name ="tgt", \
                            save_dir=args.save_dir, lower=args.lower)
    else:
        print("Building Share vocabulary")
        source_vocabs = build_vocab(args.train_source + args.train_target, args.src_vocab, \
                            name="tied", save_dir=args.save_dir, lower=args.lower)
        if mtl:
            target_vocabs = [source_vocabs[0] for _ in range(len(args.train_target))]
        else:
            target_vocabs = source_vocabs
    print("Number of source vocabularies:", len(source_vocabs))
    print("Number of target vocabularies:", len(target_vocabs))

    save_params(args, args.save_dir + "args.json")

    # source_vocabs, target_vocabs = build_vocab(args.train_source, args.train_target, mtl=mtl)

    print("Building training set and dataloaders")
    train_loaders = build_dataset(args.train_source, args.train_target, batch_size, \
            source_vocabs=source_vocabs, target_vocabs=target_vocabs, shuffle=True, mtl=mtl, \
            max_length=max_length, lower=args.lower)
    for train_loader in train_loaders:
        print(f'Train - {len(train_loader):d} batches with size: {batch_size:d}')

    print("Building dev set and dataloaders")
    dev_loaders = build_dataset(args.dev_source, args.dev_target, batch_size, \
            source_vocabs=source_vocabs, target_vocabs=target_vocabs, mtl=mtl, \
            max_length=max_length, lower=args.lower)
    for dev_loader in dev_loaders:
        print(f'Dev - {len(dev_loader):d} batches with size: {batch_size:d}')

    if args.model is not None:
        print("Loading the encoder from an external model...")
        multitask_model = load_model(args, source_vocabs, target_vocabs, device, max_length)
    else:
        print("Building model")
        multitask_model = build_model(args, source_vocabs, target_vocabs, device, max_length)

    print(f'The Transformer has {count_parameters(multitask_model):,} trainable parameters')
    print(f'The Encoder has {count_parameters(multitask_model.encoder):,} trainable parameters')
    for index, decoder in enumerate(multitask_model.decoders):
        print(f'The Decoder {index+1} has {count_parameters(decoder):,} trainable parameters')


    # Defining CrossEntropyLoss as default
    #criterion = nn.CrossEntropyLoss(ignore_index = constants.PAD_IDX)
    criterions = [LabelSmoothing(size=target_vocab.len(), padding_idx=constants.PAD_IDX, smoothing=0.1) \
                                        for target_vocab in target_vocabs]

    # Default optimizer
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr = learning_rate, betas=(0.9, 0.98), eps=1e-09)
    model_opts = [NoamOpt(args.hidden_size, args.warmup_steps, optimizer) for _ in target_vocabs]

    task_id = 0
    print_loss_total = 0  # Reset every print_every

    n_tasks = len(train_loaders)

    if not args.translate:
        print("Start training...")
        patience = args.patience

        print(f'Using {args.early_stopping_criteria} as evaluation criteria')
        early_stopping_criteria = constants.EVALUATION_CRITERIA[args.early_stopping_criteria]
        if early_stopping_criteria == 1:
            best_valid_loss = [float("inf") for _ in range(n_tasks)]
        else:
            best_valid_loss = [float(0) for _ in range(n_tasks)]

        if n_tasks > 1:
            print("Patience wont be taken into account in Multitask learning")

        for _iter in range(1, args.steps + 1):

            train_loss = train_step(multitask_model, train_loaders[task_id], \
                       LossCompute(criterions[task_id], model_opts[task_id]), device, task_id = task_id)

            print_loss_total += train_loss

            if _iter % args.print_every == 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                print(f'Task: {task_id:d} | Step: {_iter:d} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


            if _iter % args.eval_steps == 0:
                print("Evaluating...")
                valid_loss = evaluate(multitask_model, dev_loaders[task_id], LossCompute(criterions[task_id], None), \
                            device, task_id=task_id)
                validation_value = round(math.exp(valid_loss), 3)

                if early_stopping_criteria > 1:
                    accuracies, bleus = run_evaluation(multitask_model, source_vocabs[0], target_vocabs, device, args.beam_size, args.eval, args.eval_ref, max_length, early_stopping_criteria)

                    if early_stopping_criteria == 2:
                        validation_value = round(accuracies[task_id], 3)
                    else:
                        validation_value = round(bleus[task_id], 3)

                print(f'Task: {task_id:d} | Val. Loss: {valid_loss:.3f} |  Val. {args.early_stopping_criteria}: {validation_value:7.3f}')
                if improved(validation_value, best_valid_loss[task_id], early_stopping_criteria):
                    print(f'The {args.early_stopping_criteria} improved from {best_valid_loss[task_id]:.3f} to {validation_value:.3f} in the task {task_id}... saving checkpoint')
                    patience = args.patience
                    best_valid_loss[task_id] = validation_value
                    torch.save(multitask_model.state_dict(), args.save_dir + 'model.pt')
                    print("Saved model.pt")
                else:
                    if n_tasks == 1:
                        if patience == 0:
                            print("The training will stop because it reaches the limit of patience")
                            break
                        else:
                            patience -= 1
                            print(f'Patience ({patience}/{args.patience})')

                if n_tasks > 1:
                    print("Changing to the next task ...")
                    task_id = (0 if task_id == n_tasks - 1 else task_id + 1)

    try:
        multitask_model.load_state_dict(torch.load(args.save_dir + 'model.pt'))
    except:
        print(f'There is no model in the following path {args.save_dir}')
        return

    print("Evaluating and testing")
    run_translate(multitask_model, source_vocabs[0], target_vocabs, args.save_dir, device, args.beam_size, args.eval, max_length=max_length)
    run_translate(multitask_model, source_vocabs[0], target_vocabs, args.save_dir, device, args.beam_size, args.test, max_length=max_length)

