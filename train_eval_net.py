import torch.nn as nn
import torch
import torch.functional as F
from data_load_utils import FENDataset
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import logging
import sys
import os

fmtstr = "%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s"
datestr = "%Y-%m-%d %H:%M:%S"



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5))])


def main(args):
    
    logging.info(f'chess cnn learning run: \nlr = {args.lr}\nepochs = {args.epochs}\nnumber_of_convolutions = {args.number_of_convolutions}\nnumber_of_filters = {args.number_of_filters}\nbatch_size = {args.batch_size}')
    
    model_save_path = f'./models/chess_nc_{args.number_of_convolutions}_nf_{args.number_of_filters}_e_{args.epochs}_bs_{args.batch_size}_lr_{args.lr:.3f}.pth'
    
    #get train loaders
    logging.info('loading datasets...')
    train_loader = get_fen_data_loader(file_path = args.train_file_path, batch_size = args.batch_size)
    test_loader = get_fen_data_loader(file_path = args.test_file_path, batch_size = args.batch_size)
    
    #loss function
    loss_function = nn.MSELoss()
    
    #initialise network
    logging.info('initialising network...')
    model = Model(number_of_convolutions = args.number_of_convolutions, number_of_filters = args.number_of_filters)
    
    logging.info('initialising optimiser...')
    #AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    )
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logging.info(f'Using: {device}')
    
    model = model.to(device)
    
    logging.info('Model training commencing...')
    model = train(model, device, loss_function, optimizer, args.epochs, train_loader, test_loader)
    
    torch.save(model.to(torch.device('cpu')).state_dict(), model_save_path)
    
    return 0


def get_fen_data_loader(file_path: str, batch_size: int, transform = transform, shuffle = True):
    
    fen_dataset = FENDataset(evaluation_file_path=file_path, transform = transform)
    
    fen_loader = DataLoader(fen_dataset, batch_size = batch_size, shuffle = shuffle)
    
    return fen_loader


class Model(nn.Module):
    def __init__(self, number_of_convolutions, number_of_filters): 
        super().__init__() 
        self.number_of_convolutions = number_of_convolutions
        self.number_of_filters = number_of_filters
        
        self.conv1 = nn.Conv2d(16, number_of_filters, kernel_size=3, stride = 1, padding = 'same') # Input Channels, Number of Kernels, Kernel Size 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 1) # Kernel Size, Stride 
        
        self.residual_list = nn.ModuleList([nn.Conv2d(number_of_filters, number_of_filters, kernel_size=3, stride = 1, padding='same') for i in range(number_of_convolutions)])
        
        
        self.fc1 = nn.Linear(number_of_filters*(8+number_of_convolutions+1)**2, 1) 
    
    def forward(self, x): 
        
        x = self.pool(F.relu(self.conv1(x))) 
        
        for conv_layer in self.residual_list:
            x = self.pool(F.relu(conv_layer(x))) 
        
        x = torch.flatten(x, 0) 
        
        x = self.fc1(x)

        return x
    
    
def train(model, device, loss_function, optimizer, epochs, train_loader, test_loader):
    
    for epoch in range(1, epochs + 1):
        logging.info(f'current epoch: {epoch}')
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_tensors = batch[0].to(device)
            b_labels = batch[1].to(device)
            model.zero_grad()

            outputs = model(b_tensors)
            loss = loss_function(outputs[0], b_labels)

            total_loss += loss.item()
            loss.backward()
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            
            if step % 1000  == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.1f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item()
                    )
                )
                
        #run a test at the end of each epoch to allow evaluation of model loss
        model = test(model, test_loader, device)
    
    return model

def test(model, test_loader, device):
    
    logging.info('model testing in progress')
    
    model.eval()
    sum_sq_error_tot = 0

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            b_tensors = batch[0].to(device)
            b_labels = batch[1].to(device)

            outputs = model(b_tensors)
            # loss function goes here. Accuracy will not work
            
            outputs = outputs.detach().cpu().numpy()
            labels = b_labels.to("cpu").numpy()
            
            sum_sq_error = np.sum((outputs-labels)**2)
            
            sum_sq_error_tot += sum_sq_error
            
            
            if step % 1000 == 0:
                logging.info(f'Test step: {step}, Accuracy: {sum_sq_error/len(batch[0])}')

    logging.info(f"Test set: MSE:  {sum_sq_error_tot/len(test_loader.dataset)}")
    
    return model


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--train-file-path',
                            default = './prepared_data/train_chess_data.csv',
                            type = str,
                            help = 'Csv file which has two columns FEN and Evaluation representing the training data for this run'
                           )
    
    arg_parser.add_argument('--test-file-path',
                            default = './prepared_data/test_chess_data.csv',
                            type = str,
                            help = 'Csv file which has two columns FEN and Evaluation representing the test data for this run'
                           )
    
    arg_parser.add_argument('--lr',
                            default = 1e-3,
                            type = float,
                            help = 'The learning rate for the network')
    
    arg_parser.add_argument('--batch-size',
                            default = 64,
                            type = int,
                            help = 'The batch size for the training runs')
    
    arg_parser.add_argument('--epochs',
                            default = 100,
                            type = int,
                            help = 'The number of epochs over which the model will be trained')
    
    arg_parser.add_argument('--number-of-convolutions',
                            default = 5,
                            type = int,
                            help = 'The number of convolution layers for the network')
    
    arg_parser.add_argument('--number-of-filters',
                            default = 10, 
                            type = int,
                            help = 'The number of filters for the convolution layers of the netork')
    
    args = arg_parser.parse_args()
    
    
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
                 
    if not os.path.exists('./models'):
        os.mkdir('./models')
                 
    logging.basicConfig(
        filename=f"./logs/chess_nc_{args.number_of_convolutions}_nf_{args.number_of_filters}_e_{args.epochs}_bs_{args.batch_size}_lr_{args.lr:.3f}.log",
        level=logging.DEBUG,
        filemode="w",
        format=fmtstr,
        datefmt=datestr,
    )
    
    main(args)
    