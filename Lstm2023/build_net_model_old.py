#coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import copy
from random import randint, sample
import pandas as pd
from sklearn.decomposition import PCA


device = torch.device('cpu')

params = {
    # full mode [800,900,1000,1100,1200]
    # full mode [1,2,3,4,5,6,7,8,9]
    'hidden_size':[800],
    'nof_replaced_words':[0,1,2,3,4,5,6,7,8,9],#
    'run_mode':1
}


def gen_trainandtest(nof_replaced_words,sample_amount=2,nof_replaced_words_list=[]):
    sentence_lenghth = 10 #筛选出句子的长度
    embedding_dim = 32 #词嵌入的维度 这里按照对比试验的32设置
    sentance_number = 1000 #总句子数目
    # 从文件读取Raw Data
    #fo = open("/Users/zhangjilun/Library/Mobile Documents/com~apple~CloudDocs/WorkingFolder/codes/Lstm_epmomo/new_cbt_0307.txt", "r")
    fo = open("/Users/sanfeng/Desktop/Lstm2023/cbt_10_words_lstm.txt","r")
    lines = fo.readlines()
    fo.close()
    #只筛选长度为10的句子
    raw_data = lines
    ten_words_data = []
    #print('满足句长要求的句子')
    for l in raw_data :
        l = l.split(' ')
        l = l[:-1]
        if len(l) == sentence_lenghth :
            if len(ten_words_data) < sentance_number :
                ten_words_data.append(l)
                #print(l)
        else:
            pass

    #print(ten_words_data[:4])

    #计算词典，词嵌入
    vocab_dict = {}
    word_idx = 0
    for l in ten_words_data :
        for word in l :
            if word not in vocab_dict.keys():
                vocab_dict[word] = word_idx
                word_idx = word_idx + 1
    vocab_len = len(vocab_dict)
    print('Genarate one-hot dict total words '+str(vocab_len))


    embeds = nn.Embedding(vocab_len,embedding_dim)
    embeds_sent_list = []
    #为训练数据做标签，模拟一个1000分类任务，每个标签用一个1x1000的正交单位向量表示
    embads_sent_label = []
    embads_word_dic = {} #存储单词的单个embadding词向量
    test_embads_sent_list = [] 
    test_embads_corr_right_sent_list = []
    test_embads_label_list = []
    test_embads_sent_replace_count_list = []
    diverse = nof_rp_wds #替换词个数
    per_sample_number = sample_amount #每个训练数据生成的测试数据
    
    
    '''
    0411晚上的修改
    前期版本每一次只针对特定的替换词数生成测试数据，现在修改成能够一次生成基于embads的所有替换次数的数据集，存入字典中
    '''
    
    g_embads_sent_list_dict = {} #为基于模型的全局测试数据集构建提供保障的词典
    g_embads_sent_label_dict = {} #ditto
    

    for i in range(1000) :
        '''
        embads_sent_label.append([0]*1000)
        embads_sent_label[-1][i] = 1
        '''
        embads_sent_label.append(i)
    #准备词嵌入之后的训练数据


    '''
    接下来需要构建所需要的测试集，测试集需要根据同样的embads词典
    这里需要注意的是：
    1.测试数据拥有和训练数据一致的sent_list结构和label_list结构
    2.根据韵姐的描述，每一个测试数据，其来源于对原先训练数据的单词替换，因此，还需要一个test_embads_sent_replace_count_list列表来记录替换的单词个数
    3.为了方便起见，建立另一个标志列表test_embads_corr_right_sent_list，将测试样本和其来源训练样本直接对应起来
    '''



    print('Begin construct train dataset and test dataset for every train sample generates '+str(per_sample_number))
    print(' test sample ，number of replace words is '+str(diverse)+' ')
    for sent in ten_words_data :
        print('\r'+'Processing '+str(ten_words_data.index(sent)+1)+'sentences , total '+str(len(ten_words_data))+' ',end="")
        #embeds = nn.Embedding(vocab_len,embedding_dim)
        sent_idxs = [vocab_dict[w] for w in sent]
        sent_idxs = torch.LongTensor(sent_idxs)
        sent_idxs = Variable(sent_idxs)
        sent_embeds = embeds(sent_idxs)
        embeds_sent_list.append(sent_embeds)
        for sent_idx in sent_idxs :
            sent_idx = Variable(torch.LongTensor([sent_idx]))
            embads_idx = embeds(sent_idx)
            if sent_idx not in embads_word_dic.keys() :
                embads_word_dic[sent_idx] = embads_idx
        
        
        #以上是对训练数据的词嵌入，现在对测试数据进行词嵌入
        sent_test_list = []
        idxs_list = []
        times = 1
        vocab_idxs = list(range(len(vocab_dict)))
        if nof_replaced_words_list == [] :
            nof_replaced_words_list = [nof_replaced_words]
        for replace_words_number in nof_replaced_words_list : 
            diverse = replace_words_number
            while times < per_sample_number :
                idxs = list(range(10))
                rd_replace_idxs = sample(idxs, diverse) #对于特定div和特定句子，咱原句中替换的位置index
                rd_replace_word_idxs = sample(vocab_idxs, diverse) #对于替换times个位置的句子，每个位置替换的新单词idx，这里为了保证替换的绝对成功，替换的单词都是原句中不存在的单词
                
                #确保以上随机成功不重复
                while rd_replace_idxs in idxs_list :
                    rd_replace_idxs = sample(idxs, diverse)
                    
                '''
                while any(rd_replace_word_idxs) in sent_idxs :
                    print(rd_replace_word_idxs)
                    rd_replace_word_idxs = sample(vocab_idxs, diverse)
                '''
                
                idxs_list.append(rd_replace_idxs) #确保这个组合之前没有出现过    
                
                
                newly_sent_idxs = copy.copy(sent_idxs.tolist()) #经过修改的目标测试数据        
                pos = 0
                for i in range(len(newly_sent_idxs)) :
                    if i in rd_replace_idxs :
                        
                        newly_sent_idxs[i] = rd_replace_word_idxs[pos]
                        pos = pos + 1
                
                newly_label = embads_sent_label[ten_words_data.index(sent)] #对应测试数据的正确标签
                
                
                newly_sent_idxs = torch.LongTensor(newly_sent_idxs)
                newly_sent_idxs = Variable(newly_sent_idxs)
                newly_sent_embeds = embeds(newly_sent_idxs)
                #将测试数据放入对应list
                test_embads_sent_list.append(newly_sent_embeds)
                test_embads_label_list.append(newly_label)
                times = times + 1
            g_embads_sent_list_dict[diverse] = test_embads_sent_list
            g_embads_sent_label_dict[diverse] = test_embads_label_list
    
        
    '''
    以上得到的结果：
    1.embeds_sent_list : 词嵌入之后的embaded_tensor列表
    2.embeds_sent_label : 每一个句子的类别数据
    注意，需要存储embeds的词组以及对应单词到文件中，训练文件和测试文件所用的embadding必须一致。
    3.embads_word_dic: 词对应词向量的词典，需要存储到文件中去  
    '''
    return embeds_sent_list,embads_sent_label,\
           test_embads_sent_list,test_embads_label_list,\
           g_embads_sent_list_dict,g_embads_sent_label_dict


#这里为ep数据集专门定义一个dataset类
class epmemo_dataset(Dataset):
    def __init__(self,embeds_sent_list,embeds_sent_label):

        self.x = torch.tensor([t.detach().numpy() for t in embeds_sent_list])
        self.y = torch.tensor(embeds_sent_label)

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.x)

#构建基于ep_dataset的dataloader
def embeds_data_loder(embeds_sent_list,embeds_sent_label):
    ep_dataset = epmemo_dataset(embeds_sent_list,embeds_sent_label)
    dataloader = DataLoader(
        ep_dataset,
        batch_size=100,
        shuffle=True
    )
    return dataloader


#搭建lstm的结构
class RNNLM(nn.Module):

    def __init__(self, embed_size=32, hidden_size=800, num_layers=3 , number_of_classes = 1000,drp = 0.2):
        super(RNNLM, self).__init__()
        self.dropout_in= nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,dropout=drp)
        self.dropout_out = nn.Dropout(p=0.225)
        self.linear = nn.Linear(hidden_size*10, number_of_classes)

    def forward(self, x,h,c):
        #这里的x输入必须是已经经过embadding的符号
        x = self.dropout_in(x)
        out, (h, c) = self.lstm(x,(h,c))
        #print(out.shape)
        #out = out[-1,:,:] #(batch_size*rint
        out = out.reshape(100,8000) #将所有隐含单元的输出都连到linear上面,100是batchsize
        out = self.dropout_out(out)
        #out= out.reshape(10,8000)
        out = self.linear(out)
        return out, (h, c)



def train(epoch,net,trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) #Adam From paper
    if params['run_mode'] == 0 :
        print('\nEpoch: %d' % epoch)
    #初始化训练结构
    #net.train()
    train_loss = 0
    correct = 0
    total = 0
    bt_idx = 0
    #训练数据和标签从trainloder引入
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        net.zero_grad()
        optimizer.zero_grad()

        init_h = torch.zeros(net.lstm.num_layers, trainloader.batch_size, net.lstm.hidden_size)
        init_c = torch.zeros(net.lstm.num_layers, trainloader.batch_size, net.lstm.hidden_size)
        outputs,(h,c) = net(inputs,init_h,init_c)   # inputs:1000,10,32

        #torch.squeeze(inputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1) #这里指定的是输出的标签，最大的那一个
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if params['run_mode'] == 0 :
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            print('\rIn training epoch '+str(epoch)+' last correct ration is %'+str(100.*correct/total),end='')
    return net,correct,total

def test(epoch,net,testloader):
    criterion = nn.CrossEntropyLoss()
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            init_h = torch.zeros(net.lstm.num_layers, testloder.batch_size, net.lstm.hidden_size)
            init_c = torch.zeros(net.lstm.num_layers, testloder.batch_size, net.lstm.hidden_size)
            outputs,(h,c) = net(inputs,init_h,init_c)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if params['run_mode'] == 0 :
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            else:
                print('\rIn test epoch '+str(epoch)+' last correct ration is %'+str(100.*correct/total),end='')
    return testloader,correct,total

def detach(states):
    #截断梯度
    return [state.detach() for state in states]

if __name__ == "__main__":
    results_dict = []
    for nof_rp_wds in params['nof_replaced_words']:
        r_dict = {}
        for hd_size in params['hidden_size']:
            embeds_sent_list,embads_sent_label,test_embads_sent_list,test_embads_label_list,g_embads_sent_list,g_embads_label_list = gen_trainandtest(nof_rp_wds)
            trainloder = embeds_data_loder(embeds_sent_list,embads_sent_label)
            testloder = embeds_data_loder(test_embads_sent_list,test_embads_label_list) 
            epmemo_net = RNNLM(hidden_size=hd_size)
            tr_crr = 0.0
            tr_total = 0.0
            g_te_crr = 0.0
            g_te_total = 0.0
            print('\nBegin train')
            for epoch in range(1000):
                epmemo_net,tr_crr,tr_total =  train(epoch,epmemo_net,trainloder)
                if tr_crr == 100.0 :
                    break
            print('\nBegin test')
            for epoch in range(10):
                testloder,te_crr,te_total = test(epoch,epmemo_net,testloder)
                g_te_crr = g_te_crr + te_crr
                g_te_total = g_te_total + te_total
            
            print('\nfor hidden_size '+str(params['hidden_size'])+' replace words '+str(params['nof_replaced_words'])+' test correct ratio : '+str(g_te_crr/g_te_total))
            print('\n\n\n')
            r_dict[hd_size] = g_te_crr/g_te_total
            #results_dict[params['hidden_size']][params['nof_replaced_words']] = g_te_crr/g_te_total
        results_dict.append(r_dict)
        pd.DataFrame(results_dict).to_csv('final_results.csv')
    
    
#commt of test results (i # of replaced words, a acc ) :
'''
i  1      2     3      4      5      6       7     8      9
a 1.0 | 0.94 | 0.95 | 0.87 | 0.66 | 0.39 | 0.22 | 0.10 | 0.02
''' 
