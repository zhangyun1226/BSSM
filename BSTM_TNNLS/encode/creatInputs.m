clear
clc
%% 创建每个单词的标签，以及位置，还有相应的输入表示
fid=fopen('new_cbt_0307.txt');  %打开文本文件
inputLablesCell = {};           %存储单词标签
sequenceCell = {};              %存储每个句子的表示
idx=0;

while ~feof(fid)
    %% 读取每一个句子
    str = fgetl(fid);      % 读取一行, str是字符串
    s=regexp(str,'\s+');   % 找出str中的空格, 以空格作为分割单词的字符
    s = [0,s];             % 第一个单词的空格为0
    idx=idx+1;             % 句子的序号
    %% 对于每一个句子的所有单词
    for i=1:length(s)      % 将该句的所有单词全部存下来
        %% 1.1 获取第i单词的字符串
        if i~=length(s)
            currentStr = str(s(i)+1:s(i+1)-1);
        else
            currentStr = str(s(i)+1:end);
        end
        
        %% 1.2 存储单词i的标签，存储过就找到存储的位置，没有就新建一个
        hasSave = 0;                           %单词是否被存储过，0没有，1有
        for j = 1:length(inputLablesCell)      %遍历所有存储过的单词
            preStr = inputLablesCell{j};       %第j个单词的字符串
            strSim = strcmp(currentStr,preStr);%比较当前单词和之前单词的相似度
            if strSim == 1                     %若完全相同
                hasSave = 1;                   %以前存储过
                break;                         %返回
            end
        end
        if hasSave == 0                         %未存储过
            inputLablesCell{end+1} = currentStr;          %新单词分配标签
            sequenceCell{idx,i} = length(inputLablesCell);%当前句子单词i的位置填上标签
        else                                    %存储过
            sequenceCell{idx,i} = j;            %使用以前的标签
        end
    end
end
fclose(fid);

%% 将sequenceCell转换成数组
sequenceNum = size(sequenceCell,1);      %句子的总数
wordNum = size(sequenceCell,2);          %最长的句子的单词个数
sequence = zeros(sequenceNum,wordNum+1); %留一个0出来当结束的标记，长度不够的句子用0补充
for i = 1:sequenceNum                    %遍历所有句子
    for j = 1:wordNum                    %遍历所有单词
        wordLabel = sequenceCell{i,j};        %获取当前单词的标签
        if ~isempty(wordLabel)                %存在标签
            sequence(i,j) = wordLabel;        %复制到数组中
        else                             %不存在
            break;                       %下一句
        end
    end
end

%% 将存储句子写入文本，不写入长度不为10的句子
fileID = fopen('new_cbt_0307_write.txt','w');
for i =  1:sequenceNum           %遍历所有句子
    sequncei = sequence(i,:);    %当前句子i的表示
    sequncei(sequncei == 0) = [];%去除多余的单词
    if length(sequncei)>=10      %如果句子长度大于等于10
        %% 将单词写入新文本new_cbt_0307_write
        for j = 1:length(sequncei)-1     %遍历前n-1个单词
            wordLabel = sequence(i,j);   %获取当前单词的标签
            fprintf(fileID,'%s ',inputLablesCell{wordLabel});%单词后加空格写入文本
        end
        wordLabel = sequence(i,length(sequncei));            %获取末尾单词的标签
        fprintf(fileID,'%s\n',inputLablesCell{wordLabel});   %不加空格,结束
    else      %若句子长度小于10
        i    %输出句子的位置
    end
end

%% 相同的句子去重
for i =  1:sequenceNum-1      %遍历句子
    sequncei = sequence(i,:); %当前句子i的表示
    sequncei(sequncei == 0) = [];
    for j = i+1:sequenceNum       %遍历句子i以后的所有句子
        sequncej = sequence(j,:); %句子j的表示
        sequncej(sequncej == 0) = [];
        if length(sequncei) == length(sequncej) %判断两个句子长度是否相同，不相同肯定不是一个句子
            if all(sequncei - sequncej == 0)    %句子相同，判断是不是每个位置都一样
                j                               %都一样，输出句子j的位置
            end
        end
    end
end
%% 为每个单词建立一个input spike train
%% 创建数据
rate_in = 10;     % 空间层输入的频率
sysinput = 1000;  % 空间层输入神经元数量
Tmax = 30;        %周期长度
sampleLen = length(inputLablesCell); %所有单词的个数
for wordLabel = 1:sampleLen          %遍历所有单词
    %% 为每个单词建立一个泊松分布的脉冲序列
    num =  poissrnd(rate_in * Tmax/ 1000, sysinput, 1); %每个输入神经元的脉冲个数
    maxNum = max(num);                                  %最大脉冲个数
    input = inf(sysinput, maxNum);                      %输入初始化
    for i = 1: sysinput
        input(i, 1:num(i)) = round(Tmax * rand(1, num(i)));
    end
    input = sort(input,2);                              %输入排序
    inputCell{wordLabel} = input;                       %加入列表
end
save dataAll.mat  inputLablesCell inputCell sequence
%% 只要前1000句
sequence = sequence(1:1000,:);
maxWord = max(sequence(:));
inputLablesCell = inputLablesCell(1:maxWord);
inputCell = inputCell(1:maxWord);
save data.mat  inputLablesCell inputCell sequence



