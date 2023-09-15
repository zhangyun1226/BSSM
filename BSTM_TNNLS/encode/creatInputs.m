clear
clc
%% ����ÿ�����ʵı�ǩ���Լ�λ�ã�������Ӧ�������ʾ
fid=fopen('new_cbt_0307.txt');  %���ı��ļ�
inputLablesCell = {};           %�洢���ʱ�ǩ
sequenceCell = {};              %�洢ÿ�����ӵı�ʾ
idx=0;

while ~feof(fid)
    %% ��ȡÿһ������
    str = fgetl(fid);      % ��ȡһ��, str���ַ���
    s=regexp(str,'\s+');   % �ҳ�str�еĿո�, �Կո���Ϊ�ָ�ʵ��ַ�
    s = [0,s];             % ��һ�����ʵĿո�Ϊ0
    idx=idx+1;             % ���ӵ����
    %% ����ÿһ�����ӵ����е���
    for i=1:length(s)      % ���þ�����е���ȫ��������
        %% 1.1 ��ȡ��i���ʵ��ַ���
        if i~=length(s)
            currentStr = str(s(i)+1:s(i+1)-1);
        else
            currentStr = str(s(i)+1:end);
        end
        
        %% 1.2 �洢����i�ı�ǩ���洢�����ҵ��洢��λ�ã�û�о��½�һ��
        hasSave = 0;                           %�����Ƿ񱻴洢����0û�У�1��
        for j = 1:length(inputLablesCell)      %�������д洢���ĵ���
            preStr = inputLablesCell{j};       %��j�����ʵ��ַ���
            strSim = strcmp(currentStr,preStr);%�Ƚϵ�ǰ���ʺ�֮ǰ���ʵ����ƶ�
            if strSim == 1                     %����ȫ��ͬ
                hasSave = 1;                   %��ǰ�洢��
                break;                         %����
            end
        end
        if hasSave == 0                         %δ�洢��
            inputLablesCell{end+1} = currentStr;          %�µ��ʷ����ǩ
            sequenceCell{idx,i} = length(inputLablesCell);%��ǰ���ӵ���i��λ�����ϱ�ǩ
        else                                    %�洢��
            sequenceCell{idx,i} = j;            %ʹ����ǰ�ı�ǩ
        end
    end
end
fclose(fid);

%% ��sequenceCellת��������
sequenceNum = size(sequenceCell,1);      %���ӵ�����
wordNum = size(sequenceCell,2);          %��ľ��ӵĵ��ʸ���
sequence = zeros(sequenceNum,wordNum+1); %��һ��0�����������ı�ǣ����Ȳ����ľ�����0����
for i = 1:sequenceNum                    %�������о���
    for j = 1:wordNum                    %�������е���
        wordLabel = sequenceCell{i,j};        %��ȡ��ǰ���ʵı�ǩ
        if ~isempty(wordLabel)                %���ڱ�ǩ
            sequence(i,j) = wordLabel;        %���Ƶ�������
        else                             %������
            break;                       %��һ��
        end
    end
end

%% ���洢����д���ı�����д�볤�Ȳ�Ϊ10�ľ���
fileID = fopen('new_cbt_0307_write.txt','w');
for i =  1:sequenceNum           %�������о���
    sequncei = sequence(i,:);    %��ǰ����i�ı�ʾ
    sequncei(sequncei == 0) = [];%ȥ������ĵ���
    if length(sequncei)>=10      %������ӳ��ȴ��ڵ���10
        %% ������д�����ı�new_cbt_0307_write
        for j = 1:length(sequncei)-1     %����ǰn-1������
            wordLabel = sequence(i,j);   %��ȡ��ǰ���ʵı�ǩ
            fprintf(fileID,'%s ',inputLablesCell{wordLabel});%���ʺ�ӿո�д���ı�
        end
        wordLabel = sequence(i,length(sequncei));            %��ȡĩβ���ʵı�ǩ
        fprintf(fileID,'%s\n',inputLablesCell{wordLabel});   %���ӿո�,����
    else      %�����ӳ���С��10
        i    %������ӵ�λ��
    end
end

%% ��ͬ�ľ���ȥ��
for i =  1:sequenceNum-1      %��������
    sequncei = sequence(i,:); %��ǰ����i�ı�ʾ
    sequncei(sequncei == 0) = [];
    for j = i+1:sequenceNum       %��������i�Ժ�����о���
        sequncej = sequence(j,:); %����j�ı�ʾ
        sequncej(sequncej == 0) = [];
        if length(sequncei) == length(sequncej) %�ж��������ӳ����Ƿ���ͬ������ͬ�϶�����һ������
            if all(sequncei - sequncej == 0)    %������ͬ���ж��ǲ���ÿ��λ�ö�һ��
                j                               %��һ�����������j��λ��
            end
        end
    end
end
%% Ϊÿ�����ʽ���һ��input spike train
%% ��������
rate_in = 10;     % �ռ�������Ƶ��
sysinput = 1000;  % �ռ��������Ԫ����
Tmax = 30;        %���ڳ���
sampleLen = length(inputLablesCell); %���е��ʵĸ���
for wordLabel = 1:sampleLen          %�������е���
    %% Ϊÿ�����ʽ���һ�����ɷֲ�����������
    num =  poissrnd(rate_in * Tmax/ 1000, sysinput, 1); %ÿ��������Ԫ���������
    maxNum = max(num);                                  %����������
    input = inf(sysinput, maxNum);                      %�����ʼ��
    for i = 1: sysinput
        input(i, 1:num(i)) = round(Tmax * rand(1, num(i)));
    end
    input = sort(input,2);                              %��������
    inputCell{wordLabel} = input;                       %�����б�
end
save dataAll.mat  inputLablesCell inputCell sequence
%% ֻҪǰ1000��
sequence = sequence(1:1000,:);
maxWord = max(sequence(:));
inputLablesCell = inputLablesCell(1:maxWord);
inputCell = inputCell(1:maxWord);
save data.mat  inputLablesCell inputCell sequence



