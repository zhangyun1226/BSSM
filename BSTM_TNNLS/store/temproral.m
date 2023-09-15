clear;
close all;
%% ʱ��ش���
load('data.mat');                                      %������������
load('spation512.mat');
initParameter;                                         %��ʼ������
seqenceNum = 1000;
wordLen = size(sequence,2);                            %ÿ�����е���󳤶�
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%�ӳٵ������PSP��ʱ��
targetLen = [];                                        %��Ŀ����Ԫ��������Ԫ����
targetThreshold = 1;                                   %Ŀ����Ԫ�ļ�����ֵ
targetWordNum = 4;                                     %Ŀ�굥��4������ʹĿ����ԪĤ��ѹ����1
wi = targetThreshold/(activeNum*targetWordNum);        %ÿ��������Ԫ��Ŀ����Ԫ��Ȩ��
segmentConnect = round(activeNum*0.7);           %ÿ��segment�ͼ�����Ԫ�����ĸ���-16
maxSegmentConnect = 4*segmentConnect;            %ÿ��segments������4�����ģ��Ҳ�Ҫ����
segmentThreshold = segmentConnect-1;             %Ԥ��ʱ���������������ֵ�ż���Ĥ��ѹ
segmentWi = 1/(segmentThreshold);                %segment��ÿ�����ӵ�Ȩ��
%% ������ʼ��
for i = 1:columnNum
    for j = 1:neuronNum
        segmentsList{i,j} = {};%�洢��i����j��Ԫ��Ȩ��
        delayList{i,j} = {};   %�洢��i����j��Ԫ���ӳ�
    end
end
targetsCell = {};                              %Ŀ����Ԫ�������������Ԫ��Ȩ���Լ�����ľ���
segmentsNum = zeros(columnNum,neuronNum);      %ÿ����Ԫsegments������,���������ж��ٸ�����
neuronConnectNum = zeros(columnNum,neuronNum); %ÿ����Ԫ�������ӵĴ���
%% ʱ���ѧϰ
for seq = 1:1:seqenceNum                            %�������о�������
    LearnState = zeros(columnNum,neuronNum,wordLen);%0:nT��ѧϰ״̬
    seqActiveTout = {};                             %�洢0:nT�������
    for word = 1:wordLen                            %������ǰ�������е�ÿ������
        sample = sequence(seq,word);                %��ǰ����
        if sample == 0
            %% ���н���
            %% ����Ŀ��ص�Ȩ���Լ��ӳ�
            targetWeights = zeros(columnNum,neuronNum);  %��Ŀ����Ԫ��Ȩ��
            targetDelays = Inf*ones(columnNum,neuronNum);%��Ŀ����Ԫ���ӳ�
            sentenceLen = word-1;                        %��ǰ���ӵĳ���
            for i = 1:sentenceLen                        %�����������е�ÿ������
                learnNeurons = find(LearnState(:,:,i) == 1); %�þ��ӵ�i�����ʵ�ѧϰ��Ԫ
                targetWeights(learnNeurons) = wi;            %��i�����ʵ�ѧϰ��Ԫ��Ŀ���Ȩ��
                targetDelays(learnNeurons) = (word - i)*Tmax;%��i�����ʵļ���Ԫ��Ŀ����ӳ٣�nT,(n-1)T...T
            end
            %% �洢��Ϣ targetsCell
            targetsCell{seq,1} = targetWeights;                 %Ȩ��
            targetsCell{seq,2} = targetDelays;                  %�ӳ�
            targetsCell{seq,3} = sequence(seq,1:sentenceLen);   %����
            %                     %% �洢ÿ�����ӱ������˶���
            %                     targetLenCell{seq,1} = length(find(targetWeights>0));
            %                     targetLenCell{seq,2} = sentenceLen*activeNum;
            %                     targetLenCell{seq,3} = targetLenCell{seq,1}-targetLenCell{seq,2};
            break;
        else
            %% one trail ѧϰ����
            %% A.����׶�
            %% step1:��ȡ��ǰ�ļ�������ÿ�������������
            activecolumns = columnCell(sample,:);      %��ǰ�ļ�����
            startT = (word-1)*Tmax;                    %�����ĸ�����
            currentTout = columnOut(sample,:) + startT;%�����������
            seqActiveTout{word} = currentTout;         %�洢�����������
            %% step2: ȷ����������ļ�����Ԫ������Ҫô�������У�Ҫô����Ԥ��״̬����Ԫ
            activeNeurons = zeros(columnNum,neuronNum);%������Ԫ
            if (word==1)                               %��һ������ѡ���������������ٵ���Ϊ������Ԫ
                %% ѡ���������������ٵ���Ϊѧϰ��Ԫ
                for num = 1:activeNum                                            %����������
                    currentColumn = activecolumns(num);                          %������i
                    neronsCon = neuronConnectNum(currentColumn,:);               %��i����Ԫ�����������Ŀ
                    minConNeuron = find(neronsCon == min(neronsCon));            %ѡ���������ӵ�������Ԫ
                    currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %�����ж�������ѡ��һ����Ԫ
                    LearnState(currentColumn,currentNeuron,word)=1;              %������Ԫ���뼤���б�
                end
            else         %��������
                preLearnState = LearnState(:,:,word-1);                          %��ȡ��һʱ�̵�ѧϰ��Ԫ
                %% ��ǰ���ʵ�ѧϰ
                for column = 1:activeNum                                         %����������
                    %% step2.1:Ϊsegmentѡ��Ҫ���ӵ���Ԫ�����ѡ��segmentConnect����Ԫ��Ϊsegment�����Ӷ���
                    [preLearnColumns,preLearnNeurons] = find(preLearnState == 1);%��ȡ��һʱ��ѧϰ��Ԫ��λ��
                    [preLearnColumns,columnIndex] = sort(preLearnColumns);       %�����������У�һ����Ҫ����Ϊ����ǰ������������е�
                    preLearnNeurons = preLearnNeurons(columnIndex);              %��Ӧ����Ԫ����
                    connectedColumn = sort(randperm(length(preLearnColumns),segmentConnect));%������ѧϰ��Ԫ���ѡ��segmentConnect������Ԫ
                    randActiveColumns = preLearnColumns(connectedColumn);        %���ѡ��ļ�����Ԫ��������
                    randActiveNeurons = preLearnNeurons(connectedColumn);        %���ѡ��ļ�����Ԫ�������������
                    chooseIndex = sub2ind(size(neuronConnectNum),randActiveColumns,randActiveNeurons); %����Щѡ����Ԫת��Ϊindex
                    %% step2.2��ѡ��ǰ���ʵļ������о�������segments����Ԫ
                    currentColumn = activecolumns(column);        %��ǰ�ļ�����i
                    currentSegNum = segmentsNum(currentColumn,:); %��ȡ��ǰ��������������Ԫ��segments��Ŀ
                    minSegNum = min(currentSegNum);               %���ٵ�segment��Ŀ
                    neurons = find(currentSegNum == minSegNum);   %ѡ���������segments����Ԫ��
                    chooseNeuron = neurons(unidrnd(length(neurons)));%����Щ��Ԫ�У����ѡ��һ����Ԫ
                    %% step2.3 ��������Ԫ��segments���ҵ�һ��segment�����Ӷ���û�н���������Щsegments����������
                    segmentFlag = 0; %�Ƿ���Ҫ�½�segments��0��Ҫ��1����Ҫ
                    if minSegNum~=0  %���ѡ���ѧϰ��Ԫ��segment������Ϊ0
                        segments = segmentsList{currentColumn,chooseNeuron};%��ȡ����Ԫ��segments
                        for segIndex = 1:minSegNum           %����segments
                            segment = segments{segIndex};    %segmenti����Ϣ
                            segmentSynIdx = segment(:,1);    %segmenti���ӵ���Ԫ��index
                            if length(segmentSynIdx)~=maxSegmentConnect  %�����ǰsegment�����Ӹ���û�������ֵ
                                
                                diffNeurons = chooseIndex-segmentSynIdx';     %segments���ӵ���Ԫ������ѡ������Ӷ���֮��
                                [~,comIdx] = find(diffNeurons==0);            %Ϊ0��ʾ����ͬ��Ԫ��
                                if isempty(comIdx)                            %û�н���
                                    %% segment���뵱ǰ����Ϣ
                                    segmentLen = length(segmentSynIdx);       %��ǰsegment��������Ԫ����
                                    segment( segmentLen+1: segmentLen+segmentConnect,1) = chooseIndex;%2.3.1segments�����ĵ���Ԫ��index
                                    segment( segmentLen+1: segmentLen+segmentConnect,2) = segmentWi;  %2.3.2��segments�����ĵ���Ԫ��Ȩ��
                                    segment( segmentLen+1: segmentLen+segmentConnect,3) = sample;     %2.3.3��ǰ���ʵ���Ϣ
                                    %% delay
                                    postTout = currentTout(column);           %��ǰ�������
                                    preTouts = seqActiveTout{word-1};         %ȡ��һ��ļ����������
                                    preTouts = preTouts(connectedColumn);     %ѡ����������
                                    delay = postTout - (preTouts + DelayTime);%�����ӳ�
                                    segment( segmentLen+1: segmentLen+segmentConnect,4) = delay;      %2.3.4�洢��segments�����ĵ���Ԫ���ӳ�
                                    %%
                                    neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %ѡ����Ԫ������������1
                                    segments{segIndex} =  segment;                                    %����segment��Ϣ
                                    segmentsList{currentColumn,chooseNeuron} = segments;              %segment�����б�
                                    segmentFlag =1;
                                    break;
                                end
                            end
                        end
                    end
                    %% step2.4 û���ҵ����ʵ�segment�����Ϣ���½�һ��segments������Ԫ
                    if segmentFlag == 0
                        %% �½���segments
                        segment = zeros(segmentConnect,4);
                        segment(:,1) = chooseIndex;       %2.4.1��segments�����ĵ���Ԫ��index
                        segment(:,2) = segmentWi;         %2.4.2��segments�����ĵ���Ԫ��Ȩ��
                        segment(:,3) = sample;            %2.4.3��segments�ı�ǩ
                        %% 2.4.4 delay����
                        postTout = currentTout(column);   %��ǰ�������
                        preTouts = seqActiveTout{word-1}; %ȡ��һ��ļ����������
                        preTouts = preTouts(connectedColumn);      %ѡ����������
                        delay = postTout - (preTouts + DelayTime); %�����ӳ�
                        segment(:,4) = delay;                      %2.4.4�洢��segments�����ĵ���Ԫ���ӳ�
                        %% step3: ���½���segment����segment�б�
                        num = segmentsNum(currentColumn,chooseNeuron);            %��ǰѧϰ��Ԫӵ�е�segments������
                        currentSegCell = segmentsList{currentColumn,chooseNeuron};%��ǰѧϰ��Ԫӵ�е�segments
                        currentSegCell{num+1} = segment;                          %����segmentд��
                        segmentsList{currentColumn,chooseNeuron} = currentSegCell;%������Ϣ
                        neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %ѡ����Ԫ������������1
                        segmentsNum(currentColumn,chooseNeuron) = num+1;                  %����Ԫ��segment����1
                    end
                    LearnState(currentColumn,chooseNeuron,word) = 1;       %ѧϰ��Ԫ״̬Ϊ1
                end
            end
        end
    end
end

fprintf('learning is over!!!\n');
fprintf('testing star!!!\n');


