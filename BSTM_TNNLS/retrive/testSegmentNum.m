clear;
close all;
%% ʱ��ش���
%% ������������
load('data.mat');                                      
load('spation512.mat');
%% ��ʼ������
seqenceNum = 1000;                                     %���и���
initParameter;                                         %Ĭ�ϵĲ���
wordLen = size(sequence,2);                            %ÿ�����е���󳤶�
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%�ӳٵ������PSP��ʱ��
targetLen = [];                                        %��Ŀ����Ԫ��������Ԫ����
targetThreshold = 1;                                   %Ŀ����Ԫ�ļ�����ֵ
targetWordNum = 4;                                     %Ŀ�굥��4������ʹĿ����ԪĤ��ѹ����1
wi = targetThreshold/(activeNum*targetWordNum);        %ÿ��������Ԫ��Ŀ����Ԫ��Ȩ��
segmentConnect = round(activeNum*0.7);                 %ÿ��segment�ͼ�����Ԫ�����ĸ���-16
maxSegmentConnect = 4*segmentConnect;                  %ÿ��segments������4�����ģ��Ҳ�Ҫ����
segmentThreshold = segmentConnect-1;                   %Ԥ��ʱ���������������ֵ�ż���Ĥ��ѹ
segmentWi = 1/(maxSegmentConnect);                     %segment��ÿ�����ӵ�Ȩ��
maxSegmentList = [10,20,30,40];                        %Զ����ͻ������
trail = 1;                                             %ʵ�����
for disSegNum = 1:length(maxSegmentList)               %����Զ����ͻ������
    maxSegment = maxSegmentList(disSegNum);            %�ṩ������ĸ���
    t1=clock;
    for trailNum = 1:trail                            
        %% ÿ��ʵ��������³�ʼ��
        %��Ԫsegments��λ��
        for i = 1:columnNum
            for j = 1:neuronNum
                cellSegmentsIdx{i,j} = {};     
            end
        end
        contextActiveList = cell(columnNum,neuronNum); %�洢��i����j��Ԫ������segments��Ԥ��ʱʹ��
        targetActiveList = cell(columnNum,neuronNum);  %�洢��i����j��Ԫ��Ŀ����Ԫ��Ԥ��ʱʹ��
        segmentsList = {};                             %�洢segments����Ϣ���Լ������ĸ���Ԫ��segments
        targetsCell = {};                              %Ŀ����Ԫ�������������Ԫ��Ȩ���Լ�����ľ���
        cellSegmentsNum = zeros(columnNum,neuronNum);  %ÿ����Ԫsegments������,���������ж��ٸ�����
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
                        %% �������ӵ�Ŀ����Ԫ�洢����
                        targetInfo = targetActiveList(learnNeurons);
                        for preIdx = 1: length(targetInfo)
                            if isempty(find(targetInfo{preIdx} == seq))
                                targetInfo{preIdx} = [targetInfo{preIdx},seq];
                            end
                        end
                        targetActiveList(learnNeurons) = targetInfo;
                    end
                    %% �洢��Ϣ targetsCell
                    targetsCell{seq,1} = targetWeights;                 %Ȩ��
                    targetsCell{seq,2} = targetDelays;                  %�ӳ�
                    targetsCell{seq,3} = sequence(seq,1:sentenceLen);   %����
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
                            currentSegNum = cellSegmentsNum(currentColumn,:); %��ȡ��ǰ��������������Ԫ��segments��Ŀ
                            minSegNum = min(currentSegNum);               %���ٵ�segment��Ŀ
                            neurons = find(currentSegNum == minSegNum);   %ѡ���������segments����Ԫ��
                            chooseNeuron = neurons(unidrnd(length(neurons)));%����Щ��Ԫ�У����ѡ��һ����Ԫ
                            %% step2.3��������Ԫ��segments���ҵ�һ��segment�����Ӷ���û�н���������Щ�����ӽ���������
                            activeSegmentIdx = [];
                            segmentFlag = 0; %�Ƿ���Ҫ�½�segments��0��Ҫ��1����Ҫ
                            if minSegNum~=0  %���ѡ���ѧϰ��Ԫ��segment������Ϊ0
                                segmentsIdx = cellSegmentsIdx{currentColumn,chooseNeuron};%��ȡ����Ԫ��segments
                                for segI = 1:minSegNum           %����segments
                                    segment = segmentsList{segmentsIdx{segI},1};   % segmenti����Ϣ
                                    segmentSynIdx = segment(:,1);                % segmenti���ӵ���Ԫ��index
                                    if length(segmentSynIdx)~=maxSegmentConnect  % �����ǰsegment�����Ӹ���û�������ֵ
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
                                            segmentsList{segmentsIdx{segI},1} = segment;                      %����segment��Ϣ
                                            segmentFlag =1;                                                   %�����Ѿ���ӣ������½�
                                            activeSegmentIdx = segmentsIdx{segI};                             %������ӵ�segment��λ��
                                            break;
                                        end
                                    end
                                end
                            end
                            
                            %% step2.4 û���ҵ����ʵ�segment�����Ϣ���½�һ��segments������Ԫ
                            if segmentFlag == 0&&minSegNum<maxSegment
                                %% �½���segments
                                segment = zeros(segmentConnect,4);
                                segment(:,1) = chooseIndex;       %2.4.1��segments�����ĵ���Ԫ��index
                                segment(:,2) = segmentWi;         %2.4.2��segments�����ĵ���Ԫ��Ȩ��
                                segment(:,3) = sample;            %2.4.3��segments�ı�ǩ
                                %% 2.4.4 delay����
                                postTout = currentTout(column);            %��ǰ�������
                                preTouts = seqActiveTout{word-1};          %ȡ��һ��ļ����������
                                preTouts = preTouts(connectedColumn);      %ѡ����������
                                delay = postTout - (preTouts + DelayTime); %�����ӳ�
                                segment(:,4) = delay;                      %2.4.4�洢��segments�����ĵ���Ԫ���ӳ�
                                %% step3: ���½���segment����segment���б�
                                segmentNum = sum(cellSegmentsNum(:)) + 1;             %��segment��idx
                                segmentsList{segmentNum,1} = segment;                 %�������б�
                                segmentsList{segmentNum,2} = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron); %��д����Ӧ����Ԫ
                                % ����Ԫ��segment��λ��
                                cellSegmentsIdx{currentColumn,chooseNeuron} = [cellSegmentsIdx{currentColumn,chooseNeuron},segmentNum];
                                cellSegmentsNum(currentColumn,chooseNeuron) = cellSegmentsNum(currentColumn,chooseNeuron)+1;   %����Ԫ��segment����1
                                neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %ѡ����Ԫ������������1
                                activeSegmentIdx = segmentNum;                     %��segmentλ��
                            end
                            LearnState(currentColumn,chooseNeuron,word) = 1;       %ѧϰ��Ԫ״̬Ϊ1
                            
                            %% step2.5 ��������Ԫ���ӵ�����context�洢����
                            contextInfo = contextActiveList(chooseIndex);
                            for preIdx = 1: length(contextInfo)
                                contextInfo{preIdx} = [contextInfo{preIdx},activeSegmentIdx];
                            end
                            contextActiveList(chooseIndex) = contextInfo;
                        end
                    end
                end
            end
        end
        
        fprintf('learning is over!!!\n');
        fprintf('testing star!!!\n');
        
        %% test ֻ���벿�ֵ���
        predictNoneThreshold = 0.1; %Ĥ��ѹ��ֵ�������ڣ�������Ԥ������
        predictHasThreshold = 0.5;  %Ĥ��ѹ��ֵ�������ڣ���Ԥ���������Զ��
        diffThresholdMany = 0.005;
        diffThresholdNone = 0.0005;
        endTime = DelayTime+T_step;
        segmentsLen = length(segmentsList);
        zerosInit = zeros(columnNum,neuronNum);
        minTargetThreshold = 0;
        InfInit = Inf*ones(columnNum,neuronNum);
        %% �������е�����
        testInputNum = 3;                                      %���뵥�ʵĸ���
        testSeq = sequence;                %�������е�����������
        testSeq(:,testInputNum+1:end) = -1;%����������������Ժ�����
        seqenceNum = size(sequence,1);     %�������е��ܾ���
        testSeqNum = seqenceNum;           %�������е��ܾ���
        
        %% ȥ���ظ�������
        for i = 1:seqenceNum          %�������о���
            if i>testSeqNum
                break;                %����ظ����Եľ��ӱ�ɾ���󣬾����Ѿ���ĩβ��
            end
            currentSeq = testSeq(i,1:testInputNum); %��ȡ��ǰ��i�����ӵ�����
            delSeqIdx = [];                         %���ڴ���Ƿ��к���һ���ľ��ӣ������ľ�����Ҫ��ɾ��
            for j = i+1:testSeqNum                  %�鿴��ǰ�����Ժ�����в��Ծ���
                compareSeq = testSeq(j,1:testInputNum);%��ȡ����j
                if all(currentSeq - compareSeq == 0)   %����������һ��һ��
                    delSeqIdx(end+1) = j;              %j���Ӽ���ɾ������
                end
            end
            if length(delSeqIdx)~=0                          %���о�����ɾ��������
                testSeqNum = testSeqNum - length(delSeqIdx); %���Ծ��������½�
                testSeq(delSeqIdx,:) = [];                   %ȥ��ɾ�����еľ���
            end
        end
        wordLen = size(testSeq,2);    %���Ծ��ӵ���󵥴ʸ���
        
        %% ����λ���Լ���Ϣ
        str = 'test_segmentNum_';
        str = [str,num2str(maxSegment)];
        str = [str,'_'];
        str = [str,num2str(trailNum)];
        str = [str,'.txt'];
        fileID = fopen(str,'w');
        
        %% �洢��һ�����ʵ���Ϣ
        FirstWordInfo = {};
        for sentencei = 1:max(testSeq(:))
            FirstWordInfo{sentencei,1} = {};
        end
        
        %% ���Կ�ʼ
        for seq = 1:testSeqNum
            %             seq
            predictNeuronsState = zeros(columnNum,neuronNum,wordLen);%0:nT��Ԥ��״̬
            for word = 1:wordLen                       %������ǰ���ӵ�������ĵ���
                sample = testSeq(seq,word);            %��ȡ����ĵ��ʵ�����
                if sample ~= -1                        %����δ����
                    %% A.����׶�
                    %% step1: ��ȡ��ǰ�ļ�������ÿ�������������
                    activecolumns = columnCell(sample,:);      %��ǰ�ļ�����
                    startT = (word-1)*Tmax;                    %�����ĸ�����
                    currentTout = columnOut(sample,:) + startT;%�����������
                    
                    %% step2: ȷ����������ļ�����Ԫ������Ҫô�������У�Ҫô����Ԥ��״̬����Ԫ
                    activeNeurons = zerosInit;   %��ǰ���ڵļ�����Ԫ,ȫ����ʼ��Ϊ0
                    activeTout = InfInit;        %��ǰ���ڵļ�����Ԫ�����,ȫ����ʼ��ΪInf
                    if (word==1)
                        %% step2.1 ��һ��������û��Ԥ����Ԫ�ģ�ȫ����
                        activeNeurons(activecolumns,:) = 1;%����������Ԫ�����ڼ���״̬
                        activeTout(activecolumns,:) = repmat(currentTout',1,neuronNum);
                    else
                        %% step2.1 �������ʣ����μ��ÿ�����������鿴���Ƿ����Ԥ����Ԫ
                        for column = 1:activeNum           %����������
                            flagPredect = 0;               %��ǰ���Ƿ�����Ԫ��ѡΪԤ����Ԫ��0��1��
                            currentColumn = activecolumns(column); %��ǰ������
                            preNeuron = find(predictNeuronsState(currentColumn,:,word-1)==1); %�鿴��ǰ������Щ��Ԫ����Ԥ��״̬
                            
                            %% step 2.2 ����Ԥ����Ԫ����������Ԥ����Ԫ
                            if ~isempty(preNeuron)
                                flagPredect = 1;                           %����Ԥ����Ԫ
                                activeNeurons(currentColumn,preNeuron) = 1;%����Ԫ��Ϊ������Ԫ
                                activeTout(currentColumn,preNeuron) = currentTout(column);
                            end
                            
                            %% step 2.3 ������Ԥ����Ԫ������������
                            if (flagPredect == 0)
                                activeNeurons(currentColumn,:) = 1;
                                activeTout(currentColumn,:) =  currentTout(column);
                            end
                        end
                    end
                    
                    %% �������Ԫ��Ӧ�����
                    activeIndex = find(activeNeurons == 1);                %������Ԫ����Ӧ��index
                    actTout = activeTout(activeIndex);
                    %% ��������Ԫ/���/�Լ��������浽inputInfo��
                    if word == 1
                        inputInfo{1} = sample;
                        inputInfo{2} = activeIndex;
                        inputInfo{3} = actTout;
                    else
                        inputInfo{1} = [inputInfo{1};sample];
                        inputInfo{2} = [inputInfo{2};activeIndex];
                        inputInfo{3} = [inputInfo{3};actTout];
                    end
                    
                    %% B.Ԥ��׶�,�ü�����Ԫ��Ԥ��
                    if word == 1 && ~isempty(FirstWordInfo{sample})  %���Ѿ�ѧϰ������ô��Ϣ�Ѿ��洢�����ˣ�ֱ�ӵ��þ���
                        predictState = FirstWordInfo{sample,1};      %Ԥ��״̬
                        %                         predictWordList = FirstWordInfo{sample,2};   %Ԥ��ĵ����б�
                        %                         predictInformation = FirstWordInfo{sample,3};%Ԥ�����Ԫ�Լ��������
                    else                                             %�����ǵ�һ�����ʣ����ߵ�һ�����ʵ�һ�γ��֣�ѧϰԤ��
                        segmentState = zeros(segmentsLen,1);   %segments�뼤����Ԫ����������
                        %% step1:�������м�����Ԫ,�鿴��Щsegments����
                        for neuronIdx = 1:length(activeIndex)
                            neuronSegIndex = activeIndex(neuronIdx);        %�������Ԫ
                            contextInfo = contextActiveList{neuronSegIndex};%��ȡ�ü�����Ԫ��������Ԫ��segments
                            for segI = 1:neuronConnectNum(neuronSegIndex)   %��������������Ԫ�ļ���segments
                                activeSegId = contextInfo(segI);            %��ȡsegment��λ��
                                segmentState(activeSegId) = segmentState(activeSegId) + 1;%��segment�뼤����Ԫ������������1
                            end
                        end
                        predictState = zeros(columnNum,neuronNum);   %ÿ����Ԫ��Ԥ��״̬����ʼΪ0����Ԥ��
                        activeSegments = find(segmentState >= segmentThreshold);%�뼤����Ԫ������������������ֵ�ļ���
                        for segIdx = 1:length(activeSegments)
                            actSegIdx = activeSegments(segIdx);      %��ȡ�����segments��λ��
                            neuronSegIndex = segmentsList{actSegIdx,2};    %��ȡ�����segment����Ӧ����Ԫ
                            predictState(neuronSegIndex) = 1;              %����Ԫ��ǰʱ�̱�ΪԤ��״̬
                        end
                        %% �洢��һ�����ʵ���Ϣ
                        if word == 1
                            FirstWordInfo{sample,1} = predictState;
                            %                             FirstWordInfo{sample,2} = predictWordList;
                            %                             FirstWordInfo{sample,3} = predictInformation;
                        end
                    end
                    
                    %% �洢Ԥ����Ϣ
                    predictNeuronsState(:,:,word) = predictState;
                else
                    outputInfo = {};                                %�洢������е���Ϣ
                    predictToutQuick;
                    hasContextNeurons = find(hasContextNeurons~=0); %��Ԥ��ļ�����Ԫ
                    %��Ԥ��ļ�����Ԫ����������Ԫ���Ҳ���ȫ�����ô�õ�����Ҷ�ӽڵ�
                    if (length(activeIndex) - length(hasContextNeurons) >= segmentThreshold)&&length(activeIndex)~=activeNum*neuronNum
                        predicSegLen = size(outputInfo,1);
                        outputInfo{predicSegLen+1,1} = inputInfo{1};%��1�д�þ������е��ʱ�ǩ
                        outputInfo{predicSegLen+1,2} = inputInfo{2};%��2�д�þ������м�����Ԫ
                        outputInfo{predicSegLen+1,3} = inputInfo{3};%��3�д�����м������Ԫ�����
                    end
                    %% ��ʼ����Ԥ��-��ͬ����ʾ��ͬ�Ӿ䣬���ʱ�ʾ��ͬ�ڵ�
                    haveReadList = zeros(size(predictWordList));   %���ڴ洢�ڵ㵥���Ƿ����������ʼ����δ������
                    while (~isempty(predictWordList))              %�Ѿ��������ˣ�ջ��û���κ�Ԫ����
                        %% ����δ�������ĵ���-�ӽ��
                        listLen = find(haveReadList == 0);
                        %% ջ���������ڣ�ɾ����Ϣ�˳�
                        if(isempty(listLen)) || length(outputInfo)>seqenceNum
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            break;
                        end
                        listLen = listLen(end);                               %�������Ǹ����-ջ�����
                        currentWord = predictWordList(listLen);               %�����ڵĵ��ʱ�ǩ
                        currentSegmentsCell = predictInformation(listLen,1:2);%�����ڵĵ��ʵ�Ԥ����Ԫ��Ϣ
                        haveReadList(listLen) = 1;                            %�õ����Ѿ��ù���
                        %% û������Ԥ��״̬��Ϊ��Ծ״̬
                        actIndex = currentSegmentsCell{1,2};               %������Ԫ���ڵ�index
                        if length(actIndex) < segmentThreshold             %���������ԪС����ֵ�����ܴ���һ�������ĵ��ʣ�ȥ��
                            %% ȥ��������ʣ���Ϊ�������������ܻᵼ��ջ��������
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            continue;
                        end
                        
                        PredictToPredict;
                        %% step3 �ж��Ƿ�ΪҶ�ӽ��Ҷ�ӽ��1���жϵ�ǰ�Ƿ��м������Ԫû��Ԥ��ģ��������ӽ����������б�
                        leaf = 0;
                        hasContextNeurons = find(hasContextNeurons~=0);                %��Ԥ��ļ�����Ԫ
%                         %Ԥ��ļ�����Ԫ<��ǰ�ļ�����Ԫ���Ҳ���ȫ����ĵ��ʵ��µ�
%                         if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold) &&length(actIndex)~=activeNum*neuronNum
%                             leaf = 1;
%                         end
                        %% �жϵ�ǰ��û���µ�Ԥ�⣬���У���ô����ջ����û�У���ô��Ҷ�ӽ��
                        nextLen = length(tempWordList);
                        if nextLen >= segmentThreshold %%��Ҷ�ӽ��
                            predictWordList(end+1:end+nextLen) = tempWordList;        %����Ԥ�ⵥ���б�
                            predictInformation(end+1:end+nextLen,:) = tempInformation;%����Ԥ�ⵥ����Ϣ
                            haveReadList(end+1:end+nextLen) = 0;                      %��Щ���δ��������
                        else
                            leaf = 1; %% Ҷ�ӽ��
                        end
                        %% Ҷ�ӽ�㣬��ȡ�Ӿ䣬������Ϣ
                        if leaf == 1
                            %% �������дʵı�ǩ
                            wordList = find(haveReadList == 1);  %����Ϊ1�ĵ�������������
                            predicSegLen = size(outputInfo,1);   %��ǰ�洢�Ӿ�ĸ����������Ӿ����ĩβ
                            temp = predictWordList(wordList);
                            if size(temp,2)>1
                                temp = reshape(temp,length(temp),1);
                            end
                            outputInfo{predicSegLen+1,1} = [inputInfo{1};temp];     %��1�д�þ������е��ʱ�ǩ
                            tout = [];
                            activeNeurons = [];
                            for preLen = 1:length(temp)
                                tout = [tout;predictInformation{wordList(preLen),2}];
                                activeNeurons = [activeNeurons;predictInformation{wordList(preLen),1}'];
                            end
                            outputInfo{predicSegLen+1,2} = [inputInfo{2};tout];%��2�д�þ������е��ʼ������Ԫ
                            outputInfo{predicSegLen+1,3} = [inputInfo{3};activeNeurons];%��3�м������Ԫ�����
                            
%                         end
%                         if leaf == 1
                        %% ɾ�����Ӿ���������Ϣ
                            start = find(haveReadList == 0);    %�ҵ��ӻ�û��ʼ�����Ľ��
                            
                            if isempty(start)                   %������е��ʶ��Ѿ��������������Ϣ
                                predictWordList = [];
                                predictInformation = {};
                                haveReadList = [];
                            else                                  %����δ�����ĵ���
                                start = start(end);               %�ҵ����һ��δ�����ĵ��ʣ������֮���������Ϣ
                                predictWordList(start+1:end) = [];
                                predictInformation(start+1:end,:) = [];
                                haveReadList(start+1:end) = [];
                            end
                        end
                        
                    end
                    %%
                    testTree;
                    
                    break;
                end
                
            end
        end
        fprintf('mean accuracy:%f\n',mean(seqAcc(1:seq,trailNum)));
        fprintf(fileID,'mean accuracy:%f\n',mean(seqAcc(1:seq,trailNum)));
    end
    
    t2=clock;
    runtime(disSegNum) = etime(t2,t1);
    fprintf('\n run-time:%f\n\n',runtime(disSegNum));
    fprintf('the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    fprintf(fileID,'the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    str(end-5:end)=[];
    save([str,'.mat'],'seqAcc');
end

