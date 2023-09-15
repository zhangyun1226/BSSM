%% test �ṩ���ֵ���
predictNoneThreshold = 0.1; %Ĥ��ѹ��ֵ1�������ڣ�������Ԥ������
predictHasThreshold = 0.5;    %Ĥ��ѹ��ֵ2�������ڣ���Ԥ���������Զ��
diffThresholdMany = 0.05;   %����Ĥ��ѹ��ֵ1������ľ��Ӱ�����Щ
diffThresholdNone = 0.025;  %����Ĥ��ѹ��ֵ2������ľ��Ӱ�����Щ
winh = wi*0.5;              %������������Ϣ���ʵ�Ŀ����ӣ����ܵ�����
endTime = DelayTime+T_step;

%% �������е�����
testNum = 3;                  %�ṩ������ĸ���
testSeq = sequence;           %�������е�����������
testSeq(:,testNum+1:end) = -1;%����������������Ժ�����
seqenceNum = size(sequence,1); %�������е��ܾ���
testSeqNum = seqenceNum;      %�������е��ܾ���

%% ȥ���ظ�������
for i = 1:seqenceNum          %�������о���
    if i>testSeqNum
        break;                %����ظ����Եľ��ӱ�ɾ���󣬾����Ѿ���ĩβ��
    end
    currentSeq = testSeq(i,1:testNum); %��ȡ��ǰ��i�����ӵ�����
    delSeqIdx = [];                    %���ڴ���Ƿ��к���һ���ľ��ӣ������ľ�����Ҫ��ɾ��
    for j = i+1:testSeqNum             %�鿴��ǰ�����Ժ�����в��Ծ���
        compareSeq = testSeq(j,1:testNum);  %��ȡ����j
        if all(currentSeq - compareSeq == 0)%����������һ��һ��
            delSeqIdx(end+1) = j;           %j���Ӽ���ɾ������
        end
    end
    if length(delSeqIdx)~=0            %���о�����ɾ��������
        testSeqNum = testSeqNum - length(delSeqIdx); %���Ծ��������½�
        testSeq(delSeqIdx,:) = [];                   %ɾ���þ���
    end
end
maxSenLen = size(testSeq,2);  %���Ծ��ӵĵ��ʸ��� == testNum
wordLen = size(testSeq,2);    %���Ծ��ӵĵ��ʸ��� == testNum

%% �������λ���Լ���Ϣ
str = 'test_';
str = [str,num2str(testNum)];
str = [str,'.txt'];
fileID = fopen(str,'w');

%% �洢��һ�����ʵ���Ϣ����Ϊ��ȫ��������õ��������Ϊ�˼ӿ��ٶ�
FirstWordInfo = {};
for i = 1:max(testSeq(:))
    FirstWordInfo{i,1} = {};
end

%% �洢��ȷ��
seqAcc = zeros(testSeqNum,1);

%% ���Կ�ʼ
for seq = 1:testSeqNum                            %�������о���
    
    preState = zeros(columnNum,neuronNum,wordLen);%0:nT��Ԥ��״̬
    LearnState = zeros(columnNum,neuronNum,wordLen);
    outputInfo = {};                           %�洢������е���Ϣ
    
    for word = 1:wordLen                       %������ǰ���ӵ�������ĵ���
        sample = testSeq(seq,word);            %��ȡ����ĵ��ʵ�index
        if sample ~= -1                        %��������
            %% A.����׶�
            %% step1: ��ȡ��ǰ�ļ�������ÿ�������������
            activecolumns = columnCell(sample,:);      %��ǰ�ļ�����
            startT = (word-1)*Tmax;                    %�����ĸ�����
            currentTout = columnOut(sample,:) + startT;%�����������
            
            %% step2: ȷ����������ļ�����Ԫ������Ҫô�������У�Ҫô����Ԥ��״̬����Ԫ
            activeNeurons = zeros(columnNum,neuronNum);%������Ԫ
            if (word==1)
                %% step2.1 ��һ��������û��Ԥ����Ԫ�ģ�ȫ����
                activeNeurons(activecolumns,:) = 1;%����������Ԫ�����ڼ���״̬
            else
                %% step2.1 �������ʣ����μ��ÿ�����������鿴���Ƿ����Ԥ����Ԫ
                preLearnState = LearnState(:,:,word-1); %��һʱ��Ԥ�����Ԫ
                for column = 1:activeNum           %����������
                    flagPredect = 0;               %��ǰ���Ƿ�����Ԫ��ѡΪԤ����Ԫ��0��1��
                    currentColumn = activecolumns(column); %��ǰ������
                    preNeuron = find(preState(currentColumn,:,word-1)==1); %�鿴��ǰ������Щ��Ԫ����Ԥ��״̬
                    
                    %% step 2.2 ����Ԥ����Ԫ����������Ԥ����Ԫ
                    if length(preNeuron) >=1
                        flagPredect = 1;
                        activeNeurons(currentColumn,preNeuron) = 1;%����Ԫ��Ϊ������Ԫ
                    end
                    
                    %% step 2.3 ������Ԥ����Ԫ������������������������
                    if (flagPredect == 0)
                        %����������Ԫ�����ڼ���״̬
                        activeNeurons(currentColumn,:) = 1;
                    end
                    
                end
            end
            
            %% �������Ԫ��Ӧ�����
            actIndex = find(activeNeurons == 1);                %������Ԫ����Ӧ��index
            actTout = zeros(size(actIndex));                    %ÿ��������Ԫ�����
            [columnIdx,~] = ind2sub(size(segmentsNum),actIndex);%������Ԫ����Ӧ����
            for i = 1:activeNum
                idx = find(columnIdx == activecolumns(i));      %��Щ��Ԫ������Ǽ�����i��
                actTout(idx,:) = currentTout(i);                %��i�������ֵ����
            end
            %% ��������Ԫ/���/�Լ��������浽trainInfo��
            if word == 1
                trainInfo{1} = sample;
                trainInfo{2} = actIndex;
                trainInfo{3} = actTout;
            else
                trainInfo{1} = [trainInfo{1},sample];
                trainInfo{2} = [trainInfo{2};{actIndex}];
                trainInfo{3} = [trainInfo{3};{actTout}];
            end
            
            %% B.Ԥ��׶Σ������ڼ�����Ԫ
            if word == 1 && ~isempty(FirstWordInfo{sample}) %���Ѿ�ѧϰ������ô�Ѿ��洢�����ˣ�ֱ�ӵ��þ���
                predictState = FirstWordInfo{sample,1};     
                predictWordList = FirstWordInfo{sample,2};
                predictInformation = FirstWordInfo{sample,3};
            else
                predictState = zeros(columnNum,neuronNum);     %ÿ����Ԫ��Ԥ��״̬����ʼΪ0����Ԥ��
                neuronHasSegmentsIndexs = find(segmentsNum>0); %��ȡ��segments����Ԫ
                predictWordList = [];                          %Ԥ��ĵ����б�
                predictInformation = {};                       %Ԥ�����Ԫ�Լ��������
                hasContextNeurons = zeros(columnNum,neuronNum);%ÿ��Ԥ����Ԫ������Щ��Ԫ���µģ���ʼΪ0��û�е�����һ����Ԥ��
                %% step1:��������segment,�ó�Ԥ����Ԫ
                for num = 1:length(neuronHasSegmentsIndexs)        %����������segments����Ԫ
                    neuronSegIndex = neuronHasSegmentsIndexs(num); %��Ԫ��λ��
                    segments = segmentsList{neuronSegIndex};       %��ȡ����Ԫ��segments
                    
                    %% step2 �жϸ�segments�ͼ�����Ԫ���ཻ����Ԫû�У�С����ֵ�����ü���Ĥ��ѹ��
                    for segIndex = 1:segmentsNum(neuronSegIndex) %����Ԫsegments�ĸ���
                        
                        segmentW = segments{segIndex};    %��ǰsegment������Ȩ����Ϣ
                        %segmentlabel = segments{2,segIndex};%��ǰsegment�ı�ǩ
                        segmentSynIdx = segmentW(:,1);      %Ȩ�����ӵ���Ԫ��index
                        
                        %% �Ƚ�segment��Ȩ�ص���Ԫ�ͼ�����Ԫ�Ľ���
                        diffNeurons = actIndex-segmentSynIdx';
                        [~,comIdx] = find(diffNeurons==0);
                        
                        %% �жϸ�segments�Ľ����Ƿ������ֵ����С����ֵ���򲻼���Ĥ��ѹ
                        if (length(comIdx) >= segmentThreshold)
                            commonNeurons = segmentSynIdx(comIdx);
                            %% step3 ����segment���¸����ڵ�Ĥ��ѹ��Ĥ��ѹ����ʱ��Ϊ�����
                            %% 3.1 ȡ��������Ԫ���ӳٺ�Ȩ��
                            delay = segmentW(comIdx,4);              %����Ԫ��Ӧ��segment���ӳ�
                            weight = segmentW(comIdx,2);             %segment���ӵ�Ȩ��
                            
                            %% 3.2 ȡ��������Ԫ��Ӧ�����ţ�ȡ����Ӧ�����
                            Tout = [];
                            [comColumns,~] = ind2sub(size(neuronConnectNum),commonNeurons);
                            comIndex = zeros(size(comColumns));
                            for comi = 1:length(comColumns)
                                comIndex(comi) = find(activecolumns == comColumns(comi));
                            end
                            Tout= currentTout(comIndex);
                            
                            %% 3.3 ���㼤����Ԫ���currentTout�����¸����ڵ�PSP��Ĥ��ѹ�����
                            allInputs = Tout' + delay;
                            simtime = min(allInputs)-T_step:T_step:max(allInputs)+endTime; %�¸����ڵ�����ʱ��
                            AllV = [];
                            for t = simtime
                                %% input
                                temp = t - allInputs;
                                temp(temp<=0) = inf;
                                PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                V = weight'*PSP;
                                AllV(end+1) = V;
                            end
                            maxV = max(AllV);
                            [~,maxVT] = find(AllV == maxV);
                            toutSeg = simtime(maxVT);
                            
                            %% 3.4 �������֤��segments�����ˣ����������Ϣ
                            if (length(toutSeg) == 1)                          %��������
                                predictState(neuronSegIndex) = 1; %����Ԫ��ǰʱ�̱�ΪԤ��״̬
                                hasContextNeurons(commonNeurons) = 1;
                                wordName = segmentW(comIdx,3);               %��segment�ı�ǩ
                                if any( wordName ~= wordName(1))
                                    wordName = mode(wordName);
                                else
                                    wordName =  wordName(1);
                                end
                                index = find(predictWordList == wordName);     %�鿴��������Ƿ�Ԥ���
                                if (isempty(index)) %��һ�γ������Ԥ�ⵥ��
                                    predictWordList(end+1) = wordName;        %���õ��ʼ����б�
                                    predictInformation{length(predictWordList),1} = toutSeg; %��segment��������棬��1��װ���
                                    predictInformation{length(predictWordList),2} = neuronSegIndex; %��2��װ��Ӧ����Ԫ
                                else                %�ǵ�һ�γ���Ԥ��ĵ���
                                    pridictIdx =  predictInformation{index,2};
                                    if isempty(find(pridictIdx == neuronSegIndex))
                                        preTout = predictInformation{index,1};        %�ҵ�֮ǰ�洢��λ��
                                        preTout(end+1,1:length(toutSeg)) = toutSeg;%����ƴ����ȥ
                                        preTout(preTout == 0) = simtime(end);      %������Զ������ʱ��
                                        predictInformation{index,1} = preTout;        %��1��װ���
                                        predictInformation{index,2} (end+1,:)= neuronSegIndex; %��2��װ��Ӧ����Ԫ
                                    end
                                end
                            else
                                fprintf('������');
                                fprintf(fileID,'������');
                            end
                        end
                    end
                end
                
                %% �洢��һ�����ʵ���Ϣ
                if word == 1
                    FirstWordInfo{sample,1} = predictState;
                    FirstWordInfo{sample,2} = predictWordList;
                    FirstWordInfo{sample,3} = predictInformation;
                end
            end
            %             if size(FirstWordInfo,1)~=1
            %                 size(FirstWordInfo,1)
            %             end
            preState(:,:,word) = predictState;
            
        else
            %% ��ʼԤ��
            fprintf('Test %d sentence: ',seq);
            fprintf(fileID,'Test %d sentence: ',seq);
            fprintf('%s ',inputLablesCell{testSeq(seq,1:word-1)});
            fprintf(fileID,'%s ',inputLablesCell{testSeq(seq,1:word-1)});
            fprintf('\n');
            fprintf(fileID,'\n');
            
            %%     �ж�һ���Ƿ�ΪҶ�ӽ��
            hasContextNeurons = find(hasContextNeurons~=0);               %��Ԥ��ļ�����Ԫ
            
            if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)&&length(actIndex)~=activeNum*neuronNum%��Ԥ��ļ�����Ԫ����������Ԫ����ѵ����ֻһ�����ʣ���һ������ȫ���
                predicSegLen = size(outputInfo,1);
                outputInfo{predicSegLen+1,1} = trainInfo{1};%��1�д�þ������е��ʱ�ǩ
                outputInfo{predicSegLen+1,2} = trainInfo{2};%��2�д�þ������е��ʼ������Ԫ
                outputInfo{predicSegLen+1,3} = trainInfo{3};
            end
            
            %% ��ʼ����Ԥ��
            haveReadList = zeros(size(predictWordList));   %���ڴ洢�Ѿ��ù��ĵ���
            while (~isempty(predictWordList))              %�Ѿ��������ˣ�ջ��û���κ�Ԫ����
                %% ������ϵ�ջԪ��
                listLen = find(haveReadList == 0);         %����δ�ù��ĵ��ʵ�index
                
                %% ջ����������
                if(isempty(listLen))
                    predictWordList(listLen) = [];
                    predictInformation(listLen,:) = [];
                    haveReadList(listLen) = [];
                    break;
                end
                listLen = listLen(end);                    %�������Ǹ�
                
                currentWord = predictWordList(listLen);    %�����ڵĵ��ʱ�ǩ
                %                 inputLablesCell{currentWord}
                currentSegmentsCell = predictInformation(listLen,1:2);%�����ڵĵ��ʵ�Ԥ����Ԫ��Ϣ
                haveReadList(listLen) = 1;                            %�õ����Ѿ��ù���
                %% ��ֵ
                currentTout = currentSegmentsCell{1,1};    %��ǰ��Ԥ�ⵥ��������Ԫ��ʵ�����
                %% û������Ԥ��״̬��Ϊ��Ծ״̬
                actIndex = currentSegmentsCell{1,2};       %������Ԫ���ڵ�index
                if length(actIndex) < segmentThreshold
                    %% ȥ��������ʣ����������ᵼ��ջ��������
                    predictWordList(listLen) = [];
                    predictInformation(listLen,:) = [];
                    haveReadList(listLen) = [];
                    continue;
                end
                tempWordList = []; %% ��ǰ���ʵ�Ԥ�ⵥ��
                tempInformation = {}; %% ��ǰ���ʵ�Ԥ���segments
                hasContextNeurons = zeros(columnNum,neuronNum);
                %% step1:��������segment
                for num = 1:length(neuronHasSegmentsIndexs)
                    neuronSegIndex = neuronHasSegmentsIndexs(num);
                    segments = segmentsList{neuronSegIndex};%��ȡ��Ӧλ�õ�segments
                    %% step2 �жϸ�segments�ͼ�����Ԫ���ཻ����Ԫû�У�С����ֵ�����ü���Ĥ��ѹ��
                    for j = 1:segmentsNum(neuronSegIndex) %����Ԫsegments�ĸ�������Ϊsegments�еڶ����Ǳ�ǩ���������������õ��ӳ�
                        % ��ǰsegment������Ȩ��
                        segmentW = segments{1,j};
                        %segmentlabel = segments{2,j};
                        % Ȩ�ض�Ӧ����Ԫ
                        segmentSynIdx = segmentW(:,1);
                        %% �Ƚ�segment��Ȩ�ص���Ԫ�ͼ�����Ԫ�Ľ���
                        
                        %% ������ʽ�Ĺ�����Ԫ
                        diffNeurons = actIndex-segmentSynIdx';
                        [~,comIdx] = find(diffNeurons==0);
                        
                        
                        %�жϸ�segments�Ľ����Ƿ������ֵ����С����ֵ���򲻼���Ĥ��ѹ
                        if (length(comIdx) >= segmentThreshold)
                            commonNeurons = segmentSynIdx(comIdx);
                            %% step3 ����segment���¸����ڵ�Ĥ��ѹ��Ĥ��ѹ������ֵ�ĵ��
                            hasContextNeurons(commonNeurons) = 1;
                            %% 3.1 ȡ��������Ԫ���ӳ�
                            delay = segmentW(comIdx,4);              %����Ԫ��Ӧ��segment���ӳ�
                            weight = segmentW(comIdx,2);             %segment���ӵ�Ȩ��
                            
                            %% 3.2 ȡ��������Ԫ��Ӧ�����
                            %Ҫ��ɾ����index
                            delColumn = setdiff(actIndex,commonNeurons);
                            %�ҳ�Ҫɾ�������±�
                            delIndex = ismember(actIndex,delColumn);
                            %ɾ���������������ʣ�µľ��ǹ�����Ԫ�����
                            Tout = currentTout;          %���
                            Tout(delIndex==1) = [];
                            
                            %% 3.3 ���㼤����Ԫ���currentTout�����¸����ڵ�PSP��Ĥ��ѹ
                            %                             timePeriod = ceil((Tout+delay)/Tmax); %�¸����ڵ�ʱ��
                            %                             if max(timePeriod)-min(timePeriod)==0
                            %                                 simtime = timePeriod(1)*Tmax + testPeriodicTime;
                            %                             else
                            %                                 minT = min(timePeriod);
                            %                                 maxT = max(timePeriod);
                            %                                 simtime = minT*Tmax:T_step:(maxT+1.5)*Tmax;
                            %                             end
                            
                            allInputs = Tout + delay;
                            simtime = min(allInputs)-T_step:T_step:max(allInputs)+endTime;
                            
                            AllV = [];
                            for t = simtime
                                %% input
                                temp = t - Tout - delay;
                                temp(temp<=0) = inf;
                                PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                V = weight'*PSP;
                                AllV(end+1) = V;
                            end
                            maxV = max(AllV);
                            [~,maxVT] = find(AllV == maxV);
                            toutSeg = simtime(maxVT);
                            
                            if (length(toutSeg) == 1)           %��������
                                wordName = segmentW(comIdx,3);               %��segment�ı�ǩ
                                if any( wordName ~= wordName(1))
                                    wordName = mode(wordName);
                                else
                                    wordName =  wordName(1);
                                end
                                index = find(tempWordList == wordName);
                                if (isempty(index)) %��һ�γ����������
                                    tempWordList(end+1) = wordName;
                                    tempInformation{length(tempWordList),1} = toutSeg; %��1��װ���
                                    tempInformation{length(tempWordList),2} = neuronSegIndex; %��2��װ��Ӧ������Ԫ
                                else                %�ǵ�һ�γ���
                                    pridictIdx =  tempInformation{index,2};
                                    if isempty(find(pridictIdx == neuronSegIndex))
                                        preTout = tempInformation{index,1};        %�ҵ�֮ǰ�洢��λ��
                                        preTout(end+1,1:length(toutSeg)) = toutSeg;%����ƴ����ȥ
                                        preTout(preTout == 0) = simtime(end);      %������Զ������ʱ��
                                        tempInformation{index,1} = preTout;        %��1��װ���
                                        tempInformation{index,2} (end+1,:)= neuronSegIndex; %��2��װ��Ӧ����Ԫ
                                    end
                                end
                            else
                                fprintf('Ĥ��ѹ����û�����\n');
                                fprintf(fileID,'Ĥ��ѹ����û�����\n');
                            end
                        end
                    end
                end
                %% step3 �ж��Ƿ�ΪҶ�ӽ��Ҷ�ӽ��1���жϵ�ǰ�Ƿ��м������Ԫû��Ԥ��ģ��������ӽ����������б�
                leaf = 0;
                hasContextNeurons = find(hasContextNeurons~=0);                %��Ԥ��ļ�����Ԫ
                %Ԥ��ļ�����Ԫ<��ǰ�ļ�����Ԫ���Ҳ���ȫ����ĵ��ʵ��µ�
                if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)
                    leaf = 1;
                end
                %% �жϵ�ǰ��û���µ�Ԥ�⣬���У���ô����ջ����û�У���ô��Ҷ�ӽ��
                nextLen = length(tempWordList);
                if nextLen > 0 %%��Ҷ�ӽ��
                    predictWordList(end+1:end+nextLen) = tempWordList;
                    predictInformation(end+1:end+nextLen,:) = tempInformation;
                    haveReadList(end+1:end+nextLen) = 0;
                else
                    leaf = 1; %% ��Ԥ��,Ҷ�ӽ��2�����嶼û�����
                end
                if leaf == 1
                    %% ���� �������дʵı�ǩ
                    wordList = find(haveReadList == 1);
                    
                    predicSegLen = size(outputInfo,1);
                    outputInfo{predicSegLen+1,1} = [trainInfo{1},predictWordList(wordList)];%��1�д�þ������е��ʱ�ǩ
                    outputInfo{predicSegLen+1,2} = [trainInfo{2};predictInformation(wordList,2)];%��2�д�þ������е��ʼ������Ԫ
                    outputInfo{predicSegLen+1,3} = [trainInfo{3};predictInformation(wordList,1)];%��3�м������Ԫ�����
                    %% ɾ���þ��������Ϣ
                    start = find(haveReadList == 0); %%�ӻ�û��ʼ�����Ľ��
                    
                    if isempty(start)
                        predictWordList = [];
                        predictInformation = {};
                        haveReadList = [];
                    else
                        start = start(end);
                        predictWordList(start+1:end) = [];
                        predictInformation(start+1:end,:) = [];
                        haveReadList(start+1:end) = [];
                    end
                end
                
            end
            
            
            %% �����������
            %% ����1����������ֱ��������Щ������������Ϣ��Ŀ����Ԫ
            predicStnIdx = [];
            inputLen = length(find(testSeq(seq,:)~=-1));
            for i = 1:size(outputInfo,1)
                fprintf('part%4d:',i);
                fprintf(fileID,'part%4d:',i);
                info = outputInfo(i,:);
                predictSentence = outputInfo{i,1};
                fprintf(' %s',inputLablesCell{predictSentence});
                fprintf(fileID,' %s',inputLablesCell{predictSentence});
                
                
                haveGlobalFlag = 0;                 %�ֲ��Ƿ񼤻�Ŀ����Ԫ
                targetV = zeros(seqenceNum,1);
                minTout = Inf*ones(seqenceNum,1);
                minSimility = length(unique(predictSentence))/maxSenLen;
                for goalNeuron = 1:seqenceNum                     %��������Ŀ����Ԫ
                    targetScentence = targetsCell{goalNeuron,3};  %Ŀ����Ԫ�ĵ�������
                    simility = length(intersect(predictSentence,targetScentence))/length(targetScentence);%�Ƚ�Ԥ��ĺ�Ŀ��������ƶ�
                    
                    if simility >=minSimility
                        %�������ƶȣ���ȥ�ж�һ���Ƿ񼤻�
                        
                        targetWeights = targetsCell{goalNeuron,1};%Ŀ����Ԫ�ͼ�����Ԫ������Ȩ��
                        targetDelays = targetsCell{goalNeuron,2}; %Ŀ����Ԫ�ͼ�����Ԫ������Ȩ��
                        
                        currentActiveNeurons = outputInfo{i,2};%������Ԫ
                        currentTout = outputInfo{i,3};         %������Ԫ�����
                        
                        weights = [];  %��ż�����Ԫ�������Ȩ��
                        delays = [];   %��ż�����Ԫ��������ӳ�
                        Tout = [];     %��ż�����Ԫ����������
                        
                        %% Ŀ����Ԫ�ͼ�����Ԫ���������ӣ��ӳ٣��Լ������Ԥ��ĺ������ͬ������
                        for k = 1:length(currentActiveNeurons)
                            idx = currentActiveNeurons{k};
                            weights = [weights;targetWeights(idx)];
                            delays  = [delays;targetDelays(idx)];
                            Tout = [Tout;currentTout{k}];
                        end
                        
                        AllV = [];
                        %% ѡ��ģ��ʱ��
                        allInputs = Tout + delays;
                        allInputs(allInputs ==Inf) = [];
                        sortInputs = sort(round(allInputs));
                        simtime = min(sortInputs):T_step:max(sortInputs)+endTime;
                        %toutSeg = Inf;
                        if isempty(simtime)
                            AllV = 0;
                        else
                            for t = simtime
                                %% input
                                temp = t - Tout - delays;
                                temp(temp<=0) = inf;
                                PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                %                             ref = Theta*sum(-exp((toutSeg-t)/Tau_a),2);
                                V = weights'*PSP;
                                AllV(end+1) = V;
                                %                             if (V >= targetThreshold)
                                %                                 haveGlobalFlag = haveGlobalFlag+1;
                                %                                 toutSeg = t;
                                %                                 break;
                                %                             end
                            end
                        end
                        targetV(goalNeuron) = max(AllV);
                        %                         minTout(goalNeuron) = toutSeg;
                    end
                end
                maxV = max(targetV);
                if maxV < predictNoneThreshold
                    fprintf('\n \n');
                    fprintf(fileID,'\n \n');
                    continue;
                elseif maxV >= predictHasThreshold
                    maxV = maxV - diffThresholdMany; %������
                else
                    maxV = maxV - diffThresholdNone;
                end
                idx = find(targetV >= maxV);
                targetScentence = targetsCell(idx,3);
                predicStnIdx(end+1:end+length(idx)) = idx;
                fprintf('\n');
                fprintf(fileID,'\n');
                for sentenceNum = 1:length(idx)
                    fprintf('out%5d:',idx(sentenceNum));
                    fprintf(fileID,'out%5d:',idx(sentenceNum));
                    fprintf(' %s',inputLablesCell{targetScentence{sentenceNum}});
                    fprintf(fileID,' %s',inputLablesCell{targetScentence{sentenceNum}});
                    fprintf('\n');
                    fprintf(fileID,'\n');
                end
                fprintf('\n');
                fprintf(fileID,'\n');
            end
            
            %% ����ʵ����ȵ���ȷ��
            predicStnIdx = unique(sort(predicStnIdx)); %Ԥ��ľ���
            actualStnIdx = [];
            subsequence = testSeq(seq,:);       %��ǰ������
            subsequence(subsequence == -1) = []; %ȥ��-1���ǲ���
            subLen = length(subsequence);       %��ʾ�����еĳ���
            for goalNeuron = 1:seqenceNum                     %��������Ŀ����Ԫ
                targetScentence = targetsCell{goalNeuron,3};  %Ŀ����Ԫ�ĵ�������
                targetLen = length(targetScentence);          %��ʾĿ�����еĳ���
                firstIdx = find(targetScentence == subsequence(1));%��һ��������Ŀ�������е�λ��
                
                if (subLen>targetLen || isempty(firstIdx))    %����Ŀ���������
                    continue;
                end
                
                for i = 1:length(firstIdx) %������һ��������Ŀ�������е�λ��
                    targetIdx = firstIdx(i)+1; %��ȡ��2�����ʵ�λ��
                    rightFlag = 1; %����þ�����
                    for subIdx = 2:subLen  %�ӵڶ������ʿ�ʼ���Ƿ�ƥ��������
                        if targetIdx<=targetLen && subsequence(subIdx) == targetScentence(targetIdx)  %��Ŀ�굥�ʺ�������һ��
                            targetIdx = targetIdx+1;
                        else %����һ����һ�£���ô�Ͳ���
                            rightFlag = 0;
                            break;%����������
                        end
                    end
                    if rightFlag == 1
                        actualStnIdx(end+1) = goalNeuron;
                        break;
                    end
                end
            end
            %% ������ȷ��
            
            accuracyIdx = find(actualStnIdx-predicStnIdx'==0);
            accuracyLen = length(accuracyIdx);
            unionLen = length(actualStnIdx) + length(predicStnIdx) - accuracyLen;
            seqAcc(seq) = accuracyLen/unionLen;
            
            fprintf('predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
            fprintf(fileID,'predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
            fprintf('current accuracy:%f, mean accuracy:%f\n',seqAcc(seq),mean(seqAcc(1:seq)));
            fprintf(fileID,'current accuracy:%f, mean accuracy:%f\n',seqAcc(seq),mean(seqAcc(1:seq)));
            
            if seqAcc(seq)~=1
                if accuracyLen < length(actualStnIdx)
                    diffData = setdiff(actualStnIdx,predicStnIdx);
                    fprintf('absence:');
                    fprintf(fileID,'absence:');
                    for i = 1:length(diffData)
                        fprintf('%d ',diffData(i));
                        fprintf(fileID,'%d ',diffData(i));
                    end
                end
                
                if length(predicStnIdx) > length(actualStnIdx)
                    fprintf('\nsurplus:');
                    fprintf(fileID,'\nsurplus:');
                    diffData = setdiff(predicStnIdx,actualStnIdx);
                    diffData = sort(diffData);
                    for i = 1:length(diffData)
                        fprintf('%d ',diffData(i));
                        fprintf(fileID,'%d ',diffData(i));
                    end
                end
                fprintf('\n');
                fprintf(fileID,'\n');
            end
            break;
        end
        
    end
    fprintf('\n');
    fprintf(fileID,'\n');
end
str(end-3:end)=[];
save([str,'.mat'],'seqAcc');

