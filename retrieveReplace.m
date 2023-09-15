% clear;
% close all;
% load temporalout32.mat
%% test �滻���ֵ���
predictNoneThreshold = 0; %Ĥ��ѹ��ֵ�������ڣ�������Ԥ������
endTime = DelayTime+dt;
minSegmentCon = 4;
targetsCell = {};
for seq = 1:seqenceLen
    tempSen = sequence(seq,1:sentenceLen);
    tempSen(tempSen == 0) = [];
    targetsCell{seq} = tempSen;
end

%% �������е�����
trail = 5;
repNumList = 1:9;
for testRepN = 1:9
    repNum = repNumList(testRepN);
    seqenceNum = size(sequence,1);
    %% �洢��ȷ��
    seqAcc = zeros(seqenceNum,trail);
    wordLen = size(sequence,2);
    maxSenLen = wordLen;
    wordsNum = max(sequence(:));
    for testN = 1:trail
        
        fprintf('Replace words %d, trail %d.\n', repNum,testN);
        
        testSeq = sequence;
        
        %% replace words
        for i = 1:seqenceNum
            currentSeq = testSeq(i,:);           %% ��ǰ����
            currentSeq(currentSeq == 0) = [];
            seqLen = length(currentSeq);         %% ��ǰ���ӵĳ���
            idx = sort(randperm(seqLen,repNum));%% ��ǰ���ӱ仯���ʵ�λ��
            newWords = randperm(wordsNum,repNum);
            currentSeq(idx) = newWords;
            currentSeq(end+1:wordLen) = -1;
            testSeq(i,:) = currentSeq;
        end
        
        
        
        
        %% ����λ���Լ���Ϣ
        str = 'test_rep_';
%         str = [str,num2str(columnNum)];
        str = [str,num2str(repNum)];
        str = [str,'_'];
        str = [str,num2str(testN)];
        str = [str,'.txt'];
        fileID = fopen(str,'w');
        
        %% �洢��һ�����ʵ���Ϣ
        FirstWordInfo = {};
        for i = 1:max(testSeq(:))
            FirstWordInfo{i,1} = {};
        end
        
        %% ���Կ�ʼ
        for seq = 1:seqenceNum
            preState = zeros(columnNum,neuronNum,wordLen);%0:nT��Ԥ��״̬
            outputInfo = {};                              %�洢���е���Ϣ
            
            for word = 1:wordLen                      %������ǰ���ӵ�������ĵ���
                sample = testSeq(seq,word);            %��ȡ����ĵ��ʵ�����
                if sample ~= -1                        %����δ����
                    %% A.����׶�
                    %% step1: ��ȡ��ǰ�ļ�������ÿ�������������
                    activecolumns = outputColumn(sample,:);      %��ǰ�ļ�����
                    startT = (word-1)*T;                    %�����ĸ�����
                    currentTout = outputTime(sample,:) + startT;%�����������
                    
                    %% step2: ȷ����������ļ�����Ԫ������Ҫô�������У�Ҫô����Ԥ��״̬����Ԫ
                    activeNeurons = zeros(columnNum,neuronNum);%��ǰ���ڵļ�����Ԫ
                    if (word==1)
                        %% step2.1 ��һ��������û��Ԥ����Ԫ�ģ�ȫ����
                        activeNeurons(activecolumns,:) = 1;%����������Ԫ�����ڼ���״̬
                    else
                        %% step2.1 �������ʣ����μ��ÿ�����������鿴���Ƿ����Ԥ����Ԫ
                        for column = 1:activeNum           %����������
                            flagPredect = 0;               %��ǰ���Ƿ�����Ԫ��ѡΪԤ����Ԫ��0��1��
                            currentColumn = activecolumns(column); %��ǰ������
                            preNeuron = find(preState(currentColumn,:,word-1)==1); %�鿴��ǰ������Щ��Ԫ����Ԥ��״̬
                            
                            %% step 2.2 ����Ԥ����Ԫ����������Ԥ����Ԫ
                            if length(preNeuron) >=1
                                flagPredect = 1;                           %����Ԥ����Ԫ
                                activeNeurons(currentColumn,preNeuron) = 1;%����Ԫ��Ϊ������Ԫ
                            end
                            
                            %% step 2.3 ������Ԥ����Ԫ������������
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
                    %% ��������Ԫ/���/�Լ��������浽inputInfo��
                    if word == 1
                        inputInfo{1} = sample;
                        inputInfo{2} = {actIndex};
                        inputInfo{3} = {actTout};
                    else
                        inputInfo{1} = [inputInfo{1},sample];
                        inputInfo{2} = [inputInfo{2};{actIndex}];
                        inputInfo{3} = [inputInfo{3};{actTout}];
                    end
                    
                    %% B.Ԥ��׶Σ������ڼ�����Ԫ
                    if word == 1 && ~isempty(FirstWordInfo{sample})  %���Ѿ�ѧϰ������ô��Ϣ�Ѿ��洢�����ˣ�ֱ�ӵ��þ���
                        predictState = FirstWordInfo{sample,1};      %Ԥ��״̬
                        predictWordList = FirstWordInfo{sample,2};   %Ԥ��ĵ����б�
                        predictInformation = FirstWordInfo{sample,3};%Ԥ�����Ԫ�Լ��������
                    else
                        predictState = zeros(columnNum,neuronNum);     %ÿ����Ԫ��Ԥ��״̬����ʼΪ0����Ԥ��
                        neuronHasSegmentsIndexs = find(segmentsNum>0); %��ȡ��segments����Ԫ
                        predictWordList = [];                          %Ԥ��ĵ����б�
                        predictInformation = {};                       %Ԥ�����Ԫ�Լ��������
                        hasContextNeurons = zeros(columnNum,neuronNum);%ÿ��Ԥ����Ԫ������Щ��Ԫ���µģ���ʼΪ0��û�е�����һ����Ԥ��
                        %% step1:��������segment,�ó�Ԥ����Ԫ
                        %% step1:�ҵ��뼤����Ԫ������segment,�ó�Ԥ����Ԫ
                        %% ȷ����ЩԶ����ͻ����Զ�˼���
                        disSegNeu = [];                  % ������Щ��Ԫ��segment��
                        disSegNum = [];                  % ��Ԫ�ĵڼ���segments�����ϸ����ڴ��ݹ�������Ϣ
                        disSegColumn = [];
                        %% ����Щ��Ԫ���ӵ�����segments��������б�
                        
                        connectedNeurons = [];           %��Щ��Ԫ���ϸ����ڵ���Ԫ������
                        for tempi = 1:length(actIndex)
                            tempNeuron = actIndex(tempi); %��ȡ�ϸ����ڵ���Ԫ
                            tempSegments = connectedSegmentList{tempNeuron};%�������ӵ�segments����connectedNeurons
                            if ~isempty(tempSegments)
                                connectedNeurons = [connectedNeurons;tempSegments];
                            end
                        end
                        %% ȥ�أ�ĳ��segment������Ԫ���������Ի���ֶ�Σ�
                        if ~isempty(connectedNeurons)
                            disSegNeu = connectedNeurons(:,1); %��Զ����ͻ����Ԫ
                            disSegNum = connectedNeurons(:,2); %Զ����ͻ�ı��
                            [~,idx]= unique(disSegNeu*10000+disSegNum); %�����ȥ��
                            disSegNeu = disSegNeu(idx);
                            disSegNum = disSegNum(idx);
                        end
                        
                        for num = 1:length(disSegNeu)        %����������segments����Ԫ
                            neuronSegIndex = disSegNeu(num); %��Ԫ��λ��
                            segIndex = disSegNum(num);
                            segmentInfo = segmentsList{neuronSegIndex}{segIndex};       %��ȡ����Ԫ��segments
                            segmentSynIdx = segmentInfo(:,1);             %Ȩ�����ӵ���Ԫ��index
                            %% �Ƚ�segment���ӵ���Ԫ�ͼ�����Ԫ�Ľ���
                            diffNeurons = actIndex-segmentSynIdx';
                            [~,comIdx] = find(diffNeurons==0);
                            %% �жϸ�segments�Ľ����Ƿ������ֵ����С����ֵ���򲻼���Ĥ��ѹ
                            if (length(comIdx) >= minSegmentCon)
                                commonNeurons = segmentSynIdx(comIdx); %��ȡ������Ԫ
                                %% step3 ����segment���¸����ڵ�Ĥ��ѹ��Ĥ��ѹ����ʱ��Ϊ�����
                                %% 3.1 ȡ��������Ԫ���ӳٺ�Ȩ��
                                delay = segmentInfo(comIdx,4);            %������Ԫ��Ӧ��segment���ӳ�
                                weight = segmentInfo(comIdx,2);           %������Ԫ��Ӧ�����ӵ�Ȩ��
                                %% 3.2 ȡ��������Ԫ��Ӧ�����ţ�ȡ����Ӧ�����
                                Tout = [];                             %������Ԫ��Ӧ�����
                                [comColumns,~] = ind2sub(size(neuronConnectNum),commonNeurons);%������Ԫ��Ӧ�ļ�����
                                comIndex = zeros(size(comColumns));    %��������λ�ó�ʼ��Ϊ0
                                for comi = 1:length(comColumns)
                                    comIndex(comi) = find(activecolumns == comColumns(comi));%��ȡ��������Ӧ�ļ�������λ��
                                end
                                Tout= currentTout(comIndex);           %��ȡ��Ӧ�����������
                                
                                %% 3.3 ���㼤����Ԫ���currentTout�����¸����ڵ�PSP��Ĥ��ѹ�����
                                allInputs = Tout' + delay;             %������ӳٵķֲ�ʱ��
                                simtime = min(allInputs)-dt:dt:max(allInputs)+endTime; %��ȡ����Ĥ��ѹ��ʱ��ֲ�
                                AllV = [];                             %Ĥ��ѹ

                                %% 3.4 ����Ĥ��ѹ
                                for t = simtime
                                    temp = t - allInputs;
                                    temp(temp<=0) = inf;
                                    PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                    V = weight'*PSP;
                                    AllV(end+1) = V;
                                end
                                %% ���Ĥ��ѹ��ʱ����������
                                maxV = max(AllV);                      %���Ĥ��ѹ
                                [~,maxVT] = find(AllV == maxV);        %���Ĥ��ѹ��Ӧ��ʱ��
                                toutSeg = simtime(maxVT);              %��ȡ���ʱ��
                                
                                %% 3.4 �������֤��segments�����ˣ����������Ϣ
                                if (length(toutSeg) == 1)              %��segments�����
                                    predictState(neuronSegIndex) = 1;  %����Ԫ��ǰʱ�̱�ΪԤ��״̬
                                    hasContextNeurons(commonNeurons) = 1;%��Щ������Ԫ������
                                    wordName = segmentInfo(comIdx,3);     %����Щ������Ԫ��Ԥ�ⵥ�ʱ�ǩ
                                    if any( wordName ~= wordName(1))   %����������
                                        wordName = mode(wordName);     %�������Ӷ���
                                    else
                                        wordName =  wordName(1);       %��ǩ
                                    end
                                    index = find(predictWordList == wordName); %�鿴��������Ƿ�Ԥ���
                                    if (isempty(index))                        %��һ�γ������Ԥ�ⵥ��
                                        predictWordList(end+1) = wordName;     %���õ��ʱ�ǩ����Ԥ�ⵥ�ʵ��б�
                                        predictInformation{length(predictWordList),1} = toutSeg; %Ԥ����Ԫ���������
                                        predictInformation{length(predictWordList),2} = neuronSegIndex; %��2��װԤ����Ԫ���±�
                                    else                                       %�ǵ�һ�γ���Ԥ��ĵ���
                                        pridictIdx =  predictInformation{index,2};        %���ж���ǰ�Ƿ�������ͬ��Ԥ����Ԫ��������ظ�����Ϣ
                                        if isempty(find(pridictIdx == neuronSegIndex))    %��û��
                                            preTout = predictInformation{index,1};     %�ҵ�֮ǰ�洢Ԥ�ⵥ�ʵ�λ��
                                            preTout(end+1,1:length(toutSeg)) = toutSeg;%Ԥ������ƴ����ȥ
                                            preTout(preTout == 0) = simtime(end);      %������Զ������ʱ��
                                            predictInformation{index,1} = preTout;     %��1��װ���
                                            predictInformation{index,2} (end+1,:)= neuronSegIndex; %��2��װ��Ӧ����Ԫ
                                        end
                                    end
                                else
                                    fprintf('������');
                                    fprintf(fileID,'������');
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
                    %% �洢Ԥ����Ϣ
                    preState(:,:,word) = predictState;
                    
                else
                    %% ��ʼԤ��
                    fprintf('Test %d sentence: ',seq);
                    fprintf(fileID,'Test %d sentence: ',seq);
                    fprintf('%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf(fileID,'%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf('\n');
                    fprintf(fileID,'\n');
                    outputInfo = inputInfo;
                    
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
                        minSimility = 0.1;
                        for goalNeuron = 1:seqenceNum                     %��������Ŀ����Ԫ
                            targetScentence = targetsCell{goalNeuron};  %Ŀ����Ԫ�ĵ�������
                            simility = length(intersect(predictSentence,targetScentence))/length(targetScentence);%�Ƚ�Ԥ��ĺ�Ŀ��������ƶ�
                            
                            if simility >=minSimility
                                %�������ƶȣ���ȥ�ж�һ���Ƿ񼤻�
                                
                                tWeights = targetWeights(:,:,goalNeuron);%Ŀ����Ԫ�ͼ�����Ԫ������Ȩ��
                                tDelays = targetDelays(:,:,goalNeuron); %Ŀ����Ԫ�ͼ�����Ԫ������Ȩ��
                                
                                currentActiveNeurons = outputInfo{i,2};%������Ԫ
                                currentTout = outputInfo{i,3};         %������Ԫ�����
                                
                                weights = [];  %��ż�����Ԫ�������Ȩ��
                                delays = [];   %��ż�����Ԫ��������ӳ�
                                Tout = [];     %��ż�����Ԫ����������
                                
                                %% Ŀ����Ԫ�ͼ�����Ԫ���������ӣ��ӳ٣��Լ������Ԥ��ĺ������ͬ������
                                for k = 1:length(currentActiveNeurons)
                                    idx = currentActiveNeurons{k};
                                    weights = [weights;tWeights(idx)];
                                    delays  = [delays;tDelays(idx)];
                                    Tout = [Tout;currentTout{k}];
                                end
                                AllV = [];
                                %% ѡ��ģ��ʱ��
                                allInputs = Tout + delays;
                                allInputs(allInputs ==Inf) = [];
                                allPeriods = floor(allInputs/T);
                                mostPeriod = mode(allPeriods);
                                simtime = mostPeriod*T:dt:(mostPeriod+1)*T;
%                                 sortInputs = sort(round(allInputs));
%                                 simtime = min(sortInputs):nt:max(sortInputs)+endTime;
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
%                         elseif maxV >= predictHasThreshold
%                             maxV = maxV - diffThresholdMany; %������
%                         else
%                             maxV = maxV - diffThresholdNone;
                        end
                        idx = find(targetV >= maxV);
                        targetScentence = targetsCell(idx);
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
                    actualStnIdx = seq;
                    %% ������ȷ��
                    
                    accuracyIdx = find(actualStnIdx-predicStnIdx'==0);
                    accuracyLen = length(accuracyIdx);
                    unionLen = length(actualStnIdx) + length(predicStnIdx) - accuracyLen;
                    seqAcc(seq,testN) = accuracyLen/unionLen;
                    
                    fprintf('predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
                    fprintf(fileID,'predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
                    fprintf('current accuracy:%f, mean accuracy:%f\n',seqAcc(seq,testN),mean(seqAcc(1:seq,testN)));
                    fprintf(fileID,'current accuracy:%f, mean accuracy:%f\n',seqAcc(seq,testN),mean(seqAcc(1:seq,testN)));
                    
                    if seqAcc(seq,testN)~=1
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
    end
    fprintf('the mean accuracy of 5 trails:%f\n',mean(mean(seqAcc)));
    fprintf(fileID,'the mean accuracy of 5 trails:%f\n',mean(mean(seqAcc)));
    str(end-5:end)=[];
    save([str,'.mat'],'seqAcc');
end


