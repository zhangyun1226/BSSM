% clear;
% close all;
% load temporalout32.mat
%% test 替换部分单词
predictNoneThreshold = 0; %膜电压阈值，若低于，不让其预测出结果
endTime = DelayTime+dt;
minSegmentCon = 4;
targetsCell = {};
for seq = 1:seqenceLen
    tempSen = sequence(seq,1:sentenceLen);
    tempSen(tempSen == 0) = [];
    targetsCell{seq} = tempSen;
end

%% 测试序列的生成
trail = 5;
repNumList = 1:9;
for testRepN = 1:9
    repNum = repNumList(testRepN);
    seqenceNum = size(sequence,1);
    %% 存储正确率
    seqAcc = zeros(seqenceNum,trail);
    wordLen = size(sequence,2);
    maxSenLen = wordLen;
    wordsNum = max(sequence(:));
    for testN = 1:trail
        
        fprintf('Replace words %d, trail %d.\n', repNum,testN);
        
        testSeq = sequence;
        
        %% replace words
        for i = 1:seqenceNum
            currentSeq = testSeq(i,:);           %% 当前句子
            currentSeq(currentSeq == 0) = [];
            seqLen = length(currentSeq);         %% 当前句子的长度
            idx = sort(randperm(seqLen,repNum));%% 当前句子变化单词的位置
            newWords = randperm(wordsNum,repNum);
            currentSeq(idx) = newWords;
            currentSeq(end+1:wordLen) = -1;
            testSeq(i,:) = currentSeq;
        end
        
        
        
        
        %% 保存位置以及信息
        str = 'test_rep_';
%         str = [str,num2str(columnNum)];
        str = [str,num2str(repNum)];
        str = [str,'_'];
        str = [str,num2str(testN)];
        str = [str,'.txt'];
        fileID = fopen(str,'w');
        
        %% 存储第一个单词的信息
        FirstWordInfo = {};
        for i = 1:max(testSeq(:))
            FirstWordInfo{i,1} = {};
        end
        
        %% 测试开始
        for seq = 1:seqenceNum
            preState = zeros(columnNum,neuronNum,wordLen);%0:nT的预测状态
            outputInfo = {};                              %存储序列的信息
            
            for word = 1:wordLen                      %遍历当前句子的有输入的单词
                sample = testSeq(seq,word);            %获取具体的单词的样本
                if sample ~= -1                        %句子未结束
                    %% A.激活阶段
                    %% step1: 获取当前的激活柱和每个激活柱的输出
                    activecolumns = outputColumn(sample,:);      %当前的激活柱
                    startT = (word-1)*T;                    %属于哪个周期
                    currentTout = outputTime(sample,:) + startT;%激活柱的输出
                    
                    %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
                    activeNeurons = zeros(columnNum,neuronNum);%当前周期的激活神经元
                    if (word==1)
                        %% step2.1 第一个单词是没有预测神经元的，全激活
                        activeNeurons(activecolumns,:) = 1;%该柱所有神经元都处于激活状态
                    else
                        %% step2.1 其他单词，依次检测每个激活柱，查看其是否存在预测神经元
                        for column = 1:activeNum           %遍历激活柱
                            flagPredect = 0;               %当前柱是否有神经元被选为预测神经元，0无1有
                            currentColumn = activecolumns(column); %当前激活柱
                            preNeuron = find(preState(currentColumn,:,word-1)==1); %查看当前柱的哪些神经元处于预测状态
                            
                            %% step 2.2 存在预测神经元，激活所有预测神经元
                            if length(preNeuron) >=1
                                flagPredect = 1;                           %存在预测神经元
                                activeNeurons(currentColumn,preNeuron) = 1;%该神经元变为激活神经元
                            end
                            
                            %% step 2.3 不存在预测神经元，激活整个柱
                            if (flagPredect == 0)
                                %该柱所有神经元都处于激活状态
                                activeNeurons(currentColumn,:) = 1;
                            end
                        end
                    end
                    
                    %% 激活的神经元对应的输出
                    actIndex = find(activeNeurons == 1);                %激活神经元所对应的index
                    actTout = zeros(size(actIndex));                    %每个激活神经元的输出
                    [columnIdx,~] = ind2sub(size(segmentsNum),actIndex);%激活神经元所对应的柱
                    for i = 1:activeNum
                        idx = find(columnIdx == activecolumns(i));      %哪些神经元的输出是激活柱i的
                        actTout(idx,:) = currentTout(i);                %柱i的输出赋值给它
                    end
                    %% 将激活神经元/输出/以及样本保存到inputInfo中
                    if word == 1
                        inputInfo{1} = sample;
                        inputInfo{2} = {actIndex};
                        inputInfo{3} = {actTout};
                    else
                        inputInfo{1} = [inputInfo{1},sample];
                        inputInfo{2} = [inputInfo{2};{actIndex}];
                        inputInfo{3} = [inputInfo{3};{actTout}];
                    end
                    
                    %% B.预测阶段，若存在激活神经元
                    if word == 1 && ~isempty(FirstWordInfo{sample})  %若已经学习过，那么信息已经存储下来了，直接调用就行
                        predictState = FirstWordInfo{sample,1};      %预测状态
                        predictWordList = FirstWordInfo{sample,2};   %预测的单词列表
                        predictInformation = FirstWordInfo{sample,3};%预测的神经元以及它的输出
                    else
                        predictState = zeros(columnNum,neuronNum);     %每个神经元的预测状态，初始为0，无预测
                        neuronHasSegmentsIndexs = find(segmentsNum>0); %获取有segments的神经元
                        predictWordList = [];                          %预测的单词列表
                        predictInformation = {};                       %预测的神经元以及它的输出
                        hasContextNeurons = zeros(columnNum,neuronNum);%每个预测神经元是由哪些神经元导致的，初始为0，没有导致下一个的预测
                        %% step1:遍历所有segment,得出预测神经元
                        %% step1:找到与激活神经元相连的segment,得出预测神经元
                        %% 确定哪些远端树突参与远端计算
                        disSegNeu = [];                  % 计算哪些神经元有segment，
                        disSegNum = [];                  % 神经元的第几个segments接收上个周期传递过来的信息
                        disSegColumn = [];
                        %% 将这些神经元连接的下文segments加入计算列表
                        
                        connectedNeurons = [];           %哪些神经元与上个周期的神经元有连接
                        for tempi = 1:length(actIndex)
                            tempNeuron = actIndex(tempi); %获取上个周期的神经元
                            tempSegments = connectedSegmentList{tempNeuron};%与它连接的segments加入connectedNeurons
                            if ~isempty(tempSegments)
                                connectedNeurons = [connectedNeurons;tempSegments];
                            end
                        end
                        %% 去重（某个segment与多个神经元相连，所以会出现多次）
                        if ~isempty(connectedNeurons)
                            disSegNeu = connectedNeurons(:,1); %有远端树突的神经元
                            disSegNum = connectedNeurons(:,2); %远端树突的编号
                            [~,idx]= unique(disSegNeu*10000+disSegNum); %编码后去重
                            disSegNeu = disSegNeu(idx);
                            disSegNum = disSegNum(idx);
                        end
                        
                        for num = 1:length(disSegNeu)        %遍历所有有segments的神经元
                            neuronSegIndex = disSegNeu(num); %神经元的位置
                            segIndex = disSegNum(num);
                            segmentInfo = segmentsList{neuronSegIndex}{segIndex};       %获取该神经元的segments
                            segmentSynIdx = segmentInfo(:,1);             %权重连接的神经元的index
                            %% 比较segment连接的神经元和激活神经元的交集
                            diffNeurons = actIndex-segmentSynIdx';
                            [~,comIdx] = find(diffNeurons==0);
                            %% 判断该segments的交集是否大于阈值，若小于阈值，则不计算膜电压
                            if (length(comIdx) >= minSegmentCon)
                                commonNeurons = segmentSynIdx(comIdx); %获取公共神经元
                                %% step3 计算segment在下个周期的膜电压，膜电压最大的时候为其输出
                                %% 3.1 取出公共神经元的延迟和权重
                                delay = segmentInfo(comIdx,4);            %公共神经元对应的segment的延迟
                                weight = segmentInfo(comIdx,2);           %公共神经元对应的连接的权重
                                %% 3.2 取出公共神经元对应的柱号，取出对应的输出
                                Tout = [];                             %公共神经元对应的输出
                                [comColumns,~] = ind2sub(size(neuronConnectNum),commonNeurons);%公共神经元对应的激活柱
                                comIndex = zeros(size(comColumns));    %激活柱的位置初始化为0
                                for comi = 1:length(comColumns)
                                    comIndex(comi) = find(activecolumns == comColumns(comi));%获取公共柱对应的激活柱的位置
                                end
                                Tout= currentTout(comIndex);           %获取对应激活柱的输出
                                
                                %% 3.3 计算激活神经元输出currentTout的在下个周期的PSP，膜电压，输出
                                allInputs = Tout' + delay;             %输入加延迟的分布时间
                                simtime = min(allInputs)-dt:dt:max(allInputs)+endTime; %获取计算膜电压的时间分布
                                AllV = [];                             %膜电压

                                %% 3.4 计算膜电压
                                for t = simtime
                                    temp = t - allInputs;
                                    temp(temp<=0) = inf;
                                    PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                    V = weight'*PSP;
                                    AllV(end+1) = V;
                                end
                                %% 最大膜电压的时间就是其输出
                                maxV = max(AllV);                      %最大膜电压
                                [~,maxVT] = find(AllV == maxV);        %最大膜电压对应的时间
                                toutSeg = simtime(maxVT);              %获取输出时间
                                
                                %% 3.4 若有输出证明segments激活了，保存相关信息
                                if (length(toutSeg) == 1)              %若segments有输出
                                    predictState(neuronSegIndex) = 1;  %该神经元当前时刻变为预测状态
                                    hasContextNeurons(commonNeurons) = 1;%这些公共神经元有下文
                                    wordName = segmentInfo(comIdx,3);     %该这些公共神经元的预测单词标签
                                    if any( wordName ~= wordName(1))   %若存在噪声
                                        wordName = mode(wordName);     %少数服从多数
                                    else
                                        wordName =  wordName(1);       %标签
                                    end
                                    index = find(predictWordList == wordName); %查看这个单词是否被预测过
                                    if (isempty(index))                        %第一次出现这个预测单词
                                        predictWordList(end+1) = wordName;     %将该单词标签加入预测单词的列表
                                        predictInformation{length(predictWordList),1} = toutSeg; %预测神经元的输出保存
                                        predictInformation{length(predictWordList),2} = neuronSegIndex; %第2列装预测神经元的下标
                                    else                                       %非第一次出现预测的单词
                                        pridictIdx =  predictInformation{index,2};        %先判断先前是否加入过相同的预测神经元，不添加重复的信息
                                        if isempty(find(pridictIdx == neuronSegIndex))    %若没有
                                            preTout = predictInformation{index,1};     %找到之前存储预测单词的位置
                                            preTout(end+1,1:length(toutSeg)) = toutSeg;%预测的输出拼接上去
                                            preTout(preTout == 0) = simtime(end);      %多出的自动补最大时间
                                            predictInformation{index,1} = preTout;     %第1列装输出
                                            predictInformation{index,2} (end+1,:)= neuronSegIndex; %第2列装对应的神经元
                                        end
                                    end
                                else
                                    fprintf('多个输出');
                                    fprintf(fileID,'多个输出');
                                end
                            end
                        end
                        
                        
                        %% 存储第一个单词的信息
                        if word == 1
                            FirstWordInfo{sample,1} = predictState;
                            FirstWordInfo{sample,2} = predictWordList;
                            FirstWordInfo{sample,3} = predictInformation;
                        end
                    end
                    %% 存储预测信息
                    preState(:,:,word) = predictState;
                    
                else
                    %% 开始预测
                    fprintf('Test %d sentence: ',seq);
                    fprintf(fileID,'Test %d sentence: ',seq);
                    fprintf('%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf(fileID,'%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf('\n');
                    fprintf(fileID,'\n');
                    outputInfo = inputInfo;
                    
                    %% 输出整体的输出
                    %% 方案1：根据输入直接抑制那些不包含输入信息的目标神经元
                    predicStnIdx = [];
                    inputLen = length(find(testSeq(seq,:)~=-1));
                    for i = 1:size(outputInfo,1)
                        fprintf('part%4d:',i);
                        fprintf(fileID,'part%4d:',i);
                        info = outputInfo(i,:);
                        predictSentence = outputInfo{i,1};
                        fprintf(' %s',inputLablesCell{predictSentence});
                        fprintf(fileID,' %s',inputLablesCell{predictSentence});
                        
                        
                        haveGlobalFlag = 0;                 %局部是否激活目标神经元
                        targetV = zeros(seqenceNum,1);
                        minTout = Inf*ones(seqenceNum,1);
                        minSimility = 0.1;
                        for goalNeuron = 1:seqenceNum                     %遍历所有目标神经元
                            targetScentence = targetsCell{goalNeuron};  %目标神经元的单词序列
                            simility = length(intersect(predictSentence,targetScentence))/length(targetScentence);%比较预测的和目标句子相似度
                            
                            if simility >=minSimility
                                %存在相似度，就去判断一下是否激活
                                
                                tWeights = targetWeights(:,:,goalNeuron);%目标神经元和激活神经元相连的权重
                                tDelays = targetDelays(:,:,goalNeuron); %目标神经元和激活神经元相连的权重
                                
                                currentActiveNeurons = outputInfo{i,2};%激活神经元
                                currentTout = outputInfo{i,3};         %激活神经元的输出
                                
                                weights = [];  %存放激活神经元到输出的权重
                                delays = [];   %存放激活神经元到输出的延迟
                                Tout = [];     %存放激活神经元到输出的输出
                                
                                %% 目标神经元和激活神经元相连的连接，延迟，以及输出，预测的和输入的同样处理
                                for k = 1:length(currentActiveNeurons)
                                    idx = currentActiveNeurons{k};
                                    weights = [weights;tWeights(idx)];
                                    delays  = [delays;tDelays(idx)];
                                    Tout = [Tout;currentTout{k}];
                                end
                                AllV = [];
                                %% 选择模拟时间
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
%                             maxV = maxV - diffThresholdMany; %多个输出
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
                    
                    %% 和真实的相比的正确率
                    predicStnIdx = unique(sort(predicStnIdx)); %预测的句子
                    actualStnIdx = seq;
                    %% 计算正确率
                    
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


