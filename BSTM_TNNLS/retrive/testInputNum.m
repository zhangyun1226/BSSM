%% test 提供部分单词
predictNoneThreshold = 0.1; %膜电压阈值1，若低于，不让其预测出结果
predictHasThreshold = 0.5;    %膜电压阈值2，若高于，其预测出结果可以多个
diffThresholdMany = 0.05;   %超过膜电压阈值1，输出的句子包括哪些
diffThresholdNone = 0.025;  %超过膜电压阈值2，输出的句子包括哪些
winh = wi*0.5;              %不包括输入信息单词的目标句子，会受到抑制
endTime = DelayTime+T_step;

%% 测试序列的生成
testNum = 3;                  %提供的输入的个数
testSeq = sequence;           %测试序列等于输入序列
testSeq(:,testNum+1:end) = -1;%测试序列输入个数以后的清空
seqenceNum = size(sequence,1); %输入序列的总句数
testSeqNum = seqenceNum;      %测试序列的总句数

%% 去除重复的输入
for i = 1:seqenceNum          %遍历所有句子
    if i>testSeqNum
        break;                %如果重复测试的句子被删除后，句数已经到末尾了
    end
    currentSeq = testSeq(i,1:testNum); %获取当前第i个句子的序列
    delSeqIdx = [];                    %用于存放是否有和它一样的句子，这样的句子需要被删除
    for j = i+1:testSeqNum             %查看当前句子以后的所有测试句子
        compareSeq = testSeq(j,1:testNum);  %获取句子j
        if all(currentSeq - compareSeq == 0)%若两个句子一摸一样
            delSeqIdx(end+1) = j;           %j句子加入删除序列
        end
    end
    if length(delSeqIdx)~=0            %若有句子在删除序列中
        testSeqNum = testSeqNum - length(delSeqIdx); %测试句子总数下降
        testSeq(delSeqIdx,:) = [];                   %删除该句子
    end
end
maxSenLen = size(testSeq,2);  %测试句子的单词个数 == testNum
wordLen = size(testSeq,2);    %测试句子的单词个数 == testNum

%% 保存输出位置以及信息
str = 'test_';
str = [str,num2str(testNum)];
str = [str,'.txt'];
fileID = fopen(str,'w');

%% 存储第一个单词的信息，因为是全激活，会多次用到，这块是为了加快速度
FirstWordInfo = {};
for i = 1:max(testSeq(:))
    FirstWordInfo{i,1} = {};
end

%% 存储正确率
seqAcc = zeros(testSeqNum,1);

%% 测试开始
for seq = 1:testSeqNum                            %测试所有句子
    
    preState = zeros(columnNum,neuronNum,wordLen);%0:nT的预测状态
    LearnState = zeros(columnNum,neuronNum,wordLen);
    outputInfo = {};                           %存储输出序列的信息
    
    for word = 1:wordLen                       %遍历当前句子的有输入的单词
        sample = testSeq(seq,word);            %获取具体的单词的index
        if sample ~= -1                        %若单词是
            %% A.激活阶段
            %% step1: 获取当前的激活柱和每个激活柱的输出
            activecolumns = columnCell(sample,:);      %当前的激活柱
            startT = (word-1)*Tmax;                    %属于哪个周期
            currentTout = columnOut(sample,:) + startT;%激活柱的输出
            
            %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
            activeNeurons = zeros(columnNum,neuronNum);%激活神经元
            if (word==1)
                %% step2.1 第一个单词是没有预测神经元的，全激活
                activeNeurons(activecolumns,:) = 1;%该柱所有神经元都处于激活状态
            else
                %% step2.1 其他单词，依次检测每个激活柱，查看其是否存在预测神经元
                preLearnState = LearnState(:,:,word-1); %上一时刻预测的神经元
                for column = 1:activeNum           %遍历激活柱
                    flagPredect = 0;               %当前柱是否有神经元被选为预测神经元，0无1有
                    currentColumn = activecolumns(column); %当前激活柱
                    preNeuron = find(preState(currentColumn,:,word-1)==1); %查看当前柱的哪些神经元处于预测状态
                    
                    %% step 2.2 存在预测神经元，激活所有预测神经元
                    if length(preNeuron) >=1
                        flagPredect = 1;
                        activeNeurons(currentColumn,preNeuron) = 1;%该神经元变为激活神经元
                    end
                    
                    %% step 2.3 不存在预测神经元，激活整个柱，不存在这项
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
            %% 将激活神经元/输出/以及样本保存到trainInfo中
            if word == 1
                trainInfo{1} = sample;
                trainInfo{2} = actIndex;
                trainInfo{3} = actTout;
            else
                trainInfo{1} = [trainInfo{1},sample];
                trainInfo{2} = [trainInfo{2};{actIndex}];
                trainInfo{3} = [trainInfo{3};{actTout}];
            end
            
            %% B.预测阶段，若存在激活神经元
            if word == 1 && ~isempty(FirstWordInfo{sample}) %若已经学习过，那么已经存储下来了，直接调用就行
                predictState = FirstWordInfo{sample,1};     
                predictWordList = FirstWordInfo{sample,2};
                predictInformation = FirstWordInfo{sample,3};
            else
                predictState = zeros(columnNum,neuronNum);     %每个神经元的预测状态，初始为0，无预测
                neuronHasSegmentsIndexs = find(segmentsNum>0); %获取有segments的神经元
                predictWordList = [];                          %预测的单词列表
                predictInformation = {};                       %预测的神经元以及它的输出
                hasContextNeurons = zeros(columnNum,neuronNum);%每个预测神经元是由哪些神经元导致的，初始为0，没有导致下一个的预测
                %% step1:遍历所有segment,得出预测神经元
                for num = 1:length(neuronHasSegmentsIndexs)        %遍历所有有segments的神经元
                    neuronSegIndex = neuronHasSegmentsIndexs(num); %神经元的位置
                    segments = segmentsList{neuronSegIndex};       %获取该神经元的segments
                    
                    %% step2 判断该segments和激活神经元有相交的神经元没有，小于阈值，则不用计算膜电压了
                    for segIndex = 1:segmentsNum(neuronSegIndex) %该神经元segments的个数
                        
                        segmentW = segments{segIndex};    %当前segment的连接权重信息
                        %segmentlabel = segments{2,segIndex};%当前segment的标签
                        segmentSynIdx = segmentW(:,1);      %权重连接的神经元的index
                        
                        %% 比较segment有权重的神经元和激活神经元的交集
                        diffNeurons = actIndex-segmentSynIdx';
                        [~,comIdx] = find(diffNeurons==0);
                        
                        %% 判断该segments的交集是否大于阈值，若小于阈值，则不计算膜电压
                        if (length(comIdx) >= segmentThreshold)
                            commonNeurons = segmentSynIdx(comIdx);
                            %% step3 计算segment在下个周期的膜电压，膜电压最大的时候为其输出
                            %% 3.1 取出公共神经元的延迟和权重
                            delay = segmentW(comIdx,4);              %该神经元对应的segment的延迟
                            weight = segmentW(comIdx,2);             %segment连接的权重
                            
                            %% 3.2 取出公共神经元对应的柱号，取出对应的输出
                            Tout = [];
                            [comColumns,~] = ind2sub(size(neuronConnectNum),commonNeurons);
                            comIndex = zeros(size(comColumns));
                            for comi = 1:length(comColumns)
                                comIndex(comi) = find(activecolumns == comColumns(comi));
                            end
                            Tout= currentTout(comIndex);
                            
                            %% 3.3 计算激活神经元输出currentTout的在下个周期的PSP，膜电压，输出
                            allInputs = Tout' + delay;
                            simtime = min(allInputs)-T_step:T_step:max(allInputs)+endTime; %下个周期的运行时间
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
                            
                            %% 3.4 若有输出证明segments激活了，保存相关信息
                            if (length(toutSeg) == 1)                          %若被激活
                                predictState(neuronSegIndex) = 1; %该神经元当前时刻变为预测状态
                                hasContextNeurons(commonNeurons) = 1;
                                wordName = segmentW(comIdx,3);               %该segment的标签
                                if any( wordName ~= wordName(1))
                                    wordName = mode(wordName);
                                else
                                    wordName =  wordName(1);
                                end
                                index = find(predictWordList == wordName);     %查看这个单词是否被预测过
                                if (isempty(index)) %第一次出现这个预测单词
                                    predictWordList(end+1) = wordName;        %将该单词加入列表
                                    predictInformation{length(predictWordList),1} = toutSeg; %该segment的输出保存，第1列装输出
                                    predictInformation{length(predictWordList),2} = neuronSegIndex; %第2列装对应的神经元
                                else                %非第一次出现预测的单词
                                    pridictIdx =  predictInformation{index,2};
                                    if isempty(find(pridictIdx == neuronSegIndex))
                                        preTout = predictInformation{index,1};        %找到之前存储的位置
                                        preTout(end+1,1:length(toutSeg)) = toutSeg;%输入拼接上去
                                        preTout(preTout == 0) = simtime(end);      %多出的自动补最大时间
                                        predictInformation{index,1} = preTout;        %第1列装输出
                                        predictInformation{index,2} (end+1,:)= neuronSegIndex; %第2列装对应的神经元
                                    end
                                end
                            else
                                fprintf('多个输出');
                                fprintf(fileID,'多个输出');
                            end
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
            %             if size(FirstWordInfo,1)~=1
            %                 size(FirstWordInfo,1)
            %             end
            preState(:,:,word) = predictState;
            
        else
            %% 开始预测
            fprintf('Test %d sentence: ',seq);
            fprintf(fileID,'Test %d sentence: ',seq);
            fprintf('%s ',inputLablesCell{testSeq(seq,1:word-1)});
            fprintf(fileID,'%s ',inputLablesCell{testSeq(seq,1:word-1)});
            fprintf('\n');
            fprintf(fileID,'\n');
            
            %%     判断一下是否为叶子结点
            hasContextNeurons = find(hasContextNeurons~=0);               %有预测的激活神经元
            
            if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)&&length(actIndex)~=activeNum*neuronNum%有预测的激活神经元《〈激活神经元，且训练不只一个单词（第一个单词全激活）
                predicSegLen = size(outputInfo,1);
                outputInfo{predicSegLen+1,1} = trainInfo{1};%第1列存该句子所有单词标签
                outputInfo{predicSegLen+1,2} = trainInfo{2};%第2列存该句子所有单词激活的神经元
                outputInfo{predicSegLen+1,3} = trainInfo{3};
            end
            
            %% 开始建树预测
            haveReadList = zeros(size(predictWordList));   %用于存储已经用过的单词
            while (~isempty(predictWordList))              %已经遍历完了，栈内没有任何元素了
                %% 弹出最顶上的栈元素
                listLen = find(haveReadList == 0);         %所有未用过的单词的index
                
                %% 栈顶部不存在
                if(isempty(listLen))
                    predictWordList(listLen) = [];
                    predictInformation(listLen,:) = [];
                    haveReadList(listLen) = [];
                    break;
                end
                listLen = listLen(end);                    %最上面那个
                
                currentWord = predictWordList(listLen);    %它所在的单词标签
                %                 inputLablesCell{currentWord}
                currentSegmentsCell = predictInformation(listLen,1:2);%它所在的单词的预测神经元信息
                haveReadList(listLen) = 1;                            %该单词已经用过了
                %% 赋值
                currentTout = currentSegmentsCell{1,1};    %当前的预测单词所在神经元的实际输出
                %% 没有输入预测状态变为活跃状态
                actIndex = currentSegmentsCell{1,2};       %激活神经元所在的index
                if length(actIndex) < segmentThreshold
                    %% 去掉这个单词，是噪声，会导致栈顶不存在
                    predictWordList(listLen) = [];
                    predictInformation(listLen,:) = [];
                    haveReadList(listLen) = [];
                    continue;
                end
                tempWordList = []; %% 当前单词的预测单词
                tempInformation = {}; %% 当前单词的预测的segments
                hasContextNeurons = zeros(columnNum,neuronNum);
                %% step1:遍历所有segment
                for num = 1:length(neuronHasSegmentsIndexs)
                    neuronSegIndex = neuronHasSegmentsIndexs(num);
                    segments = segmentsList{neuronSegIndex};%获取相应位置的segments
                    %% step2 判断该segments和激活神经元有相交的神经元没有，小于阈值，则不用计算膜电压了
                    for j = 1:segmentsNum(neuronSegIndex) %该神经元segments的个数，因为segments中第二行是标签，所以我们这里用的延迟
                        % 当前segment的连接权重
                        segmentW = segments{1,j};
                        %segmentlabel = segments{2,j};
                        % 权重对应的神经元
                        segmentSynIdx = segmentW(:,1);
                        %% 比较segment有权重的神经元和激活神经元的交集
                        
                        %% 矩阵形式的公用神经元
                        diffNeurons = actIndex-segmentSynIdx';
                        [~,comIdx] = find(diffNeurons==0);
                        
                        
                        %判断该segments的交集是否大于阈值，若小于阈值，则不计算膜电压
                        if (length(comIdx) >= segmentThreshold)
                            commonNeurons = segmentSynIdx(comIdx);
                            %% step3 计算segment在下个周期的膜电压，膜电压超过阈值的点火
                            hasContextNeurons(commonNeurons) = 1;
                            %% 3.1 取出公共神经元的延迟
                            delay = segmentW(comIdx,4);              %该神经元对应的segment的延迟
                            weight = segmentW(comIdx,2);             %segment连接的权重
                            
                            %% 3.2 取出公共神经元对应的输出
                            %要被删除的index
                            delColumn = setdiff(actIndex,commonNeurons);
                            %找出要删除柱的下标
                            delIndex = ismember(actIndex,delColumn);
                            %删除其他柱的输出，剩下的就是公共神经元的输出
                            Tout = currentTout;          %输出
                            Tout(delIndex==1) = [];
                            
                            %% 3.3 计算激活神经元输出currentTout的在下个周期的PSP，膜电压
                            %                             timePeriod = ceil((Tout+delay)/Tmax); %下个周期的时间
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
                            
                            if (length(toutSeg) == 1)           %若被激活
                                wordName = segmentW(comIdx,3);               %该segment的标签
                                if any( wordName ~= wordName(1))
                                    wordName = mode(wordName);
                                else
                                    wordName =  wordName(1);
                                end
                                index = find(tempWordList == wordName);
                                if (isempty(index)) %第一次出现这类输出
                                    tempWordList(end+1) = wordName;
                                    tempInformation{length(tempWordList),1} = toutSeg; %第1列装输出
                                    tempInformation{length(tempWordList),2} = neuronSegIndex; %第2列装对应的柱神经元
                                else                %非第一次出现
                                    pridictIdx =  tempInformation{index,2};
                                    if isempty(find(pridictIdx == neuronSegIndex))
                                        preTout = tempInformation{index,1};        %找到之前存储的位置
                                        preTout(end+1,1:length(toutSeg)) = toutSeg;%输入拼接上去
                                        preTout(preTout == 0) = simtime(end);      %多出的自动补最大时间
                                        tempInformation{index,1} = preTout;        %第1列装输出
                                        tempInformation{index,2} (end+1,:)= neuronSegIndex; %第2列装对应的神经元
                                    end
                                end
                            else
                                fprintf('膜电压不够没有输出\n');
                                fprintf(fileID,'膜电压不够没有输出\n');
                            end
                        end
                    end
                end
                %% step3 判断是否为叶子结点叶子结点1：判断当前是否有激活的神经元没有预测的，表明句子结束，加入列表
                leaf = 0;
                hasContextNeurons = find(hasContextNeurons~=0);                %有预测的激活神经元
                %预测的激活神经元<当前的激活神经元，且不是全激活的单词导致的
                if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)
                    leaf = 1;
                end
                %% 判断当前有没有新的预测，若有，那么加入栈，若没有，那么是叶子结点
                nextLen = length(tempWordList);
                if nextLen > 0 %%非叶子结点
                    predictWordList(end+1:end+nextLen) = tempWordList;
                    predictInformation(end+1:end+nextLen,:) = tempInformation;
                    haveReadList(end+1:end+nextLen) = 0;
                else
                    leaf = 1; %% 无预测,叶子结点2，整体都没有输出
                end
                if leaf == 1
                    %% 结束 保存所有词的标签
                    wordList = find(haveReadList == 1);
                    
                    predicSegLen = size(outputInfo,1);
                    outputInfo{predicSegLen+1,1} = [trainInfo{1},predictWordList(wordList)];%第1列存该句子所有单词标签
                    outputInfo{predicSegLen+1,2} = [trainInfo{2};predictInformation(wordList,2)];%第2列存该句子所有单词激活的神经元
                    outputInfo{predicSegLen+1,3} = [trainInfo{3};predictInformation(wordList,1)];%第3列激活的神经元的输出
                    %% 删除该句子相关信息
                    start = find(haveReadList == 0); %%从还没开始遍历的结点
                    
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
                minSimility = length(unique(predictSentence))/maxSenLen;
                for goalNeuron = 1:seqenceNum                     %遍历所有目标神经元
                    targetScentence = targetsCell{goalNeuron,3};  %目标神经元的单词序列
                    simility = length(intersect(predictSentence,targetScentence))/length(targetScentence);%比较预测的和目标句子相似度
                    
                    if simility >=minSimility
                        %存在相似度，就去判断一下是否激活
                        
                        targetWeights = targetsCell{goalNeuron,1};%目标神经元和激活神经元相连的权重
                        targetDelays = targetsCell{goalNeuron,2}; %目标神经元和激活神经元相连的权重
                        
                        currentActiveNeurons = outputInfo{i,2};%激活神经元
                        currentTout = outputInfo{i,3};         %激活神经元的输出
                        
                        weights = [];  %存放激活神经元到输出的权重
                        delays = [];   %存放激活神经元到输出的延迟
                        Tout = [];     %存放激活神经元到输出的输出
                        
                        %% 目标神经元和激活神经元相连的连接，延迟，以及输出，预测的和输入的同样处理
                        for k = 1:length(currentActiveNeurons)
                            idx = currentActiveNeurons{k};
                            weights = [weights;targetWeights(idx)];
                            delays  = [delays;targetDelays(idx)];
                            Tout = [Tout;currentTout{k}];
                        end
                        
                        AllV = [];
                        %% 选择模拟时间
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
                    maxV = maxV - diffThresholdMany; %多个输出
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
            
            %% 和真实的相比的正确率
            predicStnIdx = unique(sort(predicStnIdx)); %预测的句子
            actualStnIdx = [];
            subsequence = testSeq(seq,:);       %当前子序列
            subsequence(subsequence == -1) = []; %去掉-1的那部分
            subLen = length(subsequence);       %表示子序列的长度
            for goalNeuron = 1:seqenceNum                     %遍历所有目标神经元
                targetScentence = targetsCell{goalNeuron,3};  %目标神经元的单词序列
                targetLen = length(targetScentence);          %表示目标序列的长度
                firstIdx = find(targetScentence == subsequence(1));%第一个单词在目标序列中的位置
                
                if (subLen>targetLen || isempty(firstIdx))    %不是目标的子序列
                    continue;
                end
                
                for i = 1:length(firstIdx) %遍历第一个单词在目标序列中的位置
                    targetIdx = firstIdx(i)+1; %获取第2个单词的位置
                    rightFlag = 1; %假设该句子是
                    for subIdx = 2:subLen  %从第二个单词开始看是否匹配子序列
                        if targetIdx<=targetLen && subsequence(subIdx) == targetScentence(targetIdx)  %若目标单词和子序列一致
                            targetIdx = targetIdx+1;
                        else %存在一个不一致，那么就不是
                            rightFlag = 0;
                            break;%不是子序列
                        end
                    end
                    if rightFlag == 1
                        actualStnIdx(end+1) = goalNeuron;
                        break;
                    end
                end
            end
            %% 计算正确率
            
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

