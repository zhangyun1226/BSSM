clear;
close all;
%% ʱ��ش���
initParameter;
load data1000.mat;
load spation1000.mat
neuronNum = 32;
sysinput = 100;
seqenceLen = size(sequence,1);                %ѵ��������
wordLen = size(sequence,2);                   %ÿ�����е���󳤶�
seqenceLen = 1000;                                       %ѵ���ľ�������
%% �Ƚ�ͻ��ǰ�ĵ�ѹ������������
inputVCell = {};
for wordNum = 1: max(sequence(:))
    tempPSP = squeeze(allPSPTrain(wordNum,:,:));
    inputVCell{wordNum} = inputWeight'*tempPSP;
end
%% segment ��ز���
segmentConnect = 7;                       %ÿ��segment�ͼ�����Ԫ�����ĸ���
maxSegmentConnect = 4*segmentConnect;     %ÿ��segments������4�����ģ��Ҳ�Ҫ����
targetWordNum = 4;                        %Ŀ�굥��4������ʹĿ����ԪĤ��ѹ����1
wi = 1/(activeNum*targetWordNum);         %ÿ��������Ԫ��Ŀ����Ԫ��Ȩ��
firstWi = 1/(activeNum*neuronNum*targetWordNum);
segmentThreshold = segmentConnect;
disWeight = 1/(segmentThreshold);
columnInhW = 0.25;

for num = 1:columnNum*neuronNum
    segmentsList{num} = {};%�洢Ȩ��
end
for num = 1:columnNum*neuronNum
    connectedSegmentList{num} = {};%�洢Ȩ��
end
segmentsNum = zeros(columnNum,neuronNum);              %ÿ����Ԫ�м���segment
neuronConnectNum = zeros(columnNum,neuronNum);         %ÿ����Ԫ�������ӵĴ���
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%�ӳٵ������PSP��ʱ��
%% ѵ��
targetWeights = zeros(columnNum,neuronNum,seqenceLen);  %��Ŀ����Ԫ��Ȩ��
targetDelays = Inf*ones(columnNum,neuronNum,seqenceLen);%��Ŀ����Ԫ���ӳ�
targetNeurons =  zeros(columnNum,neuronNum,seqenceLen);
% secondShare = [6,8,11,12,17,18,20,21];
for seq = 1:1:seqenceLen                               %����ѵ������
    tempTargetWeights = zeros(columnNum,neuronNum);  %��Ŀ����Ԫ��Ȩ��
    tempTargetDelays = Inf*ones(columnNum,neuronNum);%��Ŀ����Ԫ���ӳ�
    tempTargetNeurons = zeros(columnNum,neuronNum);
    sentenceLen = length(sequence(seq,:)~=0);
    for wordNum = 1:wordLen                            %����ÿ������
        %% ��¼��һ�����ʵ�λ��
        if wordNum~=1
            lastWord = word;
            allInputVlast = inputVCell{lastWord};
        else
            lastWord = 0;
            allInputVlast = zeros(columnNum,nt);
        end
        %% ��ǰ���ʵĲ���
        word = sequence(seq,wordNum);               %��ǰ����
        %% ��ʼ��Ĥ��ѹ����Ԫ�����
        toutCA3Neuron = -Inf*ones(columnNum,neuronNum);
        fireColumns = zeros(columnNum,1);
        fireNeurons = zeros(columnNum,neuronNum);
        intColumnTime = Inf*ones(columnNum,neuronNum);
        zerosV = zeros(columnNum,neuronNum);
        
        if word == 0 %%���н���
            targetWeights(:,:,seq) = tempTargetWeights;
            targetDelays(:,:,seq) = tempTargetDelays;
            targetNeurons(:,:,seq) = tempTargetNeurons;
            break;
        else
            fprintf("seq %d word %d:%s\n",seq,wordNum,inputLablesCell{word});
            %% ���㵱ǰ��ÿ����Ԫ�����,ֻ���������һ�ε����
            %% �����Ϊǰ������ �������� ������ �������� ��Ӧ��
            startTime = (wordNum-1)*nt;      % ��ʼʱ��
            winnerSegIdx = zerosV;           % ���ڴ洢�ĸ���ͻ����Ԫ���
            winnerSegV = zerosV;             % winner��ͻ��Ĥ��ѹ
            allInputV = inputVCell{word};    % ��ǰ���ʵ�ǰ����ѹ
            allV = [];                       % �洢Ĥ��ѹ��������
            %% ȷ����ЩԶ����ͻ����Զ�˼���
            disSegNeu = [];                  % ������Щ��Ԫ��segment��
            disSegNum = [];                  % ��Ԫ�ĵڼ���segments�����ϸ����ڴ��ݹ�������Ϣ
            if wordNum == 2
                %% ��һ�����ʼ������ϵ�������Ԫ��column,neuron]
                tempNeu = [];
                for tempi = 1:length(targetColumnsLast)
                    tempNeu = [tempNeu,targetColumnsLast(tempi)+(0:neuronNum-1)*columnNum];
                end
                %% ����Щ��Ԫ���ӵ�����segments��������б�
                connectedNeurons = [];           %��Щ��Ԫ���ϸ����ڵ���Ԫ������
                for tempi = 1:length(tempNeu)
                    tempNeuron = tempNeu(tempi); %��ȡ�ϸ����ڵ���Ԫ
                    tempSegments = connectedSegmentList{tempNeuron};%�������ӵ�segments����connectedNeurons
                    if ~isempty(tempSegments)
                        connectedNeurons = [connectedNeurons;tempSegments];
                    end
                end
                %% ȥ�أ�ĳ��segment������Ԫ���������Ի���ֶ�Σ�
                if ~isempty(connectedNeurons)
                    disSegNeu = connectedNeurons(:,1); %��Զ����ͻ����Ԫ
                    disSegNum = connectedNeurons(:,2); %Զ����ͻ�ı��
                    [~,idx]= unique(disSegNeu*1000+disSegNum); %�����ȥ��
                    disSegNeu = disSegNeu(idx);
                    disSegNum = disSegNum(idx);
                end
            elseif wordNum>2
                connectedNeurons = [];
                for tempi = 1:length(lastLearnNeurons)
                    tempNeuron = lastLearnNeurons(tempi);
                    tempSegments = connectedSegmentList{tempNeuron};
                    if ~isempty(tempSegments)
                        connectedNeurons = [connectedNeurons;tempSegments];
                    end
                end
                %% ��֮ǰ��ѧϰ��Ԫ�����ӵ�����
                if ~isempty(connectedNeurons)
                    disSegNeu = connectedNeurons(:,1); %��Զ����ͻ����Ԫ
                    disSegNum = connectedNeurons(:,2);
                    [~,idx]= unique(disSegNeu*1000+disSegNum);
                    disSegNeu = disSegNeu(idx);
                    disSegNum = disSegNum(idx);
                end
            end
            %% ��ʼ����Ĥ��ѹ
            intColumnTime = Inf*ones(columnNum,1);%������Ԫ�����
            toutCA3Neuron = -Inf*ones(columnNum,neuronNum);%��Ԫ�ڸ����������һ�ε��
            fireColumns = zeros(columnNum,1);              %��Щ�����
            fireNeurons = zeros(columnNum,neuronNum);      %��Щ��Ԫ���
            for t = startTime+(through:nt)*dt  %�����Ԫ�ĵ��ʱ�䣬����through��Tmax
                tIdx = t/dt;                   %��ǰʱ��
                %% step 1 ǰ������Ĳ�����ǰ����ѹ
                currentT = tIdx-startTime;
                inputV = allInputV(:,currentT);
                %% step 2 ��Ԫ������ = ��������
                tempInh = t - intColumnTime;
                tempInh(tempInh<=0) = inf;
                CA3I = ca3Ivalue*sum(exp(-tempInh/Tau_s),2);
                %% step 3 ��Ԫ���� = �����񵴣�1+2+3Ϊ���ϵ���������
                columnV = inputV + CA3I + allOsc(:,currentT);
                shareV = repmat(columnV,1,neuronNum);%����ά�ȣ�����ÿ��������������Ԫ��Ĥ��ѹ
                %% step 4 ��Ԫ�Լ����ص����ʣ�Զ����ͻ+��Ӧ��+��������
                CA3NeuV = zerosV;       %��ʼ��ÿ����Ԫ�Ķ��ص�ѹ
                tempSegmIdx = zerosV; %ÿ����Ԫ��ʤsegment��λ��
                %4.1 ��Ԫ�Ĳ�Ӧ��
                ref = -exp((toutCA3Neuron-t)/Tau_a);
                %4.2 ��Ԫ��Զ����ͻ�����ĵ�ѹ,���Զ����ͻֻ�е�ѹ���Ļ�ʤ
                for i = 1:length(disSegNeu)
                    neu = disSegNeu(i);               %��i����Ԫneu
                    segi = disSegNum(i);              %neu��Ԫ��segmentλ��
                    segment = segmentsList{neu}{segi};%��Ӧ��segment
                    %% ���㵱ǰsegment�ĵ�ѹ
                    prei = segment(:,1);              %ͻ��ǰ��Ԫ
                    delayi = segment(:,4);            %�ӳ�
                    weighti = segment(:,2);           %Ȩ��
                    pretout = toutCA3NeuronLast(prei);%���
                    %��ͻ��ǰ��Ԫ�ĵ�ѹ
                    temp = t - pretout - delayi;
                    temp(temp<=0) = Inf;
                    tempPSP = exp(-temp/Tau_s);
                    segV = weighti'*tempPSP;
                    %������Ĥ��ѹ����֮ǰ����ͻ�����������滻
                    if segV > CA3NeuV(neu)
                        CA3NeuV(neu) = segV;             %����Ԫ��Զ��Ĥ��ѹ
                        tempSegmIdx(neu) = segi;         %��Ӧ��segment
                    end
                end
                tempSegmementV = CA3NeuV;
                % 4.3 ������Ԫ������Ƽ���CA3NeuV
                allFireColumns = find(fireColumns >0);   %������
                fileColumnsLen = length(allFireColumns); %���������Ŀ
                if fileColumnsLen>0
                    if fileColumnsLen*neuronNum ~= length(find(fireNeurons)>0)%������ȫ���
                        innerInhV = -columnInhW*ones(fileColumnsLen,neuronNum);%�������������Ԫ��������
                        CA3NeuV(allFireColumns,:) = CA3NeuV(allFireColumns,:) + innerInhV;
                        CA3NeuV(fireNeurons~=0) = CA3NeuV(fireNeurons~=0)+columnInhW;%������Ԫ��������
                    end
                end
                %% ������Ԫ��Ĥ��ѹ
                CA3EV = shareV + CA3NeuV + ref;
                allV(end+1,:) = CA3EV(:);
                %                 AllColumnV(tIdx,:) = columnV(:);
                %                 CA3IV(end+1,:) = CA3I(:);
                %                 refV(end+1,:) = ref(:);
                %% ���ڳ�����ֵ����Ԫ
                neuron = find(CA3EV >=Theta);      %Ĥ��ѹ�Ƿ������ֵ���
                if(~isempty(neuron))               %��������Ԫ���
                    %Ѱ����Ԫ��Ӧ����
                    [column,~] = ind2sub(size(CA3EV),neuron);
                    %1.��������ʱ������һά
                    currentOut = t*ones(columnNum,1);
                    currentOut(column) = Inf;
                    if all(intColumnTime == Inf)
                        intColumnTime = currentOut;
                    else
                        intColumnTime(:,end+1) = currentOut;
                    end
                    %2.��Ԫ������仯��ֻ�������һ�ε�
                    toutCA3Neuron(neuron) = t;
                    %3.����������Ԫ�����仯������ס���ǵĻ�ʤsegment
                    fireColumns(column) = 1; %�����ı�־
                    fireNeurons(neuron) = 1; %��Ԫ���ı�־
                    winnerSegIdx(neuron) = tempSegmIdx(neuron); %��ʤsegment��λ��
                    winnerSegV(neuron) = tempSegmementV(neuron);%ÿ��segment�����Զ�˵�ѹ
                end
            end
            %% ʵ�ʵ�������Ŀ��������Ŀ��������Ԫ��������
            targetColumn = outputColumn(word,:);   %Ŀ����
            %% �ڲ��彨��Զ����ͻ-(��һ�����ʳ���)
            if wordNum ==1
                lastLearnNeurons = [];             %��һ�����ʣ�û��ѧϰ��Ԫ�����ý�����ͻ
                targetColumnsLast = targetColumn;    %������������������
                
                lastFireNeurons = [];              %����ÿ��Ŀ����Ԫ�����
                for tempi = 1:length(targetColumnsLast)
                    lastFireNeurons = [lastFireNeurons,targetColumnsLast(tempi)+(0:neuronNum-1)*columnNum];
                end
                
                toutCA3NeuronLast = toutCA3Neuron; %�������������Ԫ�����
                continue;
            else %��������
                %% ��ʼ��ѧϰ��Ԫ
                learnNeurons = zeros(1,activeNum);
                %% ��ȡÿ�����ĵ����Ԫ�������������������������
                winnerNum = winnerSegIdx(targetColumn,:); %Ŀ�����ϵĻ�ʤ��Ԫ��Ӧ��segment
                winnerNum(winnerNum>1)=1;                 %ֻҪ����0���Ǵ��ڻ�ʤsegment
                winnerNum = sum(winnerNum,2);             %ÿ�����ϵ��л�ʤsegment������
                winnerNum(winnerNum == 0) = neuronNum;    %һ����û�У�����ȫ����
                [winnerNum,winIdx] = sort(winnerNum,'descend');
                targetColumn = targetColumn(winIdx);
                %% ��������ȫ������ϸ����ڵ���û��ѡ��ѧϰ��Ԫ
                if isempty(lastLearnNeurons) && winnerNum(1) == neuronNum
                    %% ѡ���������������ٵ���Ϊѧϰ��Ԫ
                    preNeurons = zeros(1,activeNum);                        %Ϊ��һ������ѡ����Ԫ
                    for num = 1:activeNum
                        tempColumn = targetColumnsLast(num);                %�����ϸ����ڵ�Ŀ����
                        neronsCon = neuronConnectNum(tempColumn,:);         %������Ԫ�����������Ŀ
                        minConNeuron = find(neronsCon == min(neronsCon));   %ѡ���������ӵ�������Ԫ
                        currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %�����ж�������ѡ��һ����Ԫ
                        preNeurons(num) = currentNeuron;
                    end
                    %% ����Ԫת��Ϊ��Ӧ��index
                    lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                end
                
                %% �ӵ������������ʼ����������ͻ
                noiseFlag = 0;
                for i = 1:activeNum
                    %% step 1 ѡ��ѧϰ��Ԫ���жϵ�ǰ�������������л�ʤ���Ͳ��ֵ��Ǿ�֪ʶ
                    currentColumn = targetColumn(i);   %��ȡ��ǰ������
                    chooseFlag = 0;
                    %% û�л�ʤ����Ԫ���޾�֪ʶ��ѡ����ͻ���ٵ���ԪΪѧϰ��Ԫ
                    if winnerNum(i)==neuronNum || noiseFlag == 1
                        neurons = find(segmentsNum(currentColumn,:) == min(segmentsNum(currentColumn,:)));
                        neuronsIdx = find(toutCA3Neuron(neurons) ==min(toutCA3Neuron(neurons)));
                        chooseNeuron2 = neurons(neuronsIdx(unidrnd(length(neuronsIdx))));%�ж�������ѡ��һ��
                        chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                        chooseFlag = 1;
                    else
                        %% ���ڻ�ʤ��ͻ����Ԫ���о�֪ʶ��������
                        chooseNeuron2 = find(winnerSegIdx(currentColumn,:)>0); %�ҵ���Щ��Ԫ�л�ʤ��ͻ
                        if length(chooseNeuron2) > 1 %�������ѡ����ͻ��ѹ�����Ǹ�Ϊѧϰ��Ԫ
                            chooseNeuronV = winnerSegV(currentColumn,chooseNeuron2);
                            [~,neuronIdx] = max(chooseNeuronV);
                            chooseNeuron2 = chooseNeuron2(neuronIdx);
                        end
                        chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                        winnerIdx = winnerSegIdx(chooseNeuron); %ѧϰ��Ԫ�Ļ�ʤ��ͻλ��
                        winnerSegment = segmentsList{chooseNeuron}{winnerIdx};%ѧϰ��Ԫ�Ļ�ʤ��ͻλ��
                        chooseFlag = 2;
                    end
                     %% step 2 Ϊѧϰ��Ԫ������ͻ/����
                        ...1.�ж�winner segments�Ϻ�ѧϰ��Ԫ�����ƶȣ��ڴ˻����Ͻ�������
                        ...2.������winner-segment���½�һ����ͻ����Ԫ��������
                    if chooseFlag == 2
                        prei = winnerSegment(:,1);
                        if isempty(lastLearnNeurons) %����ϸ������ǵ�һ�����ʣ����������ڵ���������������
                            [connectColumns,~] = ind2sub(size(neuronConnectNum),prei);%ͻ��ǰ��Ԫ������
                            diffColumns = connectColumns - targetColumnsLast;
                            [~,comColumnIdx] = find(diffColumns==0);%�ȽϺ��ϸ����ڵ�����������
                            comColumn = unique(targetColumnsLast(comColumnIdx));
                            %% �������µĵ��
                            if length(comColumn)<segmentConnect %������С����ֵ���������µĵ����Ҫ����ѡ���ϸ����ڵ�ѧϰ��Ԫ
                                %% ����ѡ���ϸ����ڵ�ѧϰ��Ԫ
                                preNeurons = zeros(1,activeNum);
                                for num = 1:activeNum                                            %����������
                                    currentColumn = targetColumnsLast(num);                         %������i
                                    neronsCon = neuronConnectNum(currentColumn,:);               %��i����Ԫ�����������Ŀ
                                    minConNeuron = find(neronsCon == min(neronsCon));            %ѡ���������ӵ�������Ԫ
                                    currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %�����ж�������ѡ��һ����Ԫ
                                    preNeurons(num) = currentNeuron;
                                end
                                %% ����Ԫת��Ϊ��Ӧ��index
                                lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                                %% ѡ��������Ԫ�½�����
                                chooseFlag = 1;
                                noiseFlag = 1;
                                neurons = find(segmentsNum(currentColumn,:) == min(segmentsNum(currentColumn,:)));
                                neuronsIdx = find(toutCA3Neuron(neurons) ==min(toutCA3Neuron(neurons)));
                                chooseNeuron2 = neurons(neuronsIdx(unidrnd(length(neuronsIdx))));%�ж�������ѡ��һ��
                                chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                                %                                 secondActiveNum(seq) = 32;
                            else
                                %% ���������ڵ�����ֵ�������ĵ��µĵ����ͻ����ı䡫
                                learnNeurons(i) = chooseNeuron; %����i����ѧϰ��Ԫ��ֵ
                                continue;  %% �ڶ���������ȫһ��
                            end
                        else
                            %% �Ƚ�ͻ��ǰ��Ԫ��ѧϰ��Ԫ�����ƶ�
                            diffNeurons = prei - lastLearnNeurons;
                            [~,comIdx] = find(diffNeurons==0);
                            % 3.�����������ڵ�����ֵ������Ҫ����ͻ
                            if length(comIdx)>=segmentConnect
                                learnNeurons(i) = chooseNeuron; %����i����ѧϰ��Ԫ��ֵ
                                continue; %% ��2+n��������ȫһ��
                            elseif length(comIdx)<4&&length(comIdx)>0 %��������С��һ�룬����7-x�����ӣ�
                                segmentLen = size(winnerSegment,1);       %��ǰsegment��������Ԫ����
                                addLen = segmentConnect - length(comIdx); %��Ҫ�����ӵ���ͻ��Ŀ
                                if addLen+segmentLen > maxSegmentConnect  %��������ͻ�󳬹�����ͻ������
                                    chooseFlag = 1; %�½���ͻ
                                else %��������ͻ��û������ͻ�����������ڸ���ͻ���޸�
                                    %% ��֪���õ���ͻ����comIdx��Ҫ����Ӷ���
                                    preNeuronsIdx = setdiff(1:activeNum, comIdx); %δ�����ӵ�������ѧϰ��Ԫ
                                    preNeuronsIdx = preNeuronsIdx(randperm(length(preNeuronsIdx),addLen)); %����Щ��Ԫ�����ѡ��addLen��
                                    preNeurons = lastLearnNeurons(preNeuronsIdx);
                                    
                                    postTout = toutCA3Neuron(chooseNeuron);    %��ǰӵ����ͻ����Ԫ���
                                    preTouts = toutCA3NeuronLast(preNeurons);  %ͻ��ǰ��Ԫ�����
                                    delay = postTout - (preTouts + DelayTime); %�����ӳ�
                                    %% �������Ϣ������ͻ��
                                    winnerSegment( segmentLen+1: segmentLen+addLen,1) = preNeurons;%2.3.1segments�����ĵ���Ԫ��index
                                    winnerSegment( segmentLen+1: segmentLen+addLen,2) = disWeight;  %2.3.2��segments�����ĵ���Ԫ��Ȩ��
                                    winnerSegment( segmentLen+1: segmentLen+addLen,3) = word;     %2.3.3��ǰ���ʵ���Ϣ
                                    winnerSegment( segmentLen+1: segmentLen+addLen,4) = delay;
                                    %% ���޸ĵ�segment����segment�б�
                                    segmentsList{chooseNeuron}{winnerIdx} = winnerSegment;          %������ͻ�б������Ϣ
                                    neuronConnectNum(preNeurons) = neuronConnectNum(preNeurons) +1; %ͻ��ǰ������Ԫ������������1
                                    %% ���¼�ͻ��ǰ��Ԫ�ĺ���segment����
                                    for tempi = 1:addLen
                                        temp = connectedSegmentList{preNeurons(tempi)};
                                        if isempty(temp)
                                            temp = [chooseNeuron,winnerIdx];
                                        else
                                            temp(end+1,:) = [chooseNeuron,winnerIdx];
                                        end
                                        connectedSegmentList{preNeurons(tempi)} = temp;
                                    end
                                end
                            else %���������ڵ���4�����������µĵ���½���ͻ
                                chooseFlag = 1; %�½���ͻ
                            end
                        end
                    end
                    
                    if chooseFlag == 1 %�½���ͻ
                        %% ��Ȼ�����ģ��������Ĵ�����Ҫ���������ĵ�ѧϰ��Ԫ
                        if isempty(lastLearnNeurons) %���ϸ����ڵ���Ԫ�ǵ�һ�����ڣ���Ҫѡ���ϸ����ڵ�ѧϰ��Ԫ
                            preNeurons = zeros(1,activeNum);
                            for num = 1:activeNum                                            %����������
                                currentColumn = targetColumnsLast(num);                         %������i
                                neronsCon = neuronConnectNum(currentColumn,:);               %��i����Ԫ�����������Ŀ
                                minConNeuron = find(neronsCon == min(neronsCon));            %ѡ���������ӵ�������Ԫ
                                currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %�����ж�������ѡ��һ����Ԫ
                                preNeurons(num) = currentNeuron;
                            end
                            %% ��ѧϰ��Ԫת��Ϊ��Ӧ��index
                            lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                        end
                        % 1. ��ʼ����ͻ������
                        segment = zeros(segmentConnect,4);
                        % 2. ��һ�����ڵ�10��ѧϰ��Ԫ�����ѡ��70%��Ϊ���ӵ���Ԫ
                        preNeurons = lastLearnNeurons(randperm(activeNum,segmentConnect));
                        % 3. ����Ϣ�����½���ͻ
                        segment(:,1) =  preNeurons;              %����ͻ��ǰ��Ԫ
                        preTouts = toutCA3NeuronLast(preNeurons);%ͻ��ǰ��Ԫ�����
                        postTout = toutCA3Neuron(chooseNeuron);  %��ǰ��ͻ��Ԫ�����
                        if postTout == -Inf %����ǰ��Ԫû�����
                            postTout = startTime+through+find(allV(:,chooseNeuron)==max(allV(:,chooseNeuron)));
                        end
                        delay = postTout - (preTouts + DelayTime); %�����ӳ�
                        segment(:,4) = delay;                      %����ͻ��ǰ��Ԫ�ӳ�
                        segment(:,2) = disWeight;
                        segment(:,3) = word;
                        %% ���½���segment����segment�б�
                        num = segmentsNum(chooseNeuron);            %��ǰѧϰ��Ԫӵ�е�segments������
                        currentSegCell = segmentsList{chooseNeuron};%��ǰѧϰ��Ԫӵ�е�segments
                        currentSegCell{num+1} = segment;            %����segmentд��
                        segmentsList{chooseNeuron} = currentSegCell;%������Ϣ
                        neuronConnectNum(preNeurons) = neuronConnectNum(preNeurons) +1; %ͻ��ǰ������Ԫ������������1
                        segmentsNum(chooseNeuron) = num+1;                %����Ԫ��segment����1
                        %% ͻ��ǰ��ԪҲ��
                        for tempi = 1:segmentConnect
                            temp = connectedSegmentList{preNeurons(tempi)};
                            if isempty(temp)
                                temp = [chooseNeuron,num+1];
                            else
                                temp(end+1,:) = [chooseNeuron,num+1];
                            end
                            connectedSegmentList{preNeurons(tempi)} = temp;
                        end
                    end
                    %% ��ͻ�����޸����
                    learnNeurons(i) = chooseNeuron; %����i����ѧϰ��Ԫ��ֵ
                end
                %% ÿ�����ϵ���Ԫ����ͻ�����޸����
                
            end
            
            %% ����������ڵ������Ϣ
            lastLearnNeurons = learnNeurons;
            targetColumnsLast = targetColumn;
            %������ֻ����ѧϰ��Ԫ�����
            toutCA3NeuronLast = -Inf*ones(columnNum,neuronNum);
            learnNeuronsOut = toutCA3Neuron(learnNeurons);
            if find(learnNeuronsOut==-Inf)
                tempIdx = find(learnNeuronsOut==-Inf);
                tempNeurons = learnNeurons(tempIdx);
                [~,maxT] = max(allV(:,tempNeurons));
                learnNeuronsOut(tempIdx) = maxT+1+(startTime+through);
            end
            toutCA3NeuronLast(learnNeurons) = learnNeuronsOut;
            lastFireNeurons = find(fireNeurons == 1);
            tempTargetWeights(learnNeurons) = wi;            %��i�����ʵ�ѧϰ��Ԫ��Ŀ���Ȩ��
            tempTargetDelays(learnNeurons) = (sentenceLen+1 - wordNum)*T;%��i�����ʵļ���Ԫ��Ŀ����ӳ٣�nT,(n-1)T...T
            tempTargetNeurons(learnNeurons) = 1;
        end
        
    end
end
tempralStr = ['temporalout',num2str(neuronNum)];
tempralStr = [tempralStr,'.mat'];
save(tempralStr);
testReplace;