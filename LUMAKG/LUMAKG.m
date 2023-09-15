addpath(genpath(pwd))
clear
%%
% column为结构体，包含域：
%   word -- 该柱代表的词
%   neuron -- 结构体，包含子域：
%       flag -- 是否可被搜索，=1可被搜索，=0已被搜索过
%       path -- 后续路径
%       state -- 点火状态，0不激活，1为UA，2为PA
%       eta -- 点火次数
%       time -- 点火时间
%       precessor -- 突触前神经元，4列.第1、2列为突触前神经元的索引，第3、4列为对应的delta和weight
%       successor -- 突触后神经元，4列.第1、2列为突触后神经元的索引，第3、4列为对应的delta和weight
trainText;
tau_m = 50;
tau_s = tau_m/4;
beta = tau_m/tau_s;
Vnorm = 2.1166;         % beta^( beta /(beta-1) ) / (beta-1);
t_max = tau_m * tau_s * log(tau_s/tau_m) / (tau_s - tau_m);   % t_max大于dT

threshold = 1;
M = 32;          % 每个柱包含的神经元个数
dT = [20 30 45 62 80:17:320]';      % 相差时间间隔
%dT = [20 40 60 80 100, 120:200]';
nc = 0;         % num of column
column = [];
for s = 1: length(text)
    sentence = text{s};
    establishWMS;       % Establishing working minicolumn set;
    findingPGMs;        % Finding PGMs
    removeOverlap;      % Removing overlapping columns
    linkPA;             % Establishing a Sequence of Linked PA Neurons
    updateWeight;       % Updating weights;
    resetWM;            % Reset all neurons to inactive state;
end
strLen = length(column);
strCell={};
for tempi = 1:strLen
    strCell{tempi,1}=column(tempi).word;
end
sequenceTrain = zeros(1000,10);
for tempi = 1:1000
    sentence = text{tempi};
    for tempj = 1:10
        str = sentence{tempj};
        for tempk = 1:strLen
            if strcmp(strCell{tempk},str)
                break;
            end
        end
        sequenceTrain(tempi,tempj)=tempk;
    end
    
end
for repnum = 0:9
rightCount = zeros(1000,5);
for trail = 1:5
testText
% dT = [20 30 45 62 80:17:320]';
filestr = 'LUMAKG_rep_';
filestr = [filestr,num2str(repnum)];
filestr = [filestr,num2str(trail)];
filestr = [filestr,'.txt'];
frepId=fopen(filestr,'w');  %打开文本文件
for seqi = 1: length(testEX)
    fprintf('sentence %d:', seqi);
    fprintf(frepId,'sentence %d:', seqi);
    input = testEX{seqi};
    testsequence = [];
    testingLUMAKG2;
    %% 比较相似度
    C = zeros(1000,1);
    for tempk = 1:1000
        diff = sequenceTrain(tempk,:) - testsequence';
        C(tempk) = length(find(diff==0));
    end
    predictsen = find(C==max(C));
    if predictsen == seqi
        rightCount(seqi,trail) = 1;
    end
end
fprintf("test accuracy is %f\n",sum(rightCount(:,trail))/1000);
fprintf(frepId,"test accuracy is %f\n",sum(rightCount(:,trail))/1000);
end
allAcc = mean(sum(rightCount)/1000)
fprintf("mean test accuracy is %f",allAcc);
fprintf(frepId," mean test accuracy is %f",allAcc);
matstr = 'rep-';
matstr = [matstr,num2str(repnum)];
matstr = [matstr,'.mat'];
save(str,"allAcc","rightCount");
end

