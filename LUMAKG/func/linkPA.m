%% Establishing a Sequence of Linked PA Neurons
% PGMs -- 每一个PGMs元胞（已连接好的一条连续路径）看作一个节点
% WMS
% WM
L = length(PGMs);
if ~isempty(PGMs)
    FPA = PGMs{1}(1, 1);    % FPA 起始柱的序号
    nn = PGMs{1}(1, 2);
    final_path = PGMs{1};
else
    FPA = length(WM);
    nn = findingMNOC(WM(FPA), 1:length(WM(FPA).neuron));
    final_path = [FPA nn];
end
%% 往前找
j = FPA;
for i = FPA-1: -1: 1
    [exist, locs] = isexistPA(WM(i));
    if exist    % 如果有PA，选PA，否则选MNOC
        min_n = findingMNOC(WM(i), locs);
    elseif ~isempty(WM(j).neuron(nn).precessor)
        temp_n = find(WM(j).neuron(nn).precessor(:, 1) == WMS(i));
        if ~isempty(temp_n)    % 已有连接
            temp_n = WM(j).neuron(nn).precessor(temp_n, 2);
            min_n = findingMNOC(WM(i), temp_n);
        else     % 不存在连接
            min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
        end
    else      % 不存在连接
        min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
    end
    final_path = [i min_n; final_path];
    j = i;
    nn = min_n;
end
%% 往后找
j = final_path(end, 1);
nn = final_path(end, 2);
i = j + 1;
k = 2;
while i <= length(WM)
    if k <= L && ismember(i, PGMs{k}(:, 1))
        final_path = [final_path; PGMs{k}];
        k = k + 1;
        j = final_path(end,1);
        nn = final_path(end,2);
        i = j + 1;
    else
        [exist, locs] = isexistPA(WM(i));
        if exist    % 如果有PA，选PA，否则选MNOC
            min_n = findingMNOC(WM(i), locs);
        elseif ~isempty(WM(j).neuron(nn).successor)
            temp_n = find(WM(j).neuron(nn).successor(:, 1) == WMS(i));
            if ~isempty(temp_n)
                temp_n = WM(j).neuron(nn).successor(temp_n, 2);
                min_n = findingMNOC(WM(i), temp_n);
            else 
                min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
            end
        else
            min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
        end
        final_path = [final_path; i min_n];
        j = i;
        nn = min_n;
        i = j + 1;
    end
end
final_path(:, 1) = WMS;
%%  找出柱minicolumn中temp_n神经元中外链最少的神经元index
function min_n = findingMNOC(minicolumn, temp_n)
min_n = temp_n(1);
min_value = size(minicolumn.neuron(temp_n(1)).successor, 1);
for i = 2: length(temp_n)
    if isempty(minicolumn.neuron(temp_n(i)).successor) 
        min_n = temp_n(i);
        break;
    elseif size(minicolumn.neuron(temp_n(i)).successor, 1) < min_value
        min_n = temp_n(i);
        min_value = size(minicolumn.neuron(temp_n(i)).successor, 1);
    end
end
end
%% 判断柱中是否包含PA
function [exist, locs] = isexistPA(minicolumn)
exist = 0;
locs = [];
for m = 1: length(minicolumn.neuron)
    if minicolumn.neuron(m).state == 2
        exist = 1;
        locs = [locs m];
    end
end
end