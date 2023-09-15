%% Establishing a Sequence of Linked PA Neurons
% PGMs -- ÿһ��PGMsԪ���������Ӻõ�һ������·��������һ���ڵ�
% WMS
% WM
L = length(PGMs);
if ~isempty(PGMs)
    FPA = PGMs{1}(1, 1);    % FPA ��ʼ�������
    nn = PGMs{1}(1, 2);
    final_path = PGMs{1};
else
    FPA = length(WM);
    nn = findingMNOC(WM(FPA), 1:length(WM(FPA).neuron));
    final_path = [FPA nn];
end
%% ��ǰ��
j = FPA;
for i = FPA-1: -1: 1
    [exist, locs] = isexistPA(WM(i));
    if exist    % �����PA��ѡPA������ѡMNOC
        min_n = findingMNOC(WM(i), locs);
    elseif ~isempty(WM(j).neuron(nn).precessor)
        temp_n = find(WM(j).neuron(nn).precessor(:, 1) == WMS(i));
        if ~isempty(temp_n)    % ��������
            temp_n = WM(j).neuron(nn).precessor(temp_n, 2);
            min_n = findingMNOC(WM(i), temp_n);
        else     % ����������
            min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
        end
    else      % ����������
        min_n = findingMNOC(WM(i), 1:length(WM(i).neuron));
    end
    final_path = [i min_n; final_path];
    j = i;
    nn = min_n;
end
%% ������
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
        if exist    % �����PA��ѡPA������ѡMNOC
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
%%  �ҳ���minicolumn��temp_n��Ԫ���������ٵ���Ԫindex
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
%% �ж������Ƿ����PA
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