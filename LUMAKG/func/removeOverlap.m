%% remove overlapping 
% PGM -- �����������е�����·��
for i = 1: length(PGMs)
    a = PGMs{i};
    if isempty(a)
        continue;
    end
    if size(a, 1) == 1          % ���һ��·��ֻʣ1������������·��
        PGMs{i} = [];
        continue
    end        
    a1 = PGMs{i}(:, 1);
    for j = i+1: length(PGMs)
        if isempty(a)
            break;
        end
        b1 = PGMs{j}(:, 1);
        c = intersect(a1, b1);     % ����·�����ص���
        if ~isempty(c)   % ����·�����ص�
            if length(a1) >= length(b1)
                [~, loc] = ismember(c, b1);
                PGMs{j}(loc, :) = [];
            else
                [~, loc] = ismember(c, a1);
                a(loc, :) = [];
                a1(loc) = [];
            end
        end
    end
    PGMs{i} = a;
end
if ~isempty(PGMs)
    PGMs(cellfun(@isempty, PGMs)) = [];   % ɾ���յ�·����ʣ�µ�·���໥֮�䲻�ص��������ٰ���������
end
%% �ϲ�����·��,PGMs�е�·�������Ǵ�ǰ�������У�����PGMs{1}Ϊ3��4��5����PGMs{2}Ϊ1��2���������������й���
i = 1;
temp_PGMs = PGMs;
while length(temp_PGMs) >= i+1
    a = temp_PGMs{i}(:, 1);
    b = temp_PGMs{i+1}(:, 1);
    L = length(a) + length(b);
    if b(end) - a(1) == L-1         % a���յ���b���������
        temp_PGMs{i} = [temp_PGMs{i}; temp_PGMs{i+1}];     % �ϲ�����·��
        temp_PGMs(i+1) = [];    % ��temp_PGMs��ɾ��i+1·��
    else
        i = i + 1;
    end
end
PGMs = temp_PGMs;
        