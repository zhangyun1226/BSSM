%% remove overlapping 
% PGM -- 工作柱集合中的所有路径
for i = 1: length(PGMs)
    a = PGMs{i};
    if isempty(a)
        continue;
    end
    if size(a, 1) == 1          % 如果一条路径只剩1个柱，舍弃该路径
        PGMs{i} = [];
        continue
    end        
    a1 = PGMs{i}(:, 1);
    for j = i+1: length(PGMs)
        if isempty(a)
            break;
        end
        b1 = PGMs{j}(:, 1);
        c = intersect(a1, b1);     % 两条路径的重叠柱
        if ~isempty(c)   % 两条路径有重叠
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
    PGMs(cellfun(@isempty, PGMs)) = [];   % 删除空的路径，剩下的路径相互之间不重叠，且至少包含两个柱
end
%% 合并相邻路径,PGMs中的路径必须是从前到后排列，比如PGMs{1}为3、4、5柱，PGMs{2}为1、2柱，则需重新排列过。
i = 1;
temp_PGMs = PGMs;
while length(temp_PGMs) >= i+1
    a = temp_PGMs{i}(:, 1);
    b = temp_PGMs{i+1}(:, 1);
    L = length(a) + length(b);
    if b(end) - a(1) == L-1         % a的终点与b的起点相邻
        temp_PGMs{i} = [temp_PGMs{i}; temp_PGMs{i+1}];     % 合并两条路径
        temp_PGMs(i+1) = [];    % 从temp_PGMs中删除i+1路径
    else
        i = i + 1;
    end
end
PGMs = temp_PGMs;
        