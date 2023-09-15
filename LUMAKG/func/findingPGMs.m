%% Finding Nonoverlapping Sequences
% WMS  --  当前工作柱的index
% WM   --  当前工作柱
% M    --  每个柱中神经元的数量
PGMs = [];
np = 0;         % num of path
for i = 1: length(WMS) - 1
    for m = 1: M
        if WM(i).neuron(m).flag == 1 && WM(i).neuron(m).state > 0   % 根节点(i,m)为UA或PA,且未被搜索过
            [path, WM] = find_path(WM, WMS, i, m);
            if length(path) == 1 && size(path{1},1) == 1  % 只有1条路径，且路径长度为1，即只包含当前节点(i,m)
                continue;
            else
                for kk = 1: length(path)
                    np = np + 1;
                    PGMs{np} = path{kk};
                end
            end
        end
    end
end

%%  递归函数
function [path, WM] = find_path(WM, WMS, serial, nn)
if WM(serial).neuron(nn).flag == 0  % 该节点的子树已经搜索过,返回其路径
    path = WM(serial).neuron(nn).path;
    return
elseif isempty(WM(serial).neuron(nn).successor) || serial == length(WMS) % 该节点无后继，返回该节点
    path{1} = [serial nn];
else
    row = find(WM(serial).neuron(nn).successor(:, 1) == WMS(serial+1));
    if isempty(row)             % 该节点的后继不在WM内
        path{1} = [serial nn];
    else
        np = 0;   % num of path
        suc = WM(serial).neuron(nn).successor;
        for k = 1: length(row)   
            if WM(serial+1).neuron(suc(row(k),2)).state == 1  % 有后继，但该后继神经元为UA，则该后继柱必全为UA
                np = 1;
                path{1} = [serial nn];
                break;
            elseif WM(serial+1).neuron(suc(row(k),2)).state == 2  % 有后继，且该后继神经元为PA，调用递归
                sub_path = find_path(WM, WMS, serial+1, suc(row(k),2));  % 返回多条路径
                for kk = 1: length(sub_path)
                    np = np + 1;
                    path{np} = [serial nn; sub_path{kk}];
                end
            else    % 有后继，但该后继神经元未被激活（该后继柱只有部分为PA）
                continue;
            end
        end
        if np == 0      % 该神经元的后继神经元均未被激活（后继柱有PA，但是其它神经元的后继神经元）
            np = 1;
            path{np} = [serial nn];
        end
    end 
end
WM(serial).neuron(nn).flag = 0;     % 该节点搜索完毕，转为不可搜索状态
WM(serial).neuron(nn).path = path;
end