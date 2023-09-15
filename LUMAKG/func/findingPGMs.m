%% Finding Nonoverlapping Sequences
% WMS  --  ��ǰ��������index
% WM   --  ��ǰ������
% M    --  ÿ��������Ԫ������
PGMs = [];
np = 0;         % num of path
for i = 1: length(WMS) - 1
    for m = 1: M
        if WM(i).neuron(m).flag == 1 && WM(i).neuron(m).state > 0   % ���ڵ�(i,m)ΪUA��PA,��δ��������
            [path, WM] = find_path(WM, WMS, i, m);
            if length(path) == 1 && size(path{1},1) == 1  % ֻ��1��·������·������Ϊ1����ֻ������ǰ�ڵ�(i,m)
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

%%  �ݹ麯��
function [path, WM] = find_path(WM, WMS, serial, nn)
if WM(serial).neuron(nn).flag == 0  % �ýڵ�������Ѿ�������,������·��
    path = WM(serial).neuron(nn).path;
    return
elseif isempty(WM(serial).neuron(nn).successor) || serial == length(WMS) % �ýڵ��޺�̣����ظýڵ�
    path{1} = [serial nn];
else
    row = find(WM(serial).neuron(nn).successor(:, 1) == WMS(serial+1));
    if isempty(row)             % �ýڵ�ĺ�̲���WM��
        path{1} = [serial nn];
    else
        np = 0;   % num of path
        suc = WM(serial).neuron(nn).successor;
        for k = 1: length(row)   
            if WM(serial+1).neuron(suc(row(k),2)).state == 1  % �к�̣����ú����ԪΪUA����ú������ȫΪUA
                np = 1;
                path{1} = [serial nn];
                break;
            elseif WM(serial+1).neuron(suc(row(k),2)).state == 2  % �к�̣��Ҹú����ԪΪPA�����õݹ�
                sub_path = find_path(WM, WMS, serial+1, suc(row(k),2));  % ���ض���·��
                for kk = 1: length(sub_path)
                    np = np + 1;
                    path{np} = [serial nn; sub_path{kk}];
                end
            else    % �к�̣����ú����Ԫδ������ú����ֻ�в���ΪPA��
                continue;
            end
        end
        if np == 0      % ����Ԫ�ĺ����Ԫ��δ������������PA������������Ԫ�ĺ����Ԫ��
            np = 1;
            path{np} = [serial nn];
        end
    end 
end
WM(serial).neuron(nn).flag = 0;     % �ýڵ�������ϣ�תΪ��������״̬
WM(serial).neuron(nn).path = path;
end