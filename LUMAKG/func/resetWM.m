%% Reset all neurons to inactive state
% WMS -- 当前工作柱的index
% column
for j = 1: length(WMS)
    for m = 1: M
        column(WMS(j)).neuron(m).flag = 1;
        column(WMS(j)).neuron(m).path = [];
        column(WMS(j)).neuron(m).state = 0;
        column(WMS(j)).neuron(m).time = [];
    end
end
clear WMS
