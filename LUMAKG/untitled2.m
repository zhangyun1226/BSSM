for s = 1: length(text)
    sentence = text{s};
    establishWMS;       % Establishing working minicolumn set;
    findingPGMs;        % Finding PGMs
    removeOverlap;      % Removing overlapping columns
    linkPA;             % Establishing a Sequence of Linked PA Neurons
    updateWeight;       % Updating weights;
    resetWM;            % Reset all neurons to inactive state;
end