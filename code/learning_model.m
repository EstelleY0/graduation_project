% 맵을 사용하여 클래스 이름을 인덱스로 변환
classNames = {'lane_violation_A', 'lane_violation_B', 'break_A', 'break_B','proper_timing_A', 'proper_timing_B', 'turnlight_A', 'turnlight_B'};
classMap = containers.Map(classNames, 1:numel(classNames));

% JSON 파일을 읽어들이고 클래스를 추출
jsonFile = 'labels.json'; % JSON 파일 경로를 실제 파일 경로로 업데이트
fid = fopen(jsonFile); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
labelData = jsondecode(str);

% 클래스를 숫자로 변환하고 특징과 함께 저장
allFeaturesCell = cell(1, numel(classNames));
allLabelsCell = cell(1, numel(classNames));

% Load the pre-trained GoogLeNet model
net = googlenet;
inputSize = net.Layers(1).InputSize;
featureLayer = 'pool5-7x7_s1';

videoNames = fieldnames(labelData);
for videoIdx = 1:numel(videoNames)
    fprintf("%d video processing\n", videoIdx);
    videoName = videoNames{videoIdx};
    classValues = labelData.(videoName);
    
    % 클래스 이름을 인덱스로 변환
    classIndices = zeros(1, numel(classNames)); % 클래스 인덱스 배열 초기화
    
    for classIdx = 1:numel(classNames)
        if ismember(classNames{classIdx}, classValues) % 클래스가 존재하는 경우
            classIndices(classIdx) = classMap(classNames{classIdx}); % 해당 인덱스 할당
        else
            classIndices(classIdx) = 0; % 매핑되지 않은 경우 0으로 설정
        end
    end
    
    % 특징 추출 및 저장
    videoFolder = fullfile('output', videoName);
    imds = imageDatastore(videoFolder, 'FileExtensions', {'.jpg', '.png', '.jpeg'}, 'LabelSource', 'none');
    numFrames = numel(imds.Files);
    videoFeaturesCell = cell(1, numel(classNames));
    
    for frameIdx = 1:numFrames
        img = readimage(imds, frameIdx);
        img = imresize(img, [inputSize(1), inputSize(2)]); % Resize to network input size
        feature = activations(net, img, featureLayer, 'OutputAs', 'rows');
        
        for classIdx = 1:numel(classNames)
            if classIndices(classIdx) ~= 0
                videoFeaturesCell{classIdx} = [videoFeaturesCell{classIdx}; feature];
            else
                videoFeaturesCell{classIdx} = [videoFeaturesCell{classIdx}; zeros(size(feature))];
            end
        end
    end
    
    for classIdx = 1:numel(classNames)
        allFeaturesCell{classIdx} = [allFeaturesCell{classIdx}; videoFeaturesCell{classIdx}];
        allLabelsCell{classIdx} = [allLabelsCell{classIdx}; repmat(classIndices(classIdx), size(videoFeaturesCell{classIdx}, 1), 1)];
    end
end

% Train and test LSTM models for each class
models = cell(1, numel(classNames));
accuracies = zeros(1, numel(classNames));

%%
for classIdx = 1:numel(classNames)
    trainFeatures = allFeaturesCell{classIdx};
    trainLabels = categorical(allLabelsCell{classIdx});
    
    % Reshape features to match LSTM input layer
    numFrames = size(trainFeatures, 1); % 시간 축이 첫 번째 차원이므로 trainFeatures의 3번째 차원을 사용
    numFeatures = size(trainFeatures, 3); % 특징 차원은 첫 번째 차원
    trainFeaturesReshaped = reshape(trainFeatures, [], 1, numFeatures); % 수정 후
    % Preallocate the trainFeaturesCell cell array
    trainFeaturesCell = cell(numFrames, 1);
    
    % Iterate over the third dimension of trainFeaturesReshaped
    for frameIdx = 1:numFrames %8400
        % Assign each slice to a cell in the trainFeaturesCell array
        trainFeaturesCell{frameIdx} = trainFeaturesReshaped( frameIdx, :, :);
    end

    % Define LSTM network architecture
    numHiddenUnits = 100;
    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(numel(unique(trainLabels)))
        softmaxLayer
        classificationLayer];
    
    % Train the LSTM network
    options = trainingOptions('adam', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.001, ...
        'GradientThreshold', 1, ...
        'Verbose', true);
    
    
    netLSTM = trainNetwork(trainFeaturesCell, trainLabels, layers, options); %traqinlabels: 8400by1
    models{classIdx} = netLSTM;
    
    % Test the LSTM network
    predictedLabels = classify(netLSTM, trainFeaturesCell);
    accuracies(classIdx) = sum(predictedLabels == trainLabels) / numel(trainLabels);
end

% Calculate overall accuracy
overallAccuracy = mean(accuracies);
disp(['Overall Accuracy: ', num2str(overallAccuracy)]);
