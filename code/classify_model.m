% Map class names to indices
classNames = {'lane_violation_A', 'lane_violation_B', 'break_A', 'break_B', 'proper_timing_A', 'proper_timing_B', 'turnlight_A', 'turnlight_B'};
classMap = containers.Map(classNames, 1:numel(classNames));

% Read JSON file and extract class information
jsonFile = 'labels.json'; % Update to the actual path of the JSON file
fid = fopen(jsonFile); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
labelData = jsondecode(str);

% Initialize cells to store features and labels for all classes
allFeaturesCell = cell(1, numel(classNames));
allLabelsCell = cell(1, numel(classNames));

% Load the pre-trained GoogLeNet model
net = googlenet;
inputSize = net.Layers(1).InputSize;
featureLayer = 'pool5-7x7_s1';

% Get the list of videos
videoNames = fieldnames(labelData);

% Select the last video for validation
validationVideoIdx = numel(videoNames);
validationVideoName = videoNames{validationVideoIdx};

% Initialize validation features and labels
validationFeatures = [];
validationLabels = [];

for videoIdx = 1:numel(videoNames)
    fprintf("%d video processing\n", videoIdx);
    videoName = videoNames{videoIdx};
    classValues = labelData.(videoName);
    
    % Convert class names to indices
    classIndices = zeros(1, numel(classNames));
    for classIdx = 1:numel(classNames)
        if ismember(classNames{classIdx}, classValues)
            classIndices(classIdx) = 1;
        else
            classIndices(classIdx) = 0;
        end
    end
    
    % Extract and store features
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
    
    % Store features and labels, separating validation data
    for classIdx = 1:numel(classNames)
        if videoIdx == validationVideoIdx
            validationFeatures = [validationFeatures; videoFeaturesCell{classIdx}];
            validationLabels = [validationLabels; repmat(classIndices(classIdx), size(videoFeaturesCell{classIdx}, 1), 1)];
        else
            allFeaturesCell{classIdx} = [allFeaturesCell{classIdx}; videoFeaturesCell{classIdx}];
            allLabelsCell{classIdx} = [allLabelsCell{classIdx}; repmat(classIndices(classIdx), size(videoFeaturesCell{classIdx}, 1), 1)];
        end
    end
end

%%

% Train and test LSTM models for each class
models = cell(1, numel(classNames));
accuracies = zeros(1, numel(classNames));

%%

for classIdx = 1:numel(classNames)
    trainFeatures = allFeaturesCell{classIdx};
    trainLabels = categorical(allLabelsCell{classIdx});
    
    % Reshape features to match LSTM input layer
    numFrames = size(trainFeatures, 1);
    numFeatures = size(trainFeatures, 2);
    trainFeaturesReshaped = reshape(trainFeatures, numFrames,numFeatures, 1);
    
    % Preallocate the trainFeaturesCell cell array
    trainFeaturesCell = cell(numFrames, 1);
    for frameIdx = 1:numFrames
        trainFeaturesCell{frameIdx} = trainFeaturesReshaped(frameIdx, :, :).';
    end

    % Define LSTM network architecture
    numHiddenUnits = 20;
    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(numel(unique(trainLabels)))
        softmaxLayer
        classificationLayer];

    % Create a validation set using a portion of the training data
    valIdx = randperm(size(trainFeaturesCell, 1), round(0.3 * size(trainFeaturesCell, 1))); % 20% of training data for validation
    trainIdx = setdiff(1:size(trainFeaturesCell, 1), valIdx);
    
    trainFeaturesCellTrain = trainFeaturesCell(trainIdx);
    trainLabelsTrain = trainLabels(trainIdx);
    trainFeaturesCellVal = trainFeaturesCell(valIdx);
    trainLabelsVal = trainLabels(valIdx);
    
    % Train the LSTM network with validation
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.0005, ...
        'GradientThreshold', 1, ...
        'Verbose', false, ...
        'ValidationData', {trainFeaturesCellVal, trainLabelsVal}, ... 
        'ValidationFrequency', 50, ... 
        'ValidationPatience', 5, ...
        'Plots', 'training-progress'); 
    
    netLSTM = trainNetwork(trainFeaturesCellTrain, trainLabelsTrain, layers, options);
    models{classIdx} = netLSTM;

    % Test the LSTM network on validation data
    numValFrames = size(validationFeatures, 1);
    validationFeaturesReshaped = reshape(validationFeatures, numValFrames, numFeatures, 1);
    validationFeaturesCell = cell(numValFrames, 1);
    for frameIdx = 1:numValFrames
        validationFeaturesCell{frameIdx} = validationFeaturesReshaped(frameIdx, :, :).';
    end
    validationLabelsCategorical = categorical(validationLabels);
    
    predictedLabels = classify(netLSTM, validationFeaturesCell);

    accuracies(classIdx) = sum(predictedLabels == validationLabelsCategorical) / numel(validationLabelsCategorical);
end

%%
% Calculate overall accuracy
overallAccuracy = mean(accuracies);
disp(['Overall Accuracy: ', num2str(overallAccuracy)]);


%%
% Save the models cell array
save('models.mat', 'models');
