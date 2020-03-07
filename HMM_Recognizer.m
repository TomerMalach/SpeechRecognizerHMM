clc;
close all;
clear all;


load data_training-test.mat

%% Load HMM package
addpath('HMM');
addpath('KPMstats');
%addpath('KPMtools');
% addpath('netlab3.3'); 


%% HMM Recognizer

Fs = 16384; % 16KHz

Training = 1;
TrainAccuracy = 0;
TestAccuracy = 1;


Overlap = 0.8;
MFCCs = 13;
WindowsLength = 40*10^-3; % msec

cbc = 14;
fonems = 6;

% bestValAcc = 0;
% bestValHP = [];
% HP = []; 

% fileID_all = fopen('logfile_all.txt','w');
% fileID_best = fopen('logfile_best.txt','w');

% for MFCCs = 13:2:19
%     for WindowsLength = 30*10^-3:10*10^-3:50*10^-3
%         for fonems = 2:1:8
%             for cbc = 12:2:40
%                 for kfold = 1:5

if TestAccuracy
    dataTrain = training_data;
    ValidationAccuracy = 0;
else
    % Cross varidation (train: 70%, val: 30%)
    cv = cvpartition(size(training_data, 2),'HoldOut',0.3);
    idx = cv.test;

    dataTrain = training_data(:, ~idx);
    dataVal  = training_data(:, idx);
    
    ValidationAccuracy = 1;
end
            
Numbers = size(dataTrain, 1);
Speakers = size(dataTrain, 2);

NumberOfSamplesAtEachWindow = round(Fs * WindowsLength); 
StepSizeBetweenFrames = round(Overlap * NumberOfSamplesAtEachWindow);
    
%             0 1 2 3 4 5 6 7 8 9
St = fonems* [4 4 4 5 4 5 2 4 5 4];
Cb = cbc*    [1 1 1 1 1 1 1 1 1 1];

if Training
    
    dataTrainMFCC = cell(Numbers, Speakers); % allocate tarin dataset MFCC
    CBs = cell(Numbers, 1);
    HMMs = cell(Numbers, 3);  % PI, A, B per number  
    Tol = 1e-3;
    
    MaxIter = 1000;
    Verbose = true;
    
    HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
    
    for num = 1:Numbers
    
        states = St(num);
        CbSize = Cb(num);
        
        % Extract MFCC for all training dataset
        TotalNumberOfFrames = 0;
        
        for speaker = 1:Speakers

            % Edge Detector
            [StartPoint, EndPoint] = edge_point_detect(dataTrain{num,speaker}, Fs, 0);
                
            % Framing
            FramesSig = enframe(dataTrain{num,speaker}(StartPoint:EndPoint), NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrames = size(FramesSig, 1);
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrames, 1))';           
            
            dataTrainMFCC{num, speaker} = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), ...
                                         'OverlapLength', round(Fs*WindowsLength*Overlap), 'NumCoeffs', MFCCs));
                                     
            TotalNumberOfFrames = TotalNumberOfFrames + NumberOfFrames;
        end
        
        
        % Generate CB
        
        CB_Data = zeros(MFCCs + 1, TotalNumberOfFrames);
        Offset = 1;
        for speaker = 1:Speakers
            CB_Data(:, Offset:Offset + size(dataTrainMFCC{num, speaker}, 2) - 1) = dataTrainMFCC{num, speaker};
            Offset = Offset + size(dataTrainMFCC{num, speaker}, 2);
        end
        
        CBs{num} = vqlbg(CB_Data, CbSize);
        
        
        % Train HMM

        disp('--------------------------------------------------');
        display(['Start HMM generation for number ' num2str(num-1)]);
        
        Seqs = cell(Speakers, 1);
        for speaker = 1:Speakers 
            DistancesToCenters = dist(dataTrainMFCC{num, speaker}, CBs{num});
            [~ ,seq] = min(DistancesToCenters, [], 2);
            Seqs{speaker} = transpose(seq);
        end
         
%         PI_EST = normalise(rand(states, 1));            % Initial first state probability 
%         A_EST = mk_stochastic(triu(rand(states, states)));    % Initial states transition probability 
%         B_EST = mk_stochastic(rand(states, CbSize));    % Initial emission probability 
        
        PI_EST = zeros(states, 1);            % Initial first state probability 
        A_EST = zeros(states, states);    % Initial states transition probability 
        B_EST = ones(states, CbSize)/states;    % Initial emission probability
        
        PI_EST(1:2) = [0.7 0.3];
        
        A_EST(1:states+1:end) = 0.75;
        A_EST(states+1:states+1:end) = 0.15;
        A_EST(2*states+1:states+1:end) = 0.1;
        A_EST(end-1:end, end-1:end) = [0.8 0.2; 0 1];
        
        [LL, PI_EST, A_EST, B_EST] = dhmm_em(Seqs, PI_EST, A_EST, B_EST, 'max_iter', MaxIter, 'thresh', Tol, 'verbose', 0);
        
        HMMs{num, 1} = PI_EST;
        HMMs{num, 2} = A_EST;
        HMMs{num, 3} = B_EST;
        
        display(['HMM for number ' num2str(num-1) ' is ready!']);
    end

end


%% Train Accuracy

if TrainAccuracy
    disp('---------------- Compute Train Accuracy ----------------');
    Accuracy = evaluateRecognizer(dataTrain, HMMs, CBs, Fs, WindowsLength, MFCCs, Overlap, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    disp('Train Accuracy:');
    disp(mean(Accuracy));
end


%% Validation Accuracy

if ValidationAccuracy
    disp('---------------- Compute Validation Accuracy ----------------');
    Accuracy = evaluateRecognizer(dataVal, HMMs, CBs, Fs, WindowsLength, MFCCs, Overlap, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    disp('Validation Accuracy:');
    disp(mean(Accuracy));
end

% HP = [HP; mean(Accuracy) cbc fonems WindowsLength MFCCs];
% fprintf(fileID_all,'%s\n', num2str([round(mean(Accuracy), 2) Tol CbSize states WindowsLength MFCCs Overlap]));
% if kfold == 5
%     if mean(HP(end-4:end, 1)) > bestValAcc
%         bestValAcc = mean(HP(end-4:end, 1));
%         bestValHP = [bestValHP; mean(HP(end-4:end, 1)) cbc fonems WindowsLength MFCCs];
%         fprintf(fileID_best,'%s\n', num2str([round(mean(HP(end-4:end, 1)), 2) cbc fonems WindowsLength MFCCs]));
%         if (mean(HP(end-4:end, 1)) > 0.85)
%             save(['HMM_CB_' datestr(now,'dd-mm-yy_HH-MM') '.mat'], 'HMMs', 'CBs');
%         end
%     end
% end
% 
%                 end
%             end
%         end
%     end
% end
% fclose(fileID_all);
% fclose(fileID_best);
% 
% display('BestHP');
% display('Accuracy | Tollerance | CB Size | States | Window Length | MFCCs | Overlap');
% display(bestValHP(end, :));


%% Test Accuracy

if TestAccuracy
    disp('---------------- Compute Test Accuracy ----------------');
    Accuracy = evaluateRecognizer(test_data, HMMs, CBs, Fs, WindowsLength, MFCCs, Overlap, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    disp('Test Accuracy:');
    disp(mean(Accuracy));
end

