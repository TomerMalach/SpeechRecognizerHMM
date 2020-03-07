function [Accuracy] = evaluateRecognizer(data, HMMs, CBs, Fs, WindowsLength, MFCCs, Overlap, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames)

    Numbers = size(data, 1);
    Speakers = size(data, 2);
    
    Prediction = zeros(Numbers, Speakers);
    Accuracy = zeros(Numbers, 1);
    
    HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
    
    for num = 1:Numbers
        for speaker = 1:Speakers

            % Edge Detector
            [StartPoint, EndPoint] = edge_point_detect(data{num,speaker}, Fs, 0);
                
            % Framing
            %FramesSig = enframe(data{num,speaker}, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
            FramesSig = enframe(data{num,speaker}(StartPoint:EndPoint), NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrames = size(FramesSig, 1);
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrames, 1))';           
            
            coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), ...
                                 'OverlapLength', round(Fs*WindowsLength*Overlap), 'NumCoeffs', MFCCs));
                                     
            for i = 1:Numbers
                CbSize = size(CBs{i}, 2);
                
                DistancesToCenters = dist(coeffs, CBs{i});
                [d ,seq] = min(DistancesToCenters, [], 2);
                seq = transpose(seq);
                %seq = transpose(seq + (i-1)*CbSize);
                
                PI = HMMs{i, 1}; 
                A = HMMs{i, 2};
                B = HMMs{i, 3};
                
                logseq = dhmm_logprob(seq, PI, A, B);
                
                Prediction(i, speaker) = logseq;
                %Prediction(i, speaker) = sum(d);
            end
        end
        
        [~, argmin] = max(Prediction, [], 1);
        %[~, argmin] = min(Prediction, [], 1);
        Accuracy(num) = sum(argmin == num)/length(data(num, :));
        
        display(['Accuracy for number ' num2str(num-1) ' is ' num2str(Accuracy(num))] );
                
    end 

end

