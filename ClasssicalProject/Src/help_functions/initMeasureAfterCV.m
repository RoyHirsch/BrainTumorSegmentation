function initMeasureAfterCV = initMeasureAfterCV(numOfExamples)
    % measure structer for comfortable data sharing

    initMeasureAfterCV = struct('diceArray',[],'sensitivityArray',[],'specificityArray',[],'avarageDice',[],'stdDice',[],'avarageSens',[],'avarageSpec',[]);
    initMeasureAfterCV.diceArray = zeros(numOfExamples, 1);
    initMeasureAfterCV.sensitivityArray = zeros(numOfExamples, 1);
    initMeasureAfterCV.specificityArray = zeros(numOfExamples, 1);
    
end