function measureBeforCV = initMeasureBeforCV(numOfExamples)
    % measure structer for comfortable data sharing
    
    measureBeforCV = struct('diceArray',[],'sensitivityArray',[],'specificityArray',[],'avarageDice',[],'stdDice',[],'avarageSens',[],'avarageSpec',[]);
    measureBeforCV.diceArray = zeros(numOfExamples, 1);
    measureBeforCV.sensitivityArray = zeros(numOfExamples, 1);
    measureBeforCV.specificityArray = zeros(numOfExamples, 1);
    
end