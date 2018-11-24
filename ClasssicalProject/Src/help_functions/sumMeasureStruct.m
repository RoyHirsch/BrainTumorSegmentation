function measureStruct = sumMeasureStruct(measureStruct, numOfExamples)
    % a generic funtion to sum up data parameters in struct from kind -
    % 'measure struct'
    
    measureStruct.avarageDice = sum(measureStruct.diceArray) / numOfExamples;
    measureStruct.stdDice = std(measureStruct.diceArray);
    measureStruct.avarageSens = sum(measureStruct.sensitivityArray) / numOfExamples;
    measureStruct.avarageSpec = sum(measureStruct.specificityArray) / numOfExamples;
end